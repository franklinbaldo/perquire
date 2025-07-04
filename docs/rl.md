Below is a _battle-plan_ for grafting a **reinforcement-learning “Interrogator”** onto Perquire. It’s ruthlessly scoped: you get measurable gains in cache-efficiency without turning the repo into an RL zoo.

---

## 1 ️⃣ Pick the right RL framing — **contextual bandit, not full MDP**

| Option                                    | Why it _seems_ attractive                                                             | Why it’s over-kill                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Full policy-gradient (PPO/DDPG)           | Handles long credit assignment                                                        | Needs thousands of roll-outs → expensive, slow       |
| Q-learning on dialog states               | Fine-grained control loop                                                             | Same sample-inefficiency + huge state explosion      |
| **Contextual bandit (LinUCB / Thompson)** | _One-shot decision per step_, provable regret bounds, trivial to warm-start from logs | Reward must be immediate (info-gain ↑, cache-miss ↓) |

> **Verdict:** go _bandit_. Papers in 2025 already use it to route queries across LLMs with solid regret guarantees. See Poon et al. 2025 and MixLLM (HF) for precedents. ([arxiv.org][1], [huggingface.co][2])

---

## 2 ️⃣ Define _state_, _action_, _reward_

| Component        | Concrete design                                                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **State 𝑆ₜ**     | 〈embedding-hash, running-similarity Δ, last-N questions, answer-embedding sim〉 flattened into a 256-D feature vector (cheap to store, cache-friendly) |
| **Action 𝐴ₜ**    | Pick one template from your existing `questions.yaml` (≈ 50 actions). Future: allow “parametric” questions by adding slots and discretizing.            |
| **Reward 𝑅ₜ**    | `(sim_gain – λ·cache_miss)`, where `sim_gain = cos_sim(pred, target) – cos_sim(prev_pred, target)`; set λ≈0.05 so a cache miss wipes modest gains.      |
| **Cache signal** | Simply flag if the template/answer pair already exists in redis ↦ reward-penalty, no extra plumbing.                                                    |

---

## 3 ️⃣ **Offline warm-start → online bandit**

1. **Instrument** current loop to log {state, action, reward} tuples.
2. **Warm-start** with **LinTS** (linear Thompson Sampling) from those logs. One epoch over 10 k tuples is enough to avoid the cold-start pain.
3. **Online**: every live interrogation step updates the posterior (O(A·d²)). No separate training service needed.

You now learn _which_ templates hit the cache and boost similarity fastest for each embedding “region”.

---

## 4 ️⃣ Training & evaluation checklist

- **Dataset sanity:** drop any session that solved in < 3 steps (bandit reward is zero-variance).
- **A/B test:** old heuristic vs. bandit on the STS-B + MS MARCO dev split (see suggestion #2 earlier). Track _median_ steps-to-90 %-sim.
- **Drift guard:** if the posterior covariance trace explodes → fallback to heuristic for that session.

---

## 5 ️⃣ Skeleton patch (add to `perquire/rl_bandit.py`)

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class BanditConfig:
    dim: int = 256
    n_actions: int = 50
    prior_var: float = 1.0
    noise_var: float = 0.1
    cache_penalty: float = 0.05

class LinTS:
    def __init__(self, cfg: BanditConfig):
        self.cfg = cfg
        self.A = np.eye(cfg.dim) * cfg.prior_var      # d × d
        self.b = np.zeros((cfg.n_actions, cfg.dim))   # K × d
        self.post_mean = np.zeros((cfg.n_actions, cfg.dim))

    def select(self, state_vec: np.ndarray) -> int:
        theta_samp = np.random.multivariate_normal(np.zeros(self.cfg.dim), self.A)
        scores = self.post_mean @ state_vec + theta_samp @ state_vec
        return int(np.argmax(scores))

    def update(self, action: int, state_vec: np.ndarray, reward: float):
        self.A += np.outer(state_vec, state_vec)
        self.b[action] += reward * state_vec
        self.post_mean[action] = np.linalg.solve(self.A, self.b[action])
```

Plug it into your loop:

```python
bandit = LinTS(BanditConfig())
...
state = featurize(session)            # → np.ndarray shape (256,)
action_id = bandit.select(state)
question = templates[action_id]
answer, cache_hit, sim_gain = ask(question)
reward = sim_gain - cfg.cache_penalty * (not cache_hit)
bandit.update(action_id, state, reward)
```

_(Keep it dead simple; you can swap in a torch-based contextual bandit later.)_

---

## 6 ️⃣ Milestone timeline

| Week | Deliverable                                                   |
| ---- | ------------------------------------------------------------- |
| 1    | Logging patch + 10 k offline tuples                           |
| 2    | LinTS implemented, offline replay notebook                    |
| 3    | A/B test harness + Grafana dashboard                          |
| 4    | Ship behind a `--rl` flag, default off                        |
| 6    | Default _on_ after passing regression gate (< -5 % perf risk) |

---

### Final take

You don’t need a swarm of PPO agents—just a lean contextual bandit that _learns which question template pays off fastest_ given the current similarity landscape and cache state. It’s two files of code, one Grafana panel, and a real differentiator no competitor ships yet.

[1]: https://arxiv.org/abs/2506.17670?utm_source=chatgpt.com "Online Multi-LLM Selection via Contextual Bandits under ... - arXiv"

Below is a **self-contained “RL-weights versioning” blueprint** you can paste straight into `docs/rl_weights.md` (or a new section in `architecture.md`).
Everything sits inside the fence so it travels cleanly through PR reviews.

````markdown
## 📦 RL-Weights Snapshot & Versioning Scheme

### 1. Naming & Layout

| Object        | Example key                              | Why                                               |
| ------------- | ---------------------------------------- | ------------------------------------------------- |
| **Snapshot**  | `rl/weights/v1.0.3/lin_ts_state.zst`     | Semantic tag → easy human diff                    |
| **Signature** | `rl/weights/v1.0.3/lin_ts_state.zst.sig` | Detached Ed25519 for tamper-proofing              |
| **Metadata**  | `rl/weights/manifest.jsonl`              | Pointer list with SHA-256, ETag, size, created_at |

**Manifest line**

```json
{
  "version": "1.0.3",
  "url": "https://archive.org/download/perquire-cache/rl/weights/v1.0.3/lin_ts_state.zst",
  "sha256": "ab…cd",
  "git_commit": "8f4c2b1",
  "d": 256,
  "k": 50,
  "samples": 187_432,
  "created_at": "2025-07-01T18:00:12Z"
}
```
````

---

### 2. Publisher workflow (CI job, runs nightly)

```bash
# merge local deltas ➜ new posterior
python -m perquire.rl.merge_deltas --out lin_ts_state.zst

# sign + upload
gpg --detach-sign --armor lin_ts_state.zst               # .sig
aws --endpoint https://s3.us.archive.org \
    s3 cp lin_ts_state.zst        s3://perquire-cache/rl/weights/v$TAG/
aws s3 cp lin_ts_state.zst.sig    s3://perquire-cache/rl/weights/v$TAG/

# append manifest entry atomically
python scripts/update_manifest.py \
       --ver $TAG --file lin_ts_state.zst --sig lin_ts_state.zst.sig
```

`$TAG` = `$(git describe --tags --abbrev=0)` → `1.0.3`.

---

### 3. Consumer boot sequence

```python
from perquire.rl.loader import fetch_latest_weights

weights = fetch_latest_weights(
    manifest_url="https://archive.org/download/perquire-cache/rl/weights/manifest.jsonl",
    max_age_days=7,         # fail-soft if offline
    verify_sig=True
)
bandit.load_state(weights)
```

_If offline or signature/commit mismatch:_ fall back to zero-init → warm-start from local cache.

---

### 4. Version bump rules

1. **Patch (x.y.Δ)** – same algorithm & hyper-params, just more data.
2. **Minor (x.Δ.z)** – hyper-param change (d, k) or bug-fix in featurizer.
3. **Major (Δ.y.z)** – new RL algorithm → incompatible state shape (e.g., switch from LinTS → neural-bandit).

Clients ignore snapshots whose **major** ≠ compiled-in `RL_STATE_MAJOR`.

---

### 5. Governance cheatsheet

| Guard-rail | Action                                                                          |     |     |     |                                           |
| ---------- | ------------------------------------------------------------------------------- | --- | --- | --- | ----------------------------------------- |
| Tampering  | Ed25519 sig; reject on verify fail                                              |     |     |     |                                           |
| Rollback   | Clients store N previous snapshots; if newest corrupt, fall back                |     |     |     |                                           |
| Drift      | Snapshot size or                                                                |     | A   |     | ₂ deviates >2× median → CI aborts publish |
| Audit      | Each manifest entry keeps `git_commit`; you can reproduce the weights from logs |     |     |     |                                           |

> **TL;DR** — one immutable folder per semantic version, nightly CI publishes a signed snapshot and updates a single manifest line. The client always knows _exactly_ which posterior generated a decision and can roll back or reproduce on demand.

```

```

[2]: https://huggingface.co/papers?q=Query-Mixup&utm_source=chatgpt.com "Daily Papers - Hugging Face"
