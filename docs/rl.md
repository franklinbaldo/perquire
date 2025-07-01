Below is a *battle-plan* for grafting a **reinforcement-learning “Interrogator”** onto Perquire.  It’s ruthlessly scoped: you get measurable gains in cache-efficiency without turning the repo into an RL zoo.

---

## 1 ️⃣  Pick the right RL framing — **contextual bandit, not full MDP**

| Option                                    | Why it *seems* attractive                                                             | Why it’s over-kill                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Full policy-gradient (PPO/DDPG)           | Handles long credit assignment                                                        | Needs thousands of roll-outs → expensive, slow       |
| Q-learning on dialog states               | Fine-grained control loop                                                             | Same sample-inefficiency + huge state explosion      |
| **Contextual bandit (LinUCB / Thompson)** | *One-shot decision per step*, provable regret bounds, trivial to warm-start from logs | Reward must be immediate (info-gain ↑, cache-miss ↓) |

> **Verdict:** go *bandit*.  Papers in 2025 already use it to route queries across LLMs with solid regret guarantees.  See Poon et al. 2025 and MixLLM (HF) for precedents. ([arxiv.org][1], [huggingface.co][2])

---

## 2 ️⃣  Define *state*, *action*, *reward*

| Component        | Concrete design                                                                                                                                       |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **State 𝑆ₜ**    | 〈embedding-hash, running-similarity Δ, last-N questions, answer-embedding sim〉 flattened into a 256-D feature vector (cheap to store, cache-friendly) |
| **Action 𝐴ₜ**   | Pick one template from your existing `questions.yaml` (≈ 50 actions).  Future: allow “parametric” questions by adding slots and discretizing.         |
| **Reward 𝑅ₜ**   | `(sim_gain – λ·cache_miss)`, where `sim_gain = cos_sim(pred, target) – cos_sim(prev_pred, target)`; set λ≈0.05 so a cache miss wipes modest gains.    |
| **Cache signal** | Simply flag if the template/answer pair already exists in redis ↦ reward-penalty, no extra plumbing.                                                  |

---

## 3 ️⃣  **Offline warm-start → online bandit**

1. **Instrument** current loop to log {state, action, reward} tuples.
2. **Warm-start** with **LinTS** (linear Thompson Sampling) from those logs.  One epoch over 10 k tuples is enough to avoid the cold-start pain.
3. **Online**: every live interrogation step updates the posterior (O(A·d²)).  No separate training service needed.

You now learn *which* templates hit the cache and boost similarity fastest for each embedding “region”.

---

## 4 ️⃣  Training & evaluation checklist

* **Dataset sanity:**  drop any session that solved in < 3 steps (bandit reward is zero-variance).
* **A/B test:** old heuristic vs. bandit on the STS-B + MS MARCO dev split (see suggestion #2 earlier).  Track *median* steps-to-90 %-sim.
* **Drift guard:** if the posterior covariance trace explodes → fallback to heuristic for that session.

---

## 5 ️⃣  Skeleton patch (add to `perquire/rl_bandit.py`)

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

*(Keep it dead simple; you can swap in a torch-based contextual bandit later.)*

---

## 6 ️⃣  Milestone timeline

| Week | Deliverable                                                   |
| ---- | ------------------------------------------------------------- |
| 1    | Logging patch + 10 k offline tuples                           |
| 2    | LinTS implemented, offline replay notebook                    |
| 3    | A/B test harness + Grafana dashboard                          |
| 4    | Ship behind a `--rl` flag, default off                        |
| 6    | Default *on* after passing regression gate (< -5 % perf risk) |

---

### Final take

You don’t need a swarm of PPO agents—just a lean contextual bandit that *learns which question template pays off fastest* given the current similarity landscape and cache state.  It’s two files of code, one Grafana panel, and a real differentiator no competitor ships yet.

[1]: https://arxiv.org/abs/2506.17670?utm_source=chatgpt.com "Online Multi-LLM Selection via Contextual Bandits under ... - arXiv"
[2]: https://huggingface.co/papers?q=Query-Mixup&utm_source=chatgpt.com "Daily Papers - Hugging Face"
