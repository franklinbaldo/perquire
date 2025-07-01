## 🧭 Landscape: Who Else Is Decoding or Inspecting Embeddings?

### 1 . Direct decoding / inversion libraries
| Project | What it does | How it differs from Perquire | License / status |
|---------|--------------|------------------------------|------------------|
| **vec2text** | Trains a decoder to reconstruct the original sentence from Ada-002, BGE, etc. embeddings; includes pre-trained weights & Colab demo | One-shot prediction; no interactive Q-A or scoring loop | MIT · active |
| **TEIA (Transferable Embedding Inversion Attack)** | Treats inversion as a privacy attack; works even without access to the original model by using a surrogate | Security-focused PoC, not a user-facing explainer | Apache-2.0 · research |
| **Universal Zero-shot Embedding Inversion** (NAACL '25) | Diffusion-style decoder that hallucinates plausible text for *any* unknown embedding space | Great on out-of-domain vectors, still blind-guessing | Research code · very new |
| **GEIA** — “Sentence Embedding Leaks More Information” | Early generative attack recovering ≈ 70 % of tokens | Slower & older than vec2text | BSD-3 |

---

### 2 . Interactive-exploration dashboards  
*(Help humans poke around the space; don’t reverse the vector)*

| Tool | Highlights | Why Perquire still matters |
|------|------------|----------------------------|
| **Google PAIR-code LIT** | Live saliency, counter-factual probes, projection plots for text/image/tabular embeddings; web UI & notebooks | Great for *seeing* clusters; useless for narrating a lone rogue vector |
| **TensorBoard Embedding Projector** | Classic 2-D/3-D PCA, t-SNE, UMAP with nearest-neighbor search | Same story—visual insight only, no semantic guessing |
| **Stable-Diffusion Embedding Inspector** (A1111 plugin) | Lets artists mix & compare SD textual-inversion embeddings | Image-domain only; different user base |

---

### 3 . Security / privacy red-team kits
| Project | Angle | Note |
|---------|-------|------|
| **hacking-vectors** | Cookbook for embedding-inversion + membership/attribute inference attacks | Shows how easy PII leaks from vectors can be |
| **IronCore Labs demo** | Blog + code walk-through of Li et al. attack | Good reference for legal/privacy discussions |

---

#### Key take-aways
1. **Perquire’s differentiator** is the *interactive “twenty-questions” loop* that refines hypotheses and scores them against the original vector.  
2. **Visual tools ≠ explanations.** LIT & Projector are fantastic for sanity-checking datasets, but they can’t answer *why* one orphan vector popped out.  
3. **Privacy attacks are accelerating.** Highlight defensive monitoring—e.g., alert when vec2text-style reconstructions cross a similarity threshold.

> **TL;DR:** vec2text & TEIA are the closest functional peers; dashboards are complementary; the rest is academic or red-team fodder.
