# Attention Q-Net — Exploration & Experiment Log

Results for the explainable attention q-net (`attention_model.py`) built on the `explainability` branch:
a token-based, action-in q-model (`f(state, action) → q`) with per-head CLS→token attention as the
interpretability readout. This file records the two studies run to (1) profile its cost as an MCTS leaf
evaluator and (2) sweep its capacity (depth `L`, FFN width, and the head-to-head playing strength each buys).

## References (the value-weighted attribution follows these closely)

- Kobayashi, Kuribayashi, Yokoi & Inui (EMNLP 2020), **"Attention is Not Only a Weight: Analyzing
  Transformers with Vector Norms"** — attention output is `Σⱼ A[i,j]·f(xⱼ)` with `f(x)=(xW_v)W_o`, so
  importance = the *norm of the attention-weighted transformed value*, not `A` alone (our unsigned cousin).
  https://aclanthology.org/2020.emnlp-main.574/  (code: https://github.com/gorokoba560/norm-analysis-of-transformer)
- Elhage, Nanda, Olsson et al. (Anthropic, 2021), **"A Mathematical Framework for Transformer Circuits"** —
  the OV circuit (`W_V·W_O` = the "what content moves the output" linear map, attention = the mixing
  weights) and direct logit attribution; our signed `Δq(j) = Σₕ Aʰ[0][j]·(uʰ·vⱼʰ)` is DLA specialized to a
  scalar value head. Web-native article (no arXiv/official PDF):
  https://transformer-circuits.pub/2021/framework/index.html

## Setup (common to all results)

- **Game:** 2-player ImposterKings.
- **Dataset:** `datasets/tensors/k20l3_tokens.npz` — **240,499 token rows** post-processed from ~2,000
  self-play games (log-once/derive-many; one row per *(state, candidate action)*). Rows are **game-split**
  into train/val (val_frac 0.1) to avoid leakage; loss is **visit-share-weighted MSE** on the mover-relative
  q label. Note rows are heavily correlated (few independent games), so effective samples ≪ 240k.
- **Model default:** `AttnConfig(d_model=64, n_heads=4, ffn_hidden=128, n_layers=1, dropout=0.1, bounded=True)`.
  Explicit-action encoding ("prong B", `ACTION_DIM=51`); `CARD_DIM=43/44`. Trained with Adam, batch 1024,
  lr 1e-3, up to 50 epochs, early stopping patience 5.
- **Head-to-head protocol:** the net is used as the **MCTS leaf evaluator + policy prior** at budget
  `hybrid(20,3)` ("@k20"), played against a **vanilla-MCTS@k20** opponent (pure random rollouts) over
  **100 mirrored deals = 200 games** (each deal played twice with swapped seats; independent per-orientation
  play-rng). Parallelized over 10 workers. **Winrate is a lower-bound-style signal vs one specific opponent,
  not an exploitability bound** — >50% means "the net's guidance beats raw rollouts at k20."

---

## Study 1 — Profiling the NN-MCTS hot path

10 games each of `<challenger> vs vanilla-MCTS@k20`, serial single-thread under cProfile (`torch` at 1 thread).

| config | wall (10 games) | leaf-eval cost | rollout cost | NN eval vs a rollout |
|---|---:|---:|---:|---|
| **no-NN** (rollout both sides) | 273.5 s | — (rollout *is* the leaf) | 0.89 ms/rollout | — |
| **MLP-MCTS** | **215.0 s** | **0.41 ms**/eval | 0.93 ms/rollout | **2.3× cheaper** than a rollout |
| **attention-MCTS** | **468.5 s** | **3.24 ms**/eval | 0.98 ms/rollout | **3.3× more expensive** than a rollout |

**Leaf-work decomposition** (the leaf value comes from a *rollout* on the vanilla side and from the *net* on
the challenger side; NN-MCTS does **not** roll out — the eval replaces the rollout):

| config | vanilla side (rollouts) | challenger side (leaf method) | total leaf work |
|---|---|---|---:|
| no-NN | 187,584 × 0.89 ms = 166 s | *also rollouts* (both sides) | **166 s** |
| MLP-MCTS | 86,700 × 0.93 ms = 81 s | 75,122 evals × 0.41 ms = 31 s | **112 s** |
| attention-MCTS | 86,820 × 0.98 ms = 85 s | 83,320 evals × 3.24 ms = 270 s | **355 s** |

**Attention eval breakdown** (`_leaf_value` = 274 s of 451 s profiled = **61% of the whole run**), split ≈50/50:
- **`tokenize` — 117 s (43% of eval):** Python featurization *per candidate move* — `_card_token` 20 s,
  `state.with_` 29 s, **`determinize` 27 s over 536,508 calls**.
- **torch `forward` — 115 s (43% of eval):** `nn.linear` dispatch 30 s, **`collate` 21 s**, `layer_norm` 8.5 s.
- **Redundant determinize confirmed:** ~**6.4 determinize/leaf** (attention) vs ~**2/leaf** (MLP) — the
  evaluate re-tokenizes every legal move and each `tokenize` re-runs `legal_moves → determinize`.

**Verdict:** swapping a rollout for the **MLP** eval *speeds the game up* (215 s < 273 s) — the numpy eval is
cheaper than the playout it replaces. The **attention** eval is ~3× a rollout, so attention-MCTS is ~1.7×
slower than pure rollout. The ~7× per-leaf gap over MLP is ~50% Python featurization overhead (redundant
determinize + per-move tokenize + collate) and ~50% torch dispatch — recoverable with a numpy forward +
determinize-once-per-leaf (out of scope; now quantified).

---

## Study 2 — Capacity sweep (depth L, FFN width) + head-to-head strength

All variants trained on the same `k20l3_tokens.npz` (no rebuild). **Winrate = vs vanilla-MCTS@k20, 100
mirrored deals (200 games)**, reported as **winrate (± 95% CI)**.

| variant | config (d/L/ffn) | params | train-time | epochs | val_mse | top1_bestq | recall@2 | spearman | **winrate (± 95% CI)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **v2** (deploy) | 64 / **1** / 128 | 42,945 | 2018 s | 50 | 0.0647 | 51.3% | 75.8% | 0.434 | **57.5% (± 5.6%)** |
| **v3a** | 64 / **2** / 128 | 78,209 | 2331 s | 19 (best 14) | 0.0646 | 51.7% | 76.3% | 0.437 | **55.5% (± 5.2%)** |
| **v3b** | 64 / 1 / **256** | 61,249 | 643 s | 9 (best 4) | 0.0756 | 48.3% | 74.5% | 0.388 | **51.0% (± 5.9%)** |
| **v3c** | 64 / **2** / **256** | 111,233 | 2088 s | 17 (best 12) | **0.0628** | 51.2% | **76.6%** | 0.422 | **62.0% (± 5.4%)** |
| **MLP-256** (baseline) | mlp 256-wide | 55,809 | n/a | n/a | 0.0704 | **53.4%** | 76.8% | 0.464 | **57.0% (± 5.7%)** |

Raw head-to-head detail (wins / 200, split-rate = fraction of deals split 1–1, CPU-seconds over 10 workers):

| variant | wins | winrate | ci95 | split% | CPU-s |
|---|---:|---:|---:|---:|---:|
| v2 (attention) | 115/200 | 0.575 | ±0.056 | 65% | 22,662 |
| v3a | 111/200 | 0.555 | ±0.052 | 71% | 23,248 |
| v3b | 102/200 | 0.510 | ±0.059 | 64% | 21,228 |
| v3c | 124/200 | 0.620 | ±0.054 | 64% | 25,147 |
| MLP-256 | 114/200 | 0.570 | ±0.057 | 64% | 7,504 |

### Findings

1. **The capacity knobs interact.** Depth alone (v3a) doesn't help; **wide-FFN alone (v3b) hurts** (worst on
   both proxies and play, 51.0%); but **depth + width together (v3c) is best** on val_mse *and* winrate
   (62.0%). So "more capacity" only paid off when both moved together.
2. **v3c vs v2 is suggestive, not conclusive:** +4.5 pp point estimate but the CIs overlap
   ([56.6, 67.4] vs [51.9, 63.1]). v3c *does* clear v3b cleanly. A confirmation run (more deals, or a direct
   v3c-vs-v2 match) would settle it.
3. **Proxies (val_mse/top1/recall) barely separate the L2 variants** (~0.063 val_mse, ~51% top1, ~76%
   recall) yet winrate spread is real (51%→62%) — proxies and playing strength don't always agree
   (MLP-256 has the *highest* top1 at 53.4% but only mid-pack winrate).
4. **Overfitting, not underfitting:** the train/val gap widens with capacity (v2 +0.0007 → v3a +0.0062 →
   v3c +0.0071 → v3b diverges early), consistent with ~240k *correlated* rows (few independent games).
5. **Cost:** attention plays at MLP-equivalent strength but at ~3× the CPU (22.7k vs 7.5k CPU-s for the same
   200 games) — the price of the per-leaf torch forward (Study 1).

### Deploy decision

**v2 `models/attn_d64_L1.pt` (L1)** is the deployed hint model for the explainability UI — simplest, proven,
cheapest eval. At L1 only the **CLS row (row 0)** causally feeds q (card↔card rows are computed-but-discarded),
so the visualization renders row-0 primary. **v3c (L2)** is the standby upgrade: strongest *and* it makes the
card↔card attention rows causal (layer-1 → layer-2 CLS), so swapping to it later buys richer explanations at
(likely) no strength cost — pending a confirmation run.

---

## Metric glossary

- **winrate** — fraction of games the challenger wins vs vanilla-MCTS@k20 (mirrored; 200 games). ± is the
  **95% confidence interval** (`1.96·s/√n` over per-deal scores); overlapping CIs ⇒ difference not significant.
- **top1_bestq** — fraction of decisions where the net's argmax-q move equals the search's best (highest-visit)
  move. **recall@2** — fraction where the search's best move is in the net's top-2 by q. **spearman** — rank
  correlation between the net's q ordering and the search's visit ordering over legal moves.
- **val_mse** — visit-share-weighted MSE of the predicted q on the held-out (game-split) validation set;
  baseline (predict-the-mean) ≈ 0.258.
- **params** — total learnable parameters. **train-time** — wall-clock to train (CPU, torch 2.12.1+cpu).
- **eval cost / rollout cost** (Study 1) — mean ms per NN leaf evaluation vs per random rollout playout.

---

## Value-weighted attribution (the math behind the drawer's "signed" mode)

The attention drawer's raw-attention view shows `A[0][j]` = *where CLS looked*, which is **not** *how much
token j moved the recommendation* — attention drops the value vector. The "signed" mode fixes this with a
**value-weighted attribution**, computed post-hoc on the trained model (no retrain). Below is the full
derivation with shapes. Config: **d = d_model = 64, H = n_heads = 4, dₕ = d/H = 16**, sequence length **S**
(= `1 CLS + N cards + 3` context tokens).

### Shapes

| symbol | meaning | shape |
|---|---|---|
| `X` | readout-layer input (one row per token, post-embedding) | `[S, 64]` |
| `W_q, W_k, W_v, W_o` | the four learned `nn.Linear(64,64)` maps | `[64, 64]` |
| `w_head` | value head weight (`nn.Linear(64,1)`) | `[64]` |
| `Qʰ, Kʰ, Vʰ` | per-head query/key/value projections | `[S, 16]` |
| `Aʰ` | attention matrix, head h (`softmax` rows) | `[S, S]` |
| `o₀ʰ` / `o₀` | CLS attention output, per head / concatenated | `[16]` / `[64]` |
| `u = w_head·W_o` | readout direction (fixed), split into `uʰ` | `[64]` / `[16]` |
| `c_jʰ = uʰ·vⱼʰ` | value-to-readout alignment | scalar |
| `Δq(j) = Σₕ Aʰ[0][j]·c_jʰ` | token j's signed contribution | scalar |

### Forward pass (where each weight acts)

1. Project (after LayerNorm): `qᵢ = W_q x̃ᵢ`, `kⱼ = W_k x̃ⱼ`, `vⱼ = W_v x̃ⱼ`, each `[64]` → split to heads `[16]`.
2. **`W_q` and `W_k` build the attention** — the *only* place `W_q` enters:
   `Sʰ[i,j] = (qᵢʰ·kⱼʰ)/√16`, `Aʰ = softmax_j(Sʰ)`. `Aʰ` decides *where CLS looks*; it carries no value content.
3. Pool + project + read out:
   `o₀ʰ = Σⱼ Aʰ[0][j] vⱼʰ`, `o₀ = concat_h`, `a₀ = W_o o₀`,
   `CLS_out = x₀(residual) + a₀(attention) + FFN`, `q_logit = w_head·CLS_out`, `q = tanh(q_logit)`.

### Decomposing q into per-token contributions

The attention path's share of the logit: `w_head·a₀ = (w_head·W_o)·o₀ = u·o₀`. Expanding `o₀`:

`w_head·a₀ = Σₕ uʰ·(Σⱼ Aʰ[0][j] vⱼʰ) = Σⱼ Σₕ Aʰ[0][j] (uʰ·vⱼʰ) = Σⱼ Σₕ Aʰ[0][j] c_jʰ`.

So **`c_jʰ = uʰ·vⱼʰ`** — token j's **value vector** (from `W_v`) dotted with the **readout direction** `uʰ`
(from `W_o` and `w_head`). It is the signed amount j's content would move the logit *per unit attention*;
it does **not** involve `W_q`. The realized contribution multiplies it by the actual attention:

`Δq(j) = Σₕ Aʰ[0][j]·c_jʰ`  —  `A` is the "how much CLS looks" (W_q/W_k) half, `c_j` is the "how much j's
content moves q" (W_v/W_o/w_head) half.

### q vs dq
- **q** = `tanh(q_logit)` = the model's scalar value estimate (the drawer header + the tooltip's `weight` is
  the raw attention). One output number.
- **dq (Δq)** = a **contribution/decomposition** term (delta, not a derivative): token j's share of the
  logit. Green = raised q, red = lowered it.
- Caveats: `Σⱼ Δq(j)` is only the **attention-path** slice of `q_logit` (the residual `x₀` and the FFN also
  contribute, un-attributed); it is attributed to the **pre-tanh logit**, so the effect on the bounded `q`
  scales by `1 − q²` — monotone, so signs/rankings are exact, only magnitudes rescale. It is also a
  **last-layer** attribution (ignores earlier layers at L≥2). **Biases:** `b_v` is inside each `vⱼ` and is
  correctly attributed per token, but `b_o`/`b_head` are token-independent constants — `Σⱼ Δq(j)` matches
  the attention-path logit share up to `w_head·b_o + b_head`, which lives in the un-attributed constant
  slice (cannot belong to any token; signs/rankings unaffected).

### Code
`explain(view, action, model, attribution=True)` (`machine_learning/explain.py`) returns
`row0_signed [H,S]` (= `Aʰ[0][j]·c_jʰ`) and `attribution [S]` (head-summed `Δq(j)`); it reads the value
vectors via `AttentionModel.forward_layers(..., need_values=True)` and computes
`u = head.weight[0] @ layers[-1].attn.wo.weight`. Verified by the reference-equivalence test in
`tests/test_explain.py` (`row0_signed.sum()` equals an independent `w_head·(W_o·o₀_cls)`).

**Empirical payoff:** on a position where raw attention was dominated by the `board` token, the signed view
ranks `my_hand:Assassin` first (+0.25) and `board` only third (−0.14) — the cards that actually swing the
value surface once the value vector is included.

# Math scratch
Per attn head h , for each j-th token,
$$s_j = \frac{q_0^h \cdot k_j^h}{\sqrt{16}} ;;\text{(scalar per } j\text{)}, \qquad A^h[0][\cdot] = \text{softmax}_j(s_j) ;\in [1\times S]$$

$$out_0^h = (\sum_{j} A^h[0][j] ) v_j^h \qquad [1\times16]$$
v_j^h is the 1x16 value vector

- concat 4 heads: out₀ = [out₀¹ ; out₀² ; out₀³ ; out₀⁴] → [1×64] ; subscript 0 for 0th token?
- output projection: a₀ = out₀ @ W_o → [1×64] ; a_0 for attention
- residual + FFN: CLS_out = x₀ + a₀ + FFN(...) → [1×64]
- value head: q_logit = CLS_out · w_head (scalar), Q = tanh(q_logit)

Matrix form (if it helps)

The same line as one matrix product: o₀ʰ = Aʰ[0] @ Vʰ where Aʰ[0] is [1×S] and Vʰ is [S×16] → result [1×S] @ [S×16] = [1×16]. ✓ (And doing all rows at once is Aʰ @ Vʰ = [S×S]@[S×16] = [S×16] — every token gets its own blended output; we only read row 0's.)

Step 1 — what A @ V is

$$Out = A @ V \qquad [S\times S],@,[S\times16] = [S\times16]$$

This is called the attention output (of head h) — sometimes "context vectors." There's no single universal name; in our code it's literally the variable out (attention_model.py line 77: out = (attn @ v)...).

Step 2 — concatenate the 4 heads back together
Out_h is [Sx16] but there are 4 heads so we concatenate O = [O_1 | O_2 | O_3 | O_4] is Sx64

Step 3 — the output projection W_o

$$A_{out} = O_{cat} ,@, W_o \qquad [S\times64],@,[64\times64] = [S\times64]$$


W_o is a learned mixer: the four heads computed their blends independently, and W_o recombines them into one coherent 64-dim update per token.

Break both matrices into rows:

$$X = \begin{bmatrix} x_1 \ x_2 \ \vdots \ x_S \end{bmatrix}, \qquad A_{out} = \begin{bmatrix} a_1 \ a_2 \ \vdots \ a_S \end{bmatrix} \qquad \text{each row } [1\times64]$$

Then

$$X' = X + A_{out} = \begin{bmatrix} x_1 + a_1 \ x_2 + a_2 \ \vdots \ x_S + a_S \end{bmatrix}$$

$$a_i = O_{cat}[i] ,@, W_o \qquad [1\times64],@,[64\times64] = [1\times64]$$
This ends the attention sublayer.

Step 4 — residual add (attention's contribution joins the stream)

$$X' = X + A_{out} \qquad [S\times64]$$

Each token's representation is updated by its attention result, not replaced. (This is the residual stream from our earlier diagram.)

Step 5 — the FFN sublayer (per token, own residual)

$$X'' = X' + \text{FFN}(\text{LN}(X')) \qquad [S\times64]$$

LN = LayerNorm (a per-token "recentering")

LN takes one token's 64-vector and standardizes it across its own 64 features:

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

where, for that single row x [1×64]:
- μ = mean of its 64 numbers (scalar), σ = their std (scalar) → (x−μ)/σ has mean 0, std 1
- γ, β = learned [64] scale-and-shift vectors (so the network can undo/adjust the normalization per feature); ⊙ = elementwise

The FFN (64→128→GELU→64) touches each row independently — no cross-token mixing here.

Step 6 — the readout (last layer only)

Take row 0 only (CLS) of the final X'': (we still have 1 last linear layer with weights and bias)

$$q_{logit} = X''[0] \cdot w_{head} + b_{head} \quad(\text{scalar}), \qquad Q = \tanh(q_{logit})$$

[1×64] · [64×1] → scalar. That scalar is the value the drawer displays. Rows 1..S−1 of X'' are computed but never read (at L=1) — that's the "card rows are discarded" point from before.

Swap trick,

From step 6 and step 4/5, CLS's final vector is a sum of three pieces:

$$q_{logit} = \big(\underbrace{x_0}{\text{residual}} + \underbrace{a_0}{\text{attention}} + \underbrace{f_0}{\text{FFN}}\big)\cdot w_{head} + b_{head}$$

q_logit splits into x₀·w_head + a₀·w_head + f₀·w_head + b_head. Only the a₀·w_head piece came through attention

Move 1: collapse the linear tail into one vector u

From step 3, a₀ = o₀ @ W_o. So:

$$a_0 \cdot w_{head} = (o_0 @ W_o)\cdot w_{head} = o_0 \cdot \underbrace{(w_{head} ,@, W_o^\top)}_{u;[1\times64]}$$

Two linear maps in a row (W_o, then dot with w_head) are just one dot with a precomputed vector u. Split u into head blocks like everything else: u = [u¹|u²|u³|u⁴], each [1×16]. Then, since o₀ is the head-concat:

$$a_0\cdot w_{head} = \sum_{h=1}^{4} o_0^h \cdot u^h$$

In "physics" terms I think of `u` as a kind of good-ness measurement. since w_head is learnt via NN training that means w_head learnt weights to align input to output example. Since `u` is derived from w_head and W_o it is some measurement of goodness according to its alignemnt to the dataset.

Move 2: THE swap

Substitute step 1's row-0 blend o₀ʰ = Σⱼ Aʰ[0][j]·v_jʰ and use one algebra rule — a dot product distributes over a weighted sum:

$$\sum_h \Big(\sum_j A^h[0][j], v_j^h\Big)\cdot u^h ;=; \sum_j \sum_h A^h[0][j],\underbrace{\big(v_j^h\cdot u^h\big)}_{c_j^h \text{ (scalar)}} ;=; \sum_j \Delta q(j)$$

That's the entire trick. Forward order: blend the vectors first (Σⱼ), dot with u last → one total. Attribution order: dot each token's value vector with u first (c_jʰ), blend the scalars with the same attention weights → the same total, but it arrives pre-split per token j.

Now we have mathematically massaged a term `c_j^h = v_j^h . u^`. This can be viewed as a projection - the contribution `c` of the j-th token is its value `v` projected onto `u` the 'goodness' axis.

The micro-example, both orders (same numbers as before, plus u = [1, 0.5, −1])

A[0] = [0.7, 0.2, 0.1], v₁=[1,0,2], v₂=[0,10,0], v₃=[−5,0,1].

Forward order (steps 1→6): o₀ = [0.2, 2.0, 1.5] (computed earlier), then
o₀·u = 0.2·1 + 2.0·0.5 + 1.5·(−1) = **−0.3**. One number, no idea who caused it.

Swapped order: project each ingredient first:
c₁ = v₁·u = 1 + 0 − 2  = −1
c₂ = v₂·u = 0 + 5 + 0  = +5
c₃ = v₃·u = −5 + 0 − 1 = −6
then blend the scalars: Δq = [0.7·(−1), 0.2·(+5), 0.1·(−6)] = [−0.7, +1.0, −0.6], sum = −0.3 ✓.

Write the attention-path logit share as one explicit contraction (per head, indices j = token, t = the 16 feature dims):
$$a_0\cdot w_{head}\big|h ;=; \sum{j}\sum_{t} A^h[0,j]; V^h[j,t]; u^h[t]$$

In einsum notation: einsum("j,jt,t->", A0, V, u). It's a double contraction, and the "trick" is nothing but choosing which index to contract first:

- contract j first: (A0 @ V) · u → build o₀, then project → the forward order
- contract t first: A0 · (V @ u) → build c_j = v_j·u per token, then blend → the attribution order

Associativity of contraction. Same tensor, two evaluation orders, one of which happens to leave the answer factored per token.

And the bra-ket analogy is genuinely apt, not just vibes: with c_j = ⟨u|v_j⟩,

$$\langle u | o_0\rangle = \Big\langle u \Big| \sum_j A_j, v_j \Big\rangle = \sum_j A_j ,\langle u | v_j\rangle$$

— a linear functional distributing over a superposition, exactly the manipulation you'd do expanding a state in components and evaluating an overlap term-by-term. ⟨u| is the fixed "measurement" (the readout direction), the |v_j⟩ are the components, A_j the mixture weights, and the attribution is just reading the expectation before collapsing the sum. The place the analogy (and the method) hard-stops is the same place linearity stops: the softmax that made A and the FFN/tanh are nonlinear, so no bra-ket games survive passing through them — which is precisely why the attribution is scoped to "attention path, last layer, pre-tanh."
