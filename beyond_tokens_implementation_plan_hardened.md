# Beyond Tokens — Hardened Implementation Plan (v7-aligned)

**Executable protocol for the v7 diagnostic framework, with the construct-validity guardrail enforced as the controlling design principle.**

*This plan supersedes the prior five-phase implementation draft. That draft was technically competent but inverted v7's central commitment: it installed the robustness-engineered interventions (ASEntmax, KV-Shifting Attention, Adaptive Gradient Gating, embedding-specific learning rates, and the Advective Diffusion Transformer backbone) as **defaults**, which collapses the construct — several of those interventions manufacture the very properties the diagnostics exist to detect, making the headline tests circular. This version reverts all five to **labeled ablation arms**, pulls two training-time diagnostics out of the objective, restores the encoder ladder and the encoding sub-study, makes frequency-matched update budgets a default, restores the scaling-divergence tiers, corrects the NL templates, and corrects the efficiency accounting. A correction-to-critique map is in Appendix A.*

---

## 0. Governing Principle: The Construct-Validity Guardrail

The object of measurement is **identity behavior in a bare transformer-class architecture**. Exactly one configuration family — the **bare baseline** (Phase II) — is permitted to support Tier-1 *primitivity* claims. Every architectural hardening technique enters only as an **ablation arm** answering the bounded question "does the identity effect survive under X?", always run alongside the bare baseline, and each ablation carries an explicit ledger of the diagnostics it **may** and **may not** report (Phase III, Ablation Register).

Two classes of control are distinguished and treated oppositely:

- **Non-architectural controls** (compute-matched dummy prefix, frequency-matched update budgets, geometry covariate tracking, shuffled-signal control, held-out-diagnostic constraint, two-stage error decomposition). These change *what is measured*, not the substrate under test. **Adopted as defaults.**
- **Architectural interventions** (sparse attention, PDE backbones, gradient gating, embedding-specific LRs, KV-shifting, training-time invariance/debiasing objectives). These change the substrate and several are circular with the tests. **Declined as defaults; admitted only as ablations.**

No diagnostic quantity (IPI, attention entropy, anisotropy, decision invariance) may ever enter a training objective, auxiliary loss, or regularizer. This is a hard protocol invariant.

---

## Phase I — Synthetic Graph and Corpus Generation

### I.1 Topological families across three scale tiers

To support **both** cross-topology transfer and the scaling-divergence test, every topology family is instantiated at three node budgets. Mean degree ⟨k⟩ is held constant across tiers and families so that family is the only structural variable; edge budget scales with N.

| Tier | N (entities) | M (triples) at ⟨k⟩≈25 |
|---|---|---|
| Small | 10,000 | 250,000 |
| Medium | 100,000 | 2,500,000 |
| Large | 1,000,000 | 25,000,000 |

Families (each at all three tiers):
- **Scale-free $G_{SF}$** — Barabási–Albert preferential attachment; power-law $P(k)\propto k^{-\gamma}$, $\gamma\in[2.0,3.0]$. Hubs + sparse periphery.
- **Small-world $G_{SW}$** — Watts–Strogatz; high clustering $C$, $d\propto\log n$.
- **Erdős–Rényi $G_{ER}$** — uniform random, $p=\langle k\rangle/N$, Poisson degree; the uniform-degree control.

Relations: worksFor, owns, locatedIn, manages, reportsTo (counts scaled per tier). Surface forms are **synthetic nonce strings** from a reserved partition, zero corpus presence.

### I.2 Random-walk corpus generation **with frequency-matched update budgets** *(correction)*

Walks translate topology into sequences of $(e_i, r_k, e_j)$ transitions. The prior plan *profiled* the scale-free frequency imbalance but left it in; this is the ST3 confound (degree-induced embedding degeneration) that would make any cross-topology transfer failure uninterpretable. We **match per-node update budgets** as a Tier-0 default:

- Use **degree-corrected / importance-reweighted walks** (or stratified uniform node sampling) so that the **per-node visit count — and therefore the per-embedding gradient-update count — is equalized across nodes and across topology families**, to within a logged tolerance.
- **Log per-node and per-token update counts** for every run; these enter the analysis as a verification that matching held.
- Effect: a peripheral node in $G_{SF}$ receives the same number of gradient updates as a typical node in $G_{SW}$, so cross-topology transfer tests *structural-rule learning* rather than which nodes happened to be trained. Imbalance profiling is retained as a *reported diagnostic*, not as the training regime.

### I.3 Natural-language arm — **relational templates** matched to the edges *(correction)*

The prior plan used CausalFlip-style *causal* templates ("a change in $e_i$ leads to a change in $e_j$", "an increase in $e_i$ causes an increase in $e_j$"), but the relation set (worksFor, owns, locatedIn, manages, reportsTo) is **non-causal and static** — there is no interventional semantics to flip. We replace these with relation-appropriate **relational** templates that preserve the binary-classification target and the Default/Alternative pairing needed for decision invariance:

- **Default ($D$, interrogative):** "Does {$e_i$} {relation-verb} {$e_j$}? Yes / No." (e.g., "Does {$e_i$} work for {$e_j$}?")
- **Alternative ($A$, declarative):** "{$e_i$} {relation-verb} {$e_j$}. Correct / Incorrect?"

Relation-verb realizations are fixed per relation type (worksFor→"works for", owns→"owns", locatedIn→"is located in", manages→"manages", reportsTo→"reports to"). Negative examples are generated by edge corruption with controlled plausibility. The $D$/$A$ pair over a fixed fact licenses the **decision-invariance** measurement (Phase V) without the graph-vs-question semantic mismatch.

---

## Phase II — The Bare Baseline (substrate under test)

This is the only configuration family permitted to support primitivity claims. It contains **no architectural hardening**.

### II.1 Backbone
- **Standard decoder-only Transformer** with **dense softmax attention**. No KV-shifting, no entmax/sparse attention, no PDE/diffusion backbone.
- Trained **from random initialization** on the synthetic corpus (zero pretraining contamination).
- **Single global learning rate** $\eta_\theta$ for all parameters, including embeddings. No embedding-specific LR.
- No adaptive gradient gating; standard backpropagation.

### II.2 Compute-matching
- **Matched parameter count** (within the limits architectural differences across configs allow) and **matched training FLOPs and token budget** across all configurations. All reported.

### II.3 Compute-matched dummy prefix — **hardened specification** *(correction)*
Prepending $M$ identity/memory tokens raises sequence length $L\to L+M$, expanding the KV cache and supplying **attention sinks** that stabilize the residual stream independent of content. To length-match, **every baseline configuration (LW, TR, LL) is trained and evaluated with $M$ prepended dummy tokens**. The dummy tokens are specified so they cannot become a covert identity channel:

- Drawn from a **reserved vocabulary partition $V_{dummy}$, disjoint from both $V_{lexical}$ and $V_{entity}$**.
- The specific IDs are **resampled per sequence** from a large reserved pool, so no single dummy embedding can be optimized to carry task structure.
- Dummy embeddings are **frozen at random initialization**; dummy positions **never appear as prediction targets**.
- Result: dummy positions contribute length, KV-cache footprint, and attention-sink anchoring — and nothing else. All EE/KV efficiency and accuracy contrasts are taken against `LW+dummy`, never bare `LW`.

---

## Phase III — Experimental Matrix and the Ablation Register

### III.1 Primary configuration matrix (restored to identifiability) *(correction)*

The prior plan dropped the encoder ladder and the encoding sub-study, which removes Axis-3 attribution and the tokenization gate. Both are restored.

**Core conditions** (all on the bare baseline, all with the matched dummy prefix where applicable):

| Config | Encoding | Memory | Forcing | Role |
|---|---|---|---|---|
| LW | E0 lexical subword | parametric | none | baseline |
| EE | E2 atomic token | parametric | none | identity, parametric |
| TR | E0 lexical subword | non-parametric RAG | none | retrieval, no identity |
| KV | E2 atomic token | non-parametric slot | none | identity, externalized |
| HRL | E2 atomic token | non-parametric slot | ID-restricted retrieval | forcing: identity mandatory |
| LL | E0 lexical subword | non-parametric RAG | lexical-surface lock | forcing: lexical mandatory |
| LW+dummy | E0 + M dummy prefix | parametric | none | length-matched efficiency/accuracy baseline |

**Encoder ladder (Axis 3 — structural reliance), restored:**

| Config | Description |
|---|---|
| G | graph encoder only (GraphSAGE / GAT / TransE / RotatE); permutation-equivariant; IPI ceiling |
| G+probe | linear probe over frozen graph embeddings; no LLM |
| G+frozen-LLM | LLM consumes graph output, internal weights frozen |

**Encoding sub-study (tokenization-artifact gate), restored:** within LW (B0, parametric), vary only the encoding axis — **E0** (surface) → **E1** (ID string, *control not treatment*) → **E2** (atomic token). The contrast E1−E0 vs E2−E0 isolates the tokenization artifact; the "identity is not a relabeling" claim depends on this cell.

### III.2 The Ablation Register *(core correction — the executable guardrail)*

Each architectural intervention the prior plan made a default is here an **isolated ablation arm**, run *in addition to* the bare baseline. For each, the register fixes (a) what it changes, (b) the configuration it is paired against, (c) the diagnostics it **MAY** report, and (d) the diagnostics it **MUST NOT** report, because the intervention is engineered to produce the property that diagnostic measures (circularity). Primitivity verdicts are read **only** from the bare baseline; ablations answer only "does the identity effect survive under X?"

| Ablation arm | Changes | Paired against | MAY report | MUST NOT report (circular) |
|---|---|---|---|---|
| **A1 — ASEntmax / sparse attention** | dense softmax → adaptive α-entmax | bare baseline | survival of the identity main effect; reasoning accuracy under sparse attention | **permutation invariance (mode 8), anisotropy/effective-rank (mode 11)** — entmax is engineered against rank collapse |
| **A2 — Advective Diffusion Transformer (ADiT) + TAR + BES** | graph backbone → PDE advection-diffusion + topology-aware reweighting + boundary shaping | bare baseline (graph-grounded configs) | survival of the identity main effect under a topology-robust backbone | **cross-topology transfer (mode 5 / structural-rule attribution)** — ADiT is built to generalize under topological shift; using it to run that test is circular |
| **A3 — Adaptive Gradient Gating (AGG)** | gate repulsive CE-gradient components on bottom-20% rare tokens | bare baseline | survival of the identity effect when rare-token degeneration is mitigated | **embedding-degeneration covariate / per-node norm (mode 13)** and any result conditioned on it — AGG engineers away the degeneration the covariate observes |
| **A4 — Embedding-specific LR** | $\eta_E = 2.5\,\eta_\theta$ for the embedding layer | bare baseline | convergence-speed / grokking-onset ablation | **degeneration covariate (mode 13); scaling-divergence slope (mode 7)** — embedding-LR is confounded with the optimization-dynamics the covariate and slope measure |
| **A5 — KV-Shifting Attention** | unbind K/V projections to ease induction-head formation | bare baseline | in-context-learning speed ablation | **contextual reasoning gains (the in-context / topology-in-context results)** — KV-shifting accelerates the very mechanism those measure |
| **A6 — Permutation-aware optimization (post-training)** | RL/optimization toward permutation invariance during tuning | bare baseline | downstream-task robustness ablation | **permutation invariance / IPI (mode 8, 9)** — trains toward the criterion IPI independently tests; IPI is **not reported** on this arm |
| **A7 — Causal-invariant learning (post-training debiasing)** | causal-invariant objective during instruction tuning | bare baseline | downstream-task robustness ablation | **decision invariance / template-shift robustness (Phase V)** — trains toward the criterion decision invariance tests |

Notes: the **gradient-detachment** mechanism is meaningful *only inside arms that contain gating modules* (it detaches diagnostic features from the gating gradient). In the bare baseline there is no gating to detach, and no diagnostic enters the objective at all, so the held-out invariant holds trivially. SMoE / sparse-expert routing is **out of scope** — there is no MoE in the testbed; it would enter only as a further ablation if an MoE variant were ever introduced, and its all-to-all communication costs are not charged to the baseline.

---

## Phase IV — Mechanistic Interpretability and Diagnostic Controls (defaults)

These are **non-architectural** and apply to every configuration as evaluation-only covariates. None enters any training objective.

### IV.1 Representational-geometry tracking (Tier-0 covariate)
At every 1,000 optimization steps, on the active entity-embedding matrix:
- **Average pairwise cosine similarity** (anisotropy).
- **Uniformity loss** (narrow-cone collapse detector).
- **Singular-value decomposition** → **singular-value decay curve** → **effective rank** (number of singular values explaining 99% of variance).

These are carried as covariates into the mixed-effects model. Purpose: high IPI accompanied by high anisotropy and sharp spectral decay is **rank collapse (mode 11)**, not abstract invariance; cross-topology failure accompanied by collapsed peripheral-node norms is **embedding degeneration (mode 13)**, not rule failure.

### IV.2 Shuffled-signal control (default, evaluation-only)
At inference, compute the exact attention-entropy/variance statistics, then **randomly permute them across positions**. If an apparent advantage persists under shuffling, it is an artifact of the general distribution of attention statistics rather than position-aligned structural coordination.

### IV.3 Anchor–explorer partitioning (covariate, evaluation-only)
Partition tokens per forward pass by normalized attention entropy $H_t^{norm}=H_t^{raw}/\log N_t$: **anchor** = bottom 20%, **explorer** = top 20%. Track QA accuracy separately across partitions to separate gradient-volatility effects from identity processing. This is a reporting covariate; it does not gate or train anything.

### IV.4 Two-stage error decomposition (default)
For every task: `Total error = ER error (Stage 1: surface→correct ID) + reasoning error given correct ID (Stage 2)`, reported as stacked bars so resolution failures are never charged to reasoning. Run under oracle and fallible resolution; fallible resolver evaluated in- and out-of-distribution across the noise grid $N(\rho,H)$.

### IV.5 Held-out-diagnostic constraint (protocol invariant)
No diagnostic — IPI, anisotropy, attention entropy, decision invariance — may enter any loss, regularizer, or RL objective. Verified per run as a Tier-0 gate.

---

## Phase V — Testing, Verification, Evaluation

### V.1 Unseen-entity and cross-topology generalization
- **Unseen entities:** pretrain on $E_1$–$E_{1{,}000{,}000}$, evaluate on $E_{1{,}000{,}001}$–$E_{1{,}100{,}000}$ over identical topology (zero-shot identifier handling).
- **Cross-topology transfer:** train on $G_{SF}$ walks, evaluate on $G_{SW}$, and vice versa, **under the frequency-matched update budgets of I.2**, with **per-node embedding norm carried as a covariate**. The primitivity reading is taken **only on the bare baseline**; arm A2 (ADiT) may not report this test. A transfer failure is attributed to structural-rule failure only after confirming it does **not** track peripheral-node norm collapse (mode 13).

### V.2 Anti-shortcut and decision-invariance probes (evaluation-only)
- **Graph interdiction:** (A) **anti-statistical queries** where local lexical co-occurrence predicts the wrong answer and only path traversal the right one; (B) **counterfactual edge swaps** (5–10% of edges swapped at inference, text identical); (C) **path masking** (remove 1–2 critical hops, forcing long-range traversal). Accuracy that holds under interdiction iff reasoning is genuine.
- **Template shifts / decision invariance:** evaluate the NL arm across Default ($D$) and Alternative ($A$) templates; measure the percentage of samples whose prediction is unchanged under template swap. Credit the co-primary NL identity effect **only if it survives the shuffled-signal and template controls**. Post-training debiasing (A6, A7) is **ablation-only** and does **not** report IPI or decision invariance respectively.

### V.3 Permutation invariance with the five-part sufficiency conjunction
On the bare baseline: `IPI = 1 − Var_π[score]/Var_ref` over $K$ relabelings (inference-time for non-parametric configs; cross-seed for parametric), with G as the invariance ceiling. A config earns "computational identity" only if **all** hold: (1) IPI ≥ τ (≥0.8× G's ceiling) on long-hop unseen-entity reasoning; (2) high identity-discriminative accuracy; (3) HRL load-bearing vs LL; (4) interdiction-robust; (5) **low anisotropy** (IV.1), so invariance is not rank collapse. Arms A1 and A6 may not report IPI.

### V.4 Efficiency: Pareto accounting — **corrected** *(correction)*
- **Primary (hardware-independent):** accuracy/FLOP and accuracy/GB curves, computed against **`LW+dummy`** (length-matched), not bare LW.
- **Secondary (named hardware):** wall-clock latency, GPU memory-bandwidth utilization, and KV-cache allocation on the target cluster (e.g., NVIDIA A100 80GB, B200 180GB), reported as named-hardware context so the Pareto claim does not silently depend on a specific KV-cache regime.
- **Reporting rule, not a science gate:** an identity advantage is reported as **unconditional** only if it is a statistically significant Pareto improvement over `LW+dummy` on the hardware-independent curves; otherwise it is reported as a **tradeoff point with the multiplier stated**. Efficiency does **not** gate Tier-1 *validity* — a genuine reasoning effect that costs more is a finding, not a disqualification. (This corrects the prior plan's requirement that Tier-1 validation demand a systems-level Pareto win.)

### V.5 Tier-0 validity-gate checklist (read before any conclusion)
For every reported contrast, confirm: geometry covariates within bounds for the relevant claim; per-node update counts matched (I.2); efficiency taken against `LW+dummy` (II.3, V.4); no diagnostic entered training (IV.5); primitivity read on the bare baseline only, never an ablation arm.

### V.6 Decision-rule mapping
Outcomes map to the v7 decision rules, including the verdicts: H₀ (null + HRL collapse + no non-trivial IPI gap); H₁ (identity main effect synthetic **and** NL post-controls + sufficiency conjunction incl. low anisotropy + HRL load-bearing + full≫G + stable/growing scaling + cost stated vs LW+dummy); H₂ (only D+G clear ablation/interdiction on long-hop + U-shaped scaling); and the failure verdicts — tokenization artifact (E1−E0 only), encoder's computation (full≈G), memorization (IPI≈LW), identity ignored (high IPI + low discriminative accuracy), **rank collapse** (high IPI + high anisotropy), **length artifact** (gain vanishes vs LW+dummy), **embedding degeneration** (transfer failure tracks peripheral-norm collapse), storage-not-reasoning (flat benefit vector), shortcut (interdiction doesn't degrade), representational convenience (gap shrinks with scale), nonce-not-language (NL effect absent or fails controls), cost-dominated (no Pareto gain).

---

## Appendix A — Correction-to-Critique Map

| Critique point | Correction implemented |
|---|---|
| **Fatal: guardrail inverted** — ASEntmax, KV-Shifting, AGG, embedding-LR, ADiT installed as defaults; cross-topology test circular on ADiT. | All five reverted to **ablation arms** (A1–A5) with explicit MAY/MUST-NOT diagnostic ledgers; **bare baseline** (Phase II) is the only substrate for primitivity claims (Phase 0, III.2). |
| **Held-out constraint violated** — permutation-aware optimization and causal-invariant learning train toward the criteria IPI/decision-invariance test. | Both demoted to **ablation-only** (A6, A7); IPI not reported on A6, decision invariance not reported on A7; held-out-diagnostic constraint stated as a protocol invariant (IV.5, V.5). |
| **Matrix dropped identifiability** — no encoder ladder, no encoding sub-study. | **G / G+probe / G+frozen-LLM** restored (Axis 3); **E0→E1→E2** encoding sub-study restored (tokenization gate) (III.1). |
| **Dummy-prefix underspecified** — "random" tokens could leak into the identity channel. | Hardened spec: **reserved partition disjoint from lexical/entity vocab, resampled per sequence, frozen random embeddings, never targets** (II.3). |
| **NL templates causal-mismatched** — causal flip templates over non-causal static relations. | Replaced with **relation-matched D/A templates** preserving the decision-invariance pairing (I.3). |
| **Scaling-divergence tiers missing** — only the 100k tier instantiated. | **10k / 100k / 1M tiers** instantiated for all families; slope analysis restored (I.1). |
| **Frequency-matching absent** — imbalance profiled but left in (ST3 confound). | **Frequency-matched update budgets** made a Tier-0 default via degree-corrected/importance-reweighted walks; per-node update counts logged (I.2). |
| **Hardware-entangled efficiency** — A100/B200 wall-clock as the metric. | acc/FLOP and acc/GB made **primary (hardware-independent)**; wall-clock/bandwidth **secondary, named-hardware** (V.4). |
| **Efficiency over-gating** — Tier-1 validity required a systems-level Pareto win. | Efficiency made a **reported axis with stated tradeoff, not a pass/fail gate** on the science; unconditional-vs-tradeoff reporting rule (V.4). |
