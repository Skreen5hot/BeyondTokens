# Beyond Tokens: A Diagnostic Framework for Evaluating Explicit Identity as a Computational Primitive in Knowledge Systems

**A Registered-Report Framework — Version 7 (Measurement-Hardened)**

*Status: pre-registration / registered report. This version supersedes v6. It responds to an adversarial stress-test that attacked the framework not at the level of experimental design (handled in v3–v6) but at the level of **measurement validity** — arguing that the diagnostic instruments can read high or low for reasons rooted in optimization dynamics and representation geometry rather than identity. v7 adopts the valid controls (a representation-degeneration covariate, frequency-matched update budgets, a length-matched efficiency baseline, an evaluation-only/held-out-diagnostic constraint, and natural-language shuffled-signal/template controls) and adds an explicit **construct-validity guardrail**: the framework measures identity behavior in the bare architecture under test, and robustness-engineered variants enter only as labeled ablations, never as defaults — because several of the stress-test's proposed architectural fixes would otherwise produce the very robustness the diagnostics exist to detect. Consolidated changelog with explicit adopt/decline rationale: Appendix C.*

---

## Abstract

Large Language Models represent knowledge as statistical associations among lexical tokens, producing well-documented failures of entity persistence, alias resolution, consistency, updating, and relational reasoning. This paper asks whether **explicit identity is a superior computational primitive for factual knowledge than the lexical token**, and contributes a **diagnostic framework** for answering such questions in any knowledge-grounded system: every plausible explanation for an observed performance difference is given a distinct empirical signature, so an ambiguous result becomes structurally impossible.

The framework's reusable instruments — an Identity × Memory factorial, symmetric forcing functions (Hard Reference Lock and Lexical Lock), permutation invariance with an explicit sufficiency conjunction, graph-interdiction probes, two-stage error decomposition, an encoder-isolation ablation ladder, a distribution-shift suite with cross-topology transfer, an identity-benefit vector, a scaling-divergence test, and an efficiency axis — are organized around three pillars: **(1) is explicit identity computationally useful beyond storage; (2) can permutation invariance detect genuine identity use; (3) does identity improve reasoning or only memory.** A three-tier claim hierarchy keeps the apparatus subordinate to the question.

v7 hardens the framework against measurement confounds. A **representation-degeneration covariate** (cosine anisotropy, singular-value spectrum, per-node embedding norm) is carried on every identity, invariance, and transfer result, converting "the metric might be confounded by representational collapse" into "we measure collapse and condition on it." **Frequency-matched update budgets** neutralize degree-induced embedding degeneration at its source. A **length-matched efficiency baseline** (dummy-token prepending, with wall-clock and memory-bandwidth measurement) ensures an apparent identity gain is not an attention-sink or sequence-length artifact. All diagnostics are **evaluation-only and never training signals**, closing the gameability loophole. The natural-language arm adds **shuffled-signal and template-variation controls**. A **construct-validity guardrail** fixes the substrate under test as a bare transformer-class architecture; topology-robust backbones, sparse attention, and anti-shortcut supervision are admitted only as ablations answering "does the identity effect survive under X," never as the default that would silently dissolve the construct.

Empirical predictions stand independently of any philosophical commitment (firewalled to motivation and an appendix). The synthetic world is sense-degenerate for internal validity, with a co-primary natural-language outcome. A pre-registered analysis plan maps every fingerprint — including "advantage is memorization," "identity is ignored," "invariance is rank collapse," "gain is an attention-sink artifact," "transfer failure is embedding degeneration," and "gain does not survive cost" — to a decision rule, including the null.

---

## 1. Introduction

In a standard LLM, *Aaron works for DHS* becomes lexical tokens in a space where identity is only an emergent distributional regularity; a knowledge graph makes it `(E1452, worksFor, E893)`, where identity is explicit and persistent. The question:

> **Does explicit identity provide measurable computational advantages over lexical identity — for *reasoning*, not merely storage and retrieval?**

The methodological insight, developed across versions, is that in a system this entangled the same accuracy number can be produced by many mechanisms. v3–v6 separated the *experimental-design* mechanisms (identity used vs bypassed; reasoning vs shortcut; reasoning vs resolution error; the language model vs the encoder; real effect vs scale- or cost-vanishing effect). v7 adds the *measurement-validity* mechanisms a sophisticated reviewer raises next: an instrument can read "identity" when the true driver is optimization dynamics (rare-token stagnation, embedding degeneration) or representation geometry (rank collapse, attention sinks). The contribution remains a **diagnostic framework** that gives each mechanism a distinct fingerprint; the HDT-grounded architecture is its testbed.

We call the testbed family **HDT-Grounded Language Models (HGLMs)**.

---

## 2. Claim Hierarchy, Scope, and Construct-Validity Commitment

**The three pillars:** (1) Is explicit identity computationally useful beyond storage? (2) Can permutation invariance detect genuine identity use? (3) Does identity improve reasoning or only memory?

**Claim tiers:**

| Tier | Role | Outcomes |
|---|---|---|
| **Tier 1 — lives or dies** | required for any conclusion | Identity main effect (reasoning, synthetic + NL co-primary); permutation invariance + sufficiency conjunction; symmetric locks (HRL vs LL) |
| **Tier 2 — mechanism attribution** | needed to attribute cause | Graph interdiction; encoder-isolation ablation ladder |
| **Tier 3 — scope and robustness** | strengthens generality | Scaling divergence; identity-benefit vector; distribution-shift + cross-topology transfer; efficiency axis |
| **Tier 0 — validity gate** | precondition on reading any tier | Measurement-validity controls (§12): degeneration covariate, frequency-matched budgets, length-matched baseline, held-out-diagnostic constraint |

Tier 0 is new in v7: no Tier-1/2/3 result is interpreted until its measurement-validity covariates clear (e.g., an IPI result is read only jointly with the anisotropy it was measured under).

**Construct-validity commitment (governs the whole framework).** The object of measurement is *identity behavior in the bare architecture under test* — a standard transformer-class knowledge system. This is deliberate. A stress-test of v6 correctly identified optimization and geometry confounds, but its proposed remedies were almost all architectural interventions (sparse α-entmax attention, an advective-diffusion backbone, adaptive gradient gating, embedding-specific learning rates, auxiliary structural supervision). Each is a legitimate technique, but adopting them as defaults would be **circular**: several are engineered to produce topological robustness, anti-shortcut behavior, or anti-collapse geometry — exactly the properties the diagnostics exist to detect. Running the cross-topology generalization test on a backbone designed for topological generalization measures the backbone, not identity. Therefore:

- Confounds are **instrumented, not engineered away** — measured as covariates (§12), consistent with the framework's founding philosophy.
- Robustness-engineered variants enter **only as labeled ablations** answering "does the identity effect survive under X," and always alongside the bare-architecture baseline that defines the construct. They never replace it.

This guardrail also inoculates the work against the inverse objection (that a robust backbone manufactured the result).

A reader wanting only the central result can read Tiers 0–2 (§§5–12); Tier 3 strengthens generality.

---

## 3. Motivation (Firewalled from the Empirical Claims)

A distributional embedding of "Aaron" carries Fregean *sense*; a stable identifier behaves like a Kripkean **rigid designator**, denoting one individual independent of description, as a realist CCO/BFO IRI does. This motivates *why* explicit identity might behave like a primitive and supplies one prediction — invariance under relabeling.

**Independence statement.** No empirical claim depends on this framing; the predictions come from permutation invariance, alias stability, graph structure, the locks, scaling, and the measurement-validity covariates — not from Kripke. Full treatment in Appendix A; the main text proceeds without it.

---

## 4. Decomposing Identity: Encoding vs Binding

**Axis E — Encoding:** **E0** surface string · **E1** ID string (control) · **E2** atomic reserved token. **Axis B — Binding:** **B0** no dedicated identity · **B1** dedicated per-entity object (embedding if parametric, slot/pointer if non-parametric). Encoding is a self-contained tokenization sub-study; Binding is studied via the Identity × Memory factorial. Neither co-varies with the other or with memory location.

---

## 5. The Identity × Memory Factorial

| | **B0 — no dedicated identity** | **B1 — dedicated identity** |
|---|---|---|
| **Parametric** | **LW** lexical-in-weights | **EE** entity-embedding |
| **Non-parametric** | **TR** text-RAG | **KV** key-value entity memory |

Identity main effect = mean of `EE−LW`, `KV−TR`; memory effects `TR−LW`, `KV−EE`; interaction `(KV−TR)−(EE−LW)`. Encoding fixed at E2, topology Structured, reasoning Neural across the factorial.

---

## 6. Symmetric Forcing Functions: Hard Reference Lock and Lexical Lock

- **HRL:** model receives only identifiers; decoding only through `ID → fact retrieval → generation`; no distributional bypass.
- **LL:** model receives only surface forms; retrieval forced through a fixed lexical-surface lookup; no identity channel.
- **Free-form:** both channels available.

The clean contrast **HRL vs LL** holds the architectural restriction of locking constant and varies only the forced channel, isolating identity from the act of restriction. Readings: *HRL≈free & LL≪free* ⇒ identity load-bearing; *HRL≪free & LL≈free* ⇒ rode the lexical channel; *HRL≈LL≈free* ⇒ interchangeable. Both run under oracle and fallible resolution with two-stage decomposition (§13).

---

## 7. Competing Hypotheses

**H₀ — Identity Neutrality.** Identity improves storage/retrieval/updating/alias but not reasoning: null identity main effect (synthetic and NL); HRL collapses while LL holds; no non-trivial IPI gap; identity-benefit vector flat on the memory axis; gap shrinks with scale.

**H₁ — Identity Advantage.** Identity improves memory *and* reasoning: significant identity main effect surviving fallible resolution and replicating in the NL arm; HRL load-bearing vs LL; high IPI *with* the sufficiency conjunction (and low anisotropy, so invariance is not collapse); reasoning survives interdiction; gap stable/growing with scale; advantage survives the length-matched efficiency baseline or is reported as a stated tradeoff.

**H₂ — Hybrid Dominance.** Best long-hop reasoning requires identity *plus* symbolic traversal (D), clearing the ablation ladder; scaling gap U-shaped.

*Updating remains a Memory-Location manipulation check.*

---

## 8. Conditions and Design Matrix

| Condition | Binding | Memory | Reasoning | Role |
|---|---|---|---|---|
| LW | B0 | Param | Neural | baseline |
| EE | B1 | Param | Neural | identity, parametric |
| TR | B0 | Non-param | Neural | retrieval, no identity |
| KV | B1 | Non-param | Neural | identity, externalized |
| D | B1 | Non-param | Symbolic | hybrid traversal |
| HRL | B1 | Non-param | Neural + identity lock | forcing: identity mandatory |
| LL | B0 | Non-param | Neural + lexical lock | forcing: lexical mandatory |
| LW+dummy | B0 | Param | Neural | length-matched efficiency baseline (§12) |
| G / G+probe / G+frozen-LLM | — | — | encoder ladder | structural-reliance attribution |

A/A′, B/B′ are flat/structured variants of LW/TR; E0/E1/E2 the encoding sub-study. Robustness-engineered variants (entmax, ADT backbone, etc.) appear only as labeled ablations (§12, guardrail).

---

## 9. The Diagnostic Protocol: Failure Modes as Fingerprints

No accuracy number is reported without its fingerprint. Modes 1–10 from v6; 11–13 are the measurement-validity additions.

| # | Failure mode | Confounded with | Instrument | Distinct signature | Tier |
|---|---|---|---|---|---|
| 1 | Identity present but **bypassed** | genuine use | HRL vs LL (§6) | HRL≈free & LL≪free ⇒ load-bearing | 1 |
| 2 | **Statistical shortcut** | real reasoning | interdiction (§10.2) | accuracy drops under interdiction iff shortcut | 2 |
| 3 | **Resolution error** as reasoning error | reasoning failure | two-stage decomposition (§13) | error splits ER vs reasoning-given-correct-ID | 1 |
| 4 | **Encoder solves it** | LLM reasoning | ablation ladder (§10.3) | full≈G or ≈G+probe ⇒ LLM idle | 2 |
| 5 | **Overfit structure** | structural rules | shift + cross-topology (§10.4) | falls under shift / fails transfer | 3 |
| 6 | Helps **storage, not reasoning** | reasoning gain | identity-benefit vector (§10.2) | vector flat on memory axis | 1 |
| 7 | **Converges at scale** | computational primitive | scaling divergence (§10.4) | gap shrinks ⇒ representational | 3 |
| 8 | **Representation-specific naming** | abstract primitive | permutation invariance (§11) | IPI collapse | 1 |
| 9 | **Trivial invariance — identity ignored** | identity-as-primitive | IPI × identity-discriminative accuracy (§11) | high IPI + low discriminative acc ⇒ ignored | 1 |
| 10 | **Gain doesn't survive cost** | practical benefit | efficiency axis (§10.5) | no Pareto gain on acc/FLOP, acc/GB, latency | 3 |
| 11 | **Invariance via rank collapse** (geometry) | abstract invariance | **degeneration covariate** (§12) | high IPI **and high anisotropy / spectral decay** ⇒ collapse, not primitive | 0 |
| 12 | **Attention-sink / length artifact** | identity benefit | **length-matched dummy baseline** (§12) | LW+dummy closes the gap ⇒ artifact | 0 |
| 13 | **Degree-induced embedding degeneration** in transfer | structural-rule failure | **frequency-matched budgets + per-node norm** (§12) | transfer failure tracks peripheral-node norm collapse | 0 |

Modes 11–13 are *validity gates*: they are read before any Tier-1/2/3 conclusion. Mode 11 reinforces mode 9 — the sufficiency conjunction already excludes a rank-collapsed model (it fails identity-discriminative accuracy), and the anisotropy covariate now makes that exclusion *observed* rather than inferred.

---

## 10. Five Diagnostic Axes

### 10.1 Identity Usage
Two-stage ER accuracy (Stage 1: surface→ID); identity persistence (ICR, ISS); alias resolution; the HRL/LL load-bearing test; and **identity-discriminative accuracy** (same-surface disambiguation, entity-specific queries) as the sufficiency check for invariance. Memory-gain component of the identity-benefit vector lives here.

### 10.2 Reasoning
Hop-depth generalization (Stage-2, hops {2,5,10,20,50}); **graph interdiction** — (A) anti-statistical queries, (B) counterfactual edge swaps, (C) path masking; reasoning-gain component and the **identity-benefit vector** `(memory gain, reasoning gain)`. Storage-only effects lie flat on the memory axis.

### 10.3 Structural Reliance
**Encoder-isolation ablation ladder:** G alone (permutation-equivariant ceiling) / G+linear-probe / G+frozen-LLM / full. full≈G ⇒ LLM irrelevant; G+probe≈full ⇒ linear in graph space; full≫all ⇒ LLM reasoning. Interdiction feeds this axis.

### 10.4 Generalization
- **Unseen-entity generalization** (train E1–1M, test E1,000,001–1,100,000, novel identities).
- **Cross-topology transfer** (scale-free ↔ small-world), run **under frequency-matched update budgets** (§12) so that any transfer failure can be attributed to structural-rule failure rather than degree-induced embedding degeneration, with **per-node embedding norm** carried as a covariate (mode 13).
- **Scaling divergence** (10k/100k/1M; slope of the identity gap).
- **Distribution-shift suite** (topology, identity, edge-noise, schema drift; robustness curves).
- **Permutation invariance** (§11).

### 10.5 Efficiency
For every Tier-1 contrast, report **accuracy/FLOP, accuracy/GB, inference latency, update cost**, plus — new in v7 — **measured wall-clock latency, memory-bandwidth utilization, and cross-device communication overhead**, not theoretical FLOP/GB alone. The comparison is against the **length-matched baseline LW+dummy** (§12), not bare LW, so an identity advantage cannot be a trivial consequence of sequence-length increase or attention-sink anchoring. Cost-adjusted rule: an advantage is **unconditional** only if Pareto-improving or cost-neutral against LW+dummy; otherwise it is a **tradeoff point with the multiplier stated**.

---

## 11. Permutation Invariance: Necessary, Not Sufficient

`IPI = 1 − Var_π[score]/Var_ref`; non-parametric conditions permuted at inference, parametric across the seed protocol; G is the invariance ceiling. High IPI is **necessary but not sufficient**: a model that *ignores identity* (topology-only or rank-collapsed) is also trivially invariant.

**Sufficiency conjunction.** A condition earns "computational identity" only if **all** hold: (1) `IPI ≥ τ_IPI` (≥0.8× G's ceiling) on long-hop unseen-entity reasoning; (2) high **identity-discriminative accuracy** (§10.1) — a degenerate invariant model fails alias resolution and same-surface disambiguation; (3) **HRL load-bearing** vs LL; (4) **interdiction-robust**; and, new in v7, (5) **low representational anisotropy** — the invariance must not be explained by the singular-value-spectrum collapse measured in §12 (mode 11).

**Held-out / evaluation-only constraint (new in v7).** IPI, attention-entropy, anisotropy, and every other diagnostic are computed **only at evaluation and are never differentiable training signals**. The framework forbids any objective, auxiliary loss, or regularizer that optimizes toward a diagnostic, because a model permitted to backpropagate through an invariance or entropy metric can satisfy the diagnostic without acquiring the property. This closes the gameability loophole at the protocol level rather than the architecture level.

---

## 12. Measurement-Validity Controls (Tier 0)

These controls make optimization-dynamics and representation-geometry confounds *measured covariates* rather than uncontrolled threats, in keeping with the framework's "instrument, don't engineer away" philosophy.

**12.1 Representation-degeneration covariate.** On every identity, invariance, and transfer result we measure and report: **cosine anisotropy** (mean pairwise cosine similarity of embeddings), the **singular-value spectrum** of the embedding matrix (spectral-decay rate), and **per-node / per-token embedding norm** distributions. These are carried as covariates and entered into the mixed-effects model. The purpose: a high IPI accompanied by high anisotropy and sharp spectral decay is rank collapse (mode 11), not abstract invariance; a transfer failure accompanied by collapsed peripheral-node norms is embedding degeneration (mode 13), not rule failure. The covariate converts both stress-test confounds into reportable diagnoses.

**12.2 Frequency-matched update budgets.** Because random-walk training over long-tailed (scale-free) graphs over-trains hubs and starves peripheral nodes — leaving rare embeddings to decay toward the origin under unopposed weight decay — we equalize **per-node (and per-token) gradient-update counts** across conditions and topologies, via uniform node sampling or importance-reweighted walks. The synthetic world's full control over the generative process makes this exact. This neutralizes degree-induced degeneration *at the source*, so cross-topology transfer (§10.4) tests structural rules rather than which nodes happened to be trained. Per-node update counts are logged and reported.

**12.3 Length-matched efficiency baseline.** Prepending `M` identity/memory tokens raises sequence length `L → L+M`, increasing attention cost and KV-cache footprint, and prepended read-only tokens act as **attention sinks** that stabilize the residual stream independent of their content. A gain measured against bare `LW` therefore confounds identity with length and anchoring. We add **LW+dummy**, prepending `M` random/dummy tokens to the baseline to match length, KV-cache, and sink effects; all identity-vs-baseline efficiency and accuracy contrasts are reported against LW+dummy (mode 12). Reporting includes measured wall-clock, memory-bandwidth, and cross-device communication, not theoretical estimates alone.

**12.4 Natural-language signal controls.** The NL arm (§15) adds **shuffled-signal controls** (permute computed attention-entropy/variance across positions to test whether gains depend on exact token alignment or merely the distribution of attention statistics) and **template-variation controls** (paraphrase and minor template reformulation to detect template-matching and selection bias). A gain that survives shuffling and template variation is not a template/attention-routing artifact.

**12.5 Held-out-diagnostic constraint.** As stated in §11, no diagnostic may enter the training objective; all are evaluation-only. This is a protocol invariant enforced across every condition.

**Construct-validity guardrail (restated).** The stress-test's architectural remedies — α-entmax/ASEntmax attention, Advective Diffusion Transformer backbone, adaptive gradient gating, embedding-specific learning rates, auxiliary structural supervision — are **declined as defaults** and admitted **only as labeled ablations** of the form "does the identity effect survive under X," always reported alongside the bare-architecture baseline. Adopting them as defaults would measure the intervention, not identity, and in the cross-topology case would be circular. (Where a control is *non-architectural* — frequency-matched budgets, length-matched baselines, covariate measurement — it is adopted as a default, because it changes what is measured, not the substrate under test.)

---

## 13. Entity Resolution: Two-Stage Decomposition and Noise Model

`Total error = ER error (Stage 1) + reasoning error given correct ID (Stage 2)`, reported as stacked bars so resolution failures are never charged to reasoning. Oracle vs fallible resolution for every B1 condition; noise model `N(ρ,H)` with out-of-distribution resolver evaluation and a noise-sensitivity curve with crossover `ρ*`. Residual: bounded synthetic noise; messy reference is probed only in the NL arm.

---

## 14. Synthetic World Generation

Complete ground truth; zero contamination; synthetic nonce surface forms; from-scratch, compute-matched training (fixed token budget, matched parameters and FLOPs). **Per-node update budgets are frequency-matched (§12.2).** Scale target: Persons 100k, Orgs 10k, Places 50k, Assets 50k, Events 25k; relations worksFor 1M, owns 500k, locatedIn 500k, manages 250k, reportsTo 250k. Instantiated at 10k/100k/1M for scaling divergence, and across scale-free and small-world families for cross-topology transfer (Erdős–Rényi as a uniform-degree control). Faithful alias classes in the core; nicknames and definite descriptions only in the NL arm.

---

## 15. Natural-Language Arm — Co-Primary, with Signal Controls

The synthetic core is sense-degenerate by construction; a conclusion drawn only there concerns nonce identifiers, not language. v6 promoted the **identity main effect on Stage-2 reasoning under informative sense** to a co-primary outcome on realistic templated text over synthetic entities. v7 adds the **shuffled-signal and template-variation controls** of §12.4, because the stress-test correctly noted that LLM gains in natural language can be driven by abstract-vs-concrete attention-entropy dynamics and template-matching rather than identity. The co-primary NL claim is credited only if it survives those controls. A synthetic-vs-NL divergence remains a primary, reportable finding about the sense/reference tradeoff.

---

## 16. Bayes-Optimal Ceiling, Scoped

Exact on deterministic tasks; posterior-based with stated prior and loss for ambiguous references; reference-free factual-accuracy oracle for generation. Headroom reported only where defined.

---

## 17. Pre-Registered Analysis Plan

**Tier-0 validity gates (read before any conclusion):** degeneration covariates (anisotropy, spectrum, per-node norm) within bounds for the relevant contrast; per-node update counts matched; efficiency contrasts taken against LW+dummy; confirmation that no diagnostic entered training.

**Tier-1 primary outcomes:**
1. Identity main effect on reasoning (synthetic) — mean of `EE−LW`, `KV−TR` on Stage-2 multi-hop and structural generalization, oracle resolution, **conditioned on the degeneration covariate**.
1b. Identity main effect on reasoning (NL arm), **surviving shuffled-signal and template controls** — co-primary.
2. Permutation invariance with the **five-part sufficiency conjunction** (§11), including low anisotropy.
3. Symmetric load-bearing test — HRL vs LL vs free-form.

**Tier-2:** interdiction degradation; encoder-isolation ablation ladder.

**Tier-3:** scaling-divergence slope; identity-benefit vector geometry; distribution-shift robustness; cross-topology transfer **under frequency-matched budgets**; efficiency (measured wall-clock/bandwidth, against LW+dummy).

**Gating contrasts:** tokenization (E0→E1→E2); Identity×Memory interaction; structure-scrambled control; two-stage decomposition; cost-adjustment against LW+dummy; degeneration-covariate conditioning on every geometry-sensitive result.

**Statistics.** ≥5 world×training seeds (same seeds for cross-seed IPI); mean and 95% CI; mixed-effects model (condition fixed, seed random; degeneration covariates as fixed-effect controls; cluster-robust fallback); Benjamini–Hochberg FDR across the family, Holm on Tier-1; "meaningful" requires clearing both the effect-size threshold δ and corrected significance; convergence-failed runs re-seeded.

**Decision rules (fingerprint → conclusion):**

| Fingerprint | Conclusion |
|---|---|
| Null identity main effect (synthetic **and** NL) **and** HRL collapses while LL holds **and** no non-trivial IPI gap | **H₀** |
| Identity main effect meaningful synthetic **and** NL (post-controls); IPI high with full sufficiency conjunction incl. low anisotropy; HRL load-bearing vs LL; full≫G; gap stable/growing; advantage survives or states cost vs LW+dummy | **H₁** |
| Identity meaningful **and** only D (with G) clears ablation/interdiction on long-hop; scaling U-shaped | **H₂** |
| `E1−E0` meaningful but `E2−E0` not | tokenization artifact |
| full≈G or G+probe≈full | computation is the encoder's |
| Accuracy up but IPI(B1)≈IPI(LW) | memorization, not primitive |
| High IPI but low identity-discriminative accuracy | identity ignored (trivial invariance) |
| High IPI **with high anisotropy / sharp spectral decay** | invariance is **rank collapse**, not abstract identity |
| Identity-benefit vector flat on memory axis | storage, not reasoning |
| Accuracy holds but interdiction doesn't degrade it | statistical shortcut |
| Identity gap shrinks with scale | representational convenience |
| Synthetic effect present, NL effect absent (or fails shuffle/template controls) | about nonce IDs / templates, not language |
| Gain vanishes against **LW+dummy** | **attention-sink / length artifact**, not identity |
| Cross-topology failure tracks **peripheral-node norm collapse** | **embedding degeneration**, not structural-rule failure |
| Advantage present but no Pareto gain on measured efficiency | cost-dominated; report as tradeoff |

---

## 18. Predicted Outcomes and Fingerprints

| Capability / probe | LW | LW+dummy | EE | KV | D | HRL | LL | G | Confirms |
|---|---|---|---|---|---|---|---|---|---|
| Identity-discriminative accuracy | Low | Low | High | High | High | High | Low | — | IPI sufficiency |
| Long-hop reasoning (Stage 2) | Low | Low | Med | Med | High | Med | Low | High | reasoning = traversal |
| Interdiction robustness | Low | Low | Med | High | High | High | Low | High | shortcut-resistance |
| Load-bearing (HRL vs LL) | — | — | — | — | — | High | Low | — | optional vs necessary |
| Permutation invariance (IPI) | Low | Low | High | High | High | High | Low | Ceiling | identity-as-computation |
| Anisotropy (covariate; lower=better) | — | — | Low | Low | Low | Low | — | Low | invariance ≠ collapse |
| Cross-topology transfer (freq-matched) | Low | Low | Med | High | High | High | Low | High | structural rules |
| Scaling-gap slope | — | — | + | + | U | + | − | — | computational primitive |
| Efficiency (acc/GB vs LW+dummy) | ref | ref | Med | Low | Low | Low | High | Med | cost realism |

The Tier-1 rows (HRL-vs-LL gap; IPI *with low anisotropy and high discriminative accuracy*) are what license "primitive." The anisotropy and LW+dummy rows are the validity gates that prevent a geometric or length artifact from being read as identity.

---

## 19. Threats to Validity

Design threats (handled v3–v6): identity↔memory entanglement (factorial); encoding↔binding conflation (orthogonal axes); identity bypassed / lock asymmetry (HRL vs LL); statistical shortcut (interdiction); resolution-as-reasoning error (two-stage); encoder solves it (ablation ladder); structural overfitting (shift + cross-topology); storage-as-reasoning gain (benefit vector); convergence at scale (scaling); identity-as-naming (IPI); trivial invariance / identity ignored (sufficiency conjunction); cost (efficiency); nonce-IDs-not-language (NL co-primary); compute confound (matched params/FLOPs); philosophy driving predictions (firewalled).

Measurement threats (new, handled v7): invariance via rank collapse (degeneration covariate, §12.1); rare-token stagnation / degree-induced degeneration (frequency-matched budgets, §12.2); attention-sink/length artifact (LW+dummy, §12.3); concreteness/attention-entropy and template confounds in the NL arm (shuffled-signal/template controls, §12.4); diagnostic gameability (held-out-diagnostic constraint, §12.5); and **construct invalidation by robustness engineering** (construct-validity guardrail, §2/§12 — architectural remedies admitted only as ablations).

Statistical: seeds, mixed-effects with covariate controls, FDR/Holm, δ.

---

## 20. Relation to Prior Work

Architectural identity conditions are continuous with entity-memory research (Entities-as-Experts, KnowBERT); KG embeddings (TransE, RotatE, GraphSAGE, GAT) supply the encoder ladder, and GNN permutation-equivariance is why G is the IPI ceiling; knowledge editing (ROME, MEMIT) grounds treating updating as memory-location; GraphRAG is the pipeline question distinguished from the substrate question; HDT motivates identity-as-dictionary; memory/pointer networks ground KV. The measurement-validity controls draw on the representation-degeneration and anisotropy literature, attention-sink/register-token findings, and topological-OOD robustness work; these are used to *measure* confounds, while the corresponding architectural remedies (sparse attention, diffusion-PDE backbones, dynamic reweighting) are reserved as ablations under the construct-validity guardrail.

**Narrowed novelty:** the primary contribution is the **diagnostic framework** — the Encoding/Binding factorial, symmetric forcing functions, permutation invariance with a five-part sufficiency conjunction, the failure-mode fingerprint protocol, the Tier-0 measurement-validity controls, efficiency accounting, and cross-topology transfer — a toolkit reusable across knowledge-grounded systems, with an explicit construct-validity guardrail. The HDT-grounded architecture is the testbed.

---

## 21. Contributions

1. A **diagnostic framework** in which every plausible explanation for a performance difference — design *and* measurement — has a distinct, pre-registered empirical fingerprint.
2. **Encoding × Binding** decomposition and **Identity × Memory** factorial, free of memory-location and tokenization confounds.
3. **Symmetric forcing functions** (HRL, LL) isolating identity from restriction.
4. **Permutation invariance with a five-part sufficiency conjunction** (incl. low anisotropy), with an explicit guard against trivial invariance and rank collapse.
5. **Tier-0 measurement-validity controls** — degeneration covariate, frequency-matched update budgets, length-matched efficiency baseline, held-out-diagnostic constraint, NL shuffled-signal/template controls.
6. A **construct-validity guardrail** fixing the substrate under test and admitting robustness-engineered architectures only as ablations — preventing circular measurement.
7. A **five-axis evaluation** (identity usage, reasoning, structural reliance, generalization, efficiency) with measured wall-clock/bandwidth cost.
8. **Cross-topology transfer under frequency-matched budgets**; a **co-primary NL outcome** with signal controls; and a pre-registered analysis plan mapping fingerprints to decision rules including the null, memorization, identity-ignored, rank-collapse, length-artifact, embedding-degeneration, and cost-dominated verdicts.

---

## 22. Significance for Realist Knowledge Infrastructure (Motivation)

A realist knowledge system needs identity to be a stable, first-class object — a rigid designator surviving paraphrase, alias, and update, as a CCO/BFO IRI is. The framework makes the realist claim falsifiable at the level of mechanism *and* measurement: permutation invariance with its sufficiency conjunction tests whether identity is genuinely rigid, used, and not merely a geometric collapse; the symmetric locks test whether it is load-bearing; interdiction tests whether reasoning is structural; cross-topology transfer under matched budgets tests whether structural rules were learned; the scaling slope tests whether value compounds; the measured efficiency axis tests whether it pays for itself. If H₀ holds, identity earns its place on provenance grounds while reasoning stays lexical. If H₁ holds and the fingerprints confirm it — including the validity gates — identity belongs in the architecture on reasoning grounds. If H₂ holds, identity must be paired with explicit traversal. The empirical core stands independently of this motivation (§3).

---

## 23. Conclusion

v7 hardens the framework where a sophisticated reviewer attacks last: not the design, but the *measurement*. It carries a representation-degeneration covariate on every geometry-sensitive result, so high invariance accompanied by rank collapse is diagnosed rather than mistaken for abstraction; it frequency-matches update budgets, so cross-topology transfer tests rules and not training luck; it length-matches the efficiency baseline with dummy tokens, so an attention-sink is never read as identity; it forbids any diagnostic from entering training, closing the gameability loophole; and it controls the natural-language arm for concreteness and template artifacts. Crucially, it does all of this by **instrumenting** confounds as covariates rather than engineering them away — and fixes a construct-validity guardrail that admits robustness-engineered architectures only as ablations, because a backbone built for topological generalization cannot be the thing that runs the topological-generalization test. Adjudicated by pre-registered decision rules that now include the rank-collapse, length-artifact, and embedding-degeneration verdicts alongside the null and the memorization verdict, the study is built to return a *mechanism with a cost, measured under controlled geometry* — not merely a number — and the toolkit it introduces applies well beyond the architecture it is exercised on.

---

## Appendix A — Motivation: Rigid Designation and Fregean Sense (firewalled)

A Fregean account distinguishes *Sinn* (sense) from *Bedeutung* (reference). A distributional embedding is sense-laden — "Aaron" is placed by its associations. Kripke's *Naming and Necessity* argues proper names are **rigid designators**: they denote the same individual across possible worlds without descriptive mediation. A realist ontology IRI implements rigid designation in software, denoting one individual by stipulation and provenance, sense-free. This motivates the expectation that explicit identity should contribute invariantly under relabeling and should survive contexts where descriptive sense is absent or misleading. The synthetic world is deliberately sense-degenerate to isolate reference from sense; the NL arm reintroduces sense to test the tradeoff. None of the empirical predictions require this framing; readers may treat this appendix as optional.

## Appendix B — References (verify exact details before submission)

- Kripke, S. *Naming and Necessity.* Harvard University Press, 1980.
- Frege, G. "Über Sinn und Bedeutung." 1892.
- Fernández, J. D., et al. "Binary RDF representation for publication and exchange (HDT)." *Journal of Web Semantics*, 2013.
- Bordes, A., et al. "Translating Embeddings for Modeling Multi-relational Data (TransE)." NeurIPS, 2013.
- Sun, Z., et al. "RotatE." ICLR, 2019.
- Hamilton, W., Ying, R., Leskovec, J. "Inductive Representation Learning on Large Graphs (GraphSAGE)." NeurIPS, 2017.
- Veličković, P., et al. "Graph Attention Networks." ICLR, 2018.
- Févry, T., et al. "Entities as Experts." EMNLP, 2020.
- Peters, M., et al. "Knowledge Enhanced Contextual Word Representations (KnowBERT)." EMNLP, 2019.
- Meng, K., et al. "Locating and Editing Factual Associations in GPT (ROME)." NeurIPS, 2022.
- Meng, K., et al. "Mass-Editing Memory in a Transformer (MEMIT)." 2022.
- Edge, D., et al. "From Local to Global: A Graph RAG Approach." 2024.
- Weston, J., Chopra, S., Bordes, A. "Memory Networks." ICLR, 2015.
- Vinyals, O., Fortunato, M., Jaitly, N. "Pointer Networks." NeurIPS, 2015.
- Watts, D., Strogatz, S. "Collective dynamics of 'small-world' networks." *Nature*, 1998.
- Barabási, A.-L., Albert, R. "Emergence of scaling in random networks." *Science*, 1999.
- *(Measurement-validity literature on representation degeneration/anisotropy, attention sinks/register tokens, sparse attention (entmax), and topological-OOD robustness to be cited at the specific claims in §12; verify canonical sources — including the status of advection-diffusion transformer work — before submission.)*

## Appendix C — Consolidated Version Changelog

**Original → v3:** identity ladder replaces relabeling; full design matrix; oracle/fallible resolution; rigid-designator framing + NL arm; synthetic nonce forms + from-scratch matched compute; scoped Bayes ceiling; updating reclassified; pre-registered plan; HGLM = HDT-Grounded; related work; nicknames scoped to NL arm.

**v3 → v4:** Identity × Memory factorial; Encoding/Binding split; matched-difficulty + structure-scrambled controls; G promoted to gating control; resolution noise model + OOD resolver; permutation invariance added.

**v4 → v5:** Hard Reference Lock; graph interdiction; two-stage error decomposition; encoder-isolation ablation ladder; distribution-shift suite; identity-benefit vector; scaling-divergence test; failure-mode fingerprints across four axes.

**v5 → v6:** Lexical Lock (symmetric forcing); IPI sufficiency conjunction (+ trivial-invariance guard); philosophy firewalled; NL arm promoted to co-primary; efficiency axis; cross-topology transfer; three-pillar / three-tier claim hierarchy; reframed so the diagnostic framework is the contribution.

**v6 → v7 (this version), mapping to the adversarial stress-test:**

| Stress-test point | v7 disposition |
|---|---|
| ST1 — asymmetric gradient dynamics, rare-token stagnation, embedding degeneration. | **Adopt (non-architectural):** frequency-matched update budgets (§12.2) + per-node-norm covariate (§12.1). **Decline as default (architectural):** gradient gating, embedding-specific LRs → ablations only (guardrail). |
| ST2 — trivial invariance via rank collapse; diagnostic gameability. | **Adopt:** anisotropy/spectrum covariate as 5th sufficiency term (§11, §12.1); held-out-diagnostic constraint (§12.5). **Note:** sufficiency conjunction already excluded rank-collapsed models via discriminative accuracy. **Decline as default:** entmax swap, auxiliary supervision → ablations only. |
| ST3 — cross-topology confounded by degree-induced degeneration. | **Adopt:** frequency-matched budgets + per-node-norm covariate; Erdős–Rényi uniform control (§10.4, §12.2). **Decline as default:** advective-diffusion backbone, topology-aware reweighting → ablations only (circularity guardrail). |
| ST4 — NL arm confounded by concreteness/attention-entropy and template-matching. | **Adopt:** shuffled-signal and template-variation controls; NL co-primary credited only post-controls (§12.4, §15). **Decline as default:** causal-invariant/RL training-level debiasing → ablations only. |
| ST5 — efficiency axis lacks length-matched baseline; ignores systemic cost. | **Adopt:** LW+dummy length-matched baseline; measured wall-clock, memory-bandwidth, cross-device overhead (§12.3, §10.5). **Note:** SMoE routing pathologies are out of scope (no MoE in the testbed) and would enter only if an MoE variant is ablated. |
| Meta — review's prescriptions are architectural and partly circular. | **Construct-validity guardrail** (§2, §12): substrate under test fixed as bare transformer-class; robustness-engineered variants admitted only as labeled ablations alongside the baseline; non-architectural controls adopted as defaults. |
