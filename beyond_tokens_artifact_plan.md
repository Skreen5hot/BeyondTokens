# Beyond Tokens — Research Artifact Generation & Procurement Plan

**Companion to the v7 hardened implementation plan. Specifies every artifact the experiment requires, whether it is generated or procured, how it is validated before use, how it is provenance-recorded, and in what order it must be built.**

---

## 0. Scope and Principles

**What counts as a research artifact here.** Six classes: (A) data, (B) generation & training tooling, (C) diagnostic instruments, (D) models, (E) compute/infrastructure, (F) provenance/governance records. The artifact plan is itself an integrity control: the study's validity rests on a contamination-free, *regenerable* world and on diagnostics whose logging must exist before any training run.

**Build-vs-procure default.** Generate everything whose correctness or contamination-status is load-bearing (all data, the bespoke samplers, the diagnostic library, the bare baseline). Procure mature, auditable, neutral infrastructure (graph primitives, a minimal training scaffold, KG-embedding encoders, statistics, tracking). Procured code is **wrapped behind a thin adapter, never forked into the core**, so the bare baseline stays inspectable and compute-matchable.

**Guardrail-aware procurement.** Every artifact carries a *guardrail role*: it serves the **bare baseline** (must be stripped to standard softmax / single LR / auditable), an **ablation arm** (procured strong implementation, quarantined, may not report its circular diagnostic), or a **covariate instrument** (neutral, evaluation-only). Procurement must not smuggle robustness defaults into the baseline — e.g., a procured transformer scaffold is stripped of any sparse-attention or fused-optimizer defaults before it becomes LW.

**Edge-Canonical honesty.** The data generators, the diagnostic library, and the provenance layer are built **edge-canonical**: deterministic, dependency-light, runnable unmodified in Node.js/browser, regenerable from `spec + seed` with no required infrastructure — consistent with the broader program's substrate commitments and with reviewer reproducibility. **Model training is not edge-canonical and will not be forced to be**: training transformers up to 1M entities / 25M triples across many configurations and seeds is GPU-cluster work, as the protocol's own A100/B200 accounting acknowledges. Stating this plainly avoids a false claim; the artifacts that feed the rest of the program remain edge-canonical, the training stage is honestly cluster-bound.

**Provenance principle.** Every artifact is **content-addressed** (`spec_hash + seed + code_commit → dataset/checkpoint ID`) and accompanied by a datasheet (data) or model card (models). Any dataset must be re-derivable bit-for-bit from its manifest; any checkpoint must record the exact data ID, config, seed, and code commit that produced it.

**Critical sequencing principle — instruments before runs.** The diagnostic library (C) and the Tier-0 gate logging (per-node update counts, geometry traces, two-stage error hooks) must be built and unit-tested **before** any training run. If the geometry tracker or update-count logger is discovered missing after training, the runs are wasted — and at the 1M tier that is the most expensive possible mistake. This is the artifact-plan analogue of the held-out-diagnostic discipline.

---

## 1. Artifact Inventory (Master Register)

Legend — **G** generate, **P** procure(+wrap). Guardrail role — **BB** bare baseline, **ABL** ablation-only, **COV** covariate/instrument, **N** neutral infra.

### A. Data artifacts

| Artifact | G/P | Role | Depends on | Acceptance gate |
|---|---|---|---|---|
| A1 Topological graphs ($G_{SF}$, $G_{SW}$, $G_{ER}$ × 10k/100k/1M) | G | N | graph generator (B1) | degree distribution matches target (KS / γ-fit / clustering / Poisson within tolerance); exact N, M; connectivity verified |
| A2 Reserved vocabularies ($V_{lexical}$, $V_{entity}$, $V_{dummy}$) | G | BB | tokenizer (B3) | pairwise-disjointness proof; nonce surface forms have zero corpus presence |
| A3 Frequency-matched walk corpora | G | N | walk sampler (B2), A1 | per-node visit-count CoV ≤ tolerance across nodes **and** across families; full node/edge coverage; deterministic re-derivation hash-matches |
| A4 NL templated corpus (D/A relational templates) | G | N | A3, template engine (B4) | relation-verb realizations correct per relation; D/A pairs reference identical facts; balanced positive/negative with controlled-plausibility corruption |
| A5 Alias & ambiguity sets (exact / abbreviation / misspelling / ambiguous) | G | COV | A2, A1 | each class's defining property verified (co-denotation; deterministic truncation; controlled edit distance; ≥2 ground-truth referents) |
| A6 Anti-statistical interdiction probes | G | COV | A3 **profiled**, A1 | per item: measured corpus co-occurrence predicts the *wrong* answer **and** path traversal predicts the *right* one (both verified); yield report |
| A7 Edge-swap & path-mask probe sets | G | COV | A1 | swap/mask applied at inference only; ground-truth pre/post answers recorded |
| A8 Unseen-entity holdouts ($E_{1{,}000{,}001}$–$E_{1{,}100{,}000}$) | G | N | A1 | identical topology to train; zero identifier overlap with train |
| A9 Cross-topology transfer pairs (SF↔SW, ER control) | G | N | A1, A3 | matched ⟨k⟩; frequency-matched both directions |
| A10 Resolver noise grid ($N(\rho,H)$) incl. OOD regimes | G | COV | A2, A5 | corruption rate ρ and ambiguity entropy H hit targets; OOD regime disjoint from resolver-train regime |
| A11 Diagnostic fixtures (known-answer test data) | G | COV | — | each fixture has a closed-form expected instrument output (see C gates) |

### B. Generation & training tooling

| Artifact | G/P | Role | Notes / acceptance gate |
|---|---|---|---|
| B1 Graph generator (BA / WS / ER) | P→G wrap | N | procure NetworkX/igraph/graph-tool for primitives; **igraph or graph-tool at 1M for performance**; wrap behind a deterministic, seeded edge-canonical interface; gate = A1 distributions |
| B2 Frequency-matched walk sampler | G | N | **bespoke** — degree-corrected / importance-reweighted walks equalizing per-node update counts; emits per-node visit log; gate = A3 CoV tolerance |
| B3 Tokenizer & reserved-partition manager | G | BB | enforces $V_{dummy}\!\perp\!V_{lexical}\!\perp\!V_{entity}$; per-sequence dummy resampling; gate = A2 disjointness |
| B4 NL template engine | G | N | relation-matched D/A templates; gate = A4 |
| B5 Dummy-prefix injector | G | BB | reserved IDs, resampled per sequence, **frozen random embeddings**, never targets; gate = dummy-row gradient norm == 0 |
| B6 Bare-baseline transformer (decoder-only, dense softmax, single global LR, from scratch) | P→G strip | BB | procure a **minimal, auditable scaffold** (nanoGPT-class), strip to standard softmax + single LR; **no fused/sparse/robustness defaults**; gate = matched params/FLOPs reproduced; overfit-tiny-batch sanity |
| B7 Config variants (EE, TR, KV, HRL, LL) | G | BB | built on B6: entity-embedding table; RAG retrieval; KV slot memory; ID-restricted/lexical locks; gate = each lock provably blocks its forbidden channel |
| B8 Training harness (compute-matched, seed-managed, checkpointing) | P→G wrap | N | procure trainer (Lightning/Accelerate/raw); enforce matched-FLOP budget; deterministic seeding; gate = bitwise-reproducible loss curve on fixed seed at 10k tier |
| B9 Analysis & statistics pipeline | P→G wrap | N | mixed-effects (statsmodels/pymer4); FDR/Holm (statsmodels); covariate conditioning; gate = recovers injected effects on synthetic stats fixtures |

### C. Diagnostic instruments (the validity-gate library — **build first**)

| Artifact | G/P | Role | Acceptance gate (unit-tested on A11 fixtures) |
|---|---|---|---|
| C1 IPI estimator (+ five-part sufficiency conjunction) | G | COV | IPI ≈ 1.0 on a permutation-equivariant reference (a GNN); IPI ≈ 0 on a label-memorizer fixture |
| C2 Geometry tracker (anisotropy, uniformity, SVD/effective rank @1k steps) | G | COV | reproduces known anisotropy/effective-rank on synthetic collapsed-cone vs isotropic matrices |
| C3 Two-stage error decomposer (ER vs reasoning-given-correct-ID) | G | COV | recovers an injected (ER error, reasoning error) split exactly |
| C4 Shuffled-signal control | G | COV | returns null delta on a model fixture known to ignore position |
| C5 Anchor–explorer partitioner | G | COV | partitions a fixture with known entropy profile into correct 20/20 buckets |
| C6 Interdiction generators (anti-statistical / edge-swap / path-mask) | G | COV | generated items satisfy A6/A7 gates by construction |
| C7 Per-node / per-token update-count logger | G | COV | logged counts match a hand-computed reference on a tiny graph |
| C8 Length-matched efficiency harness (acc/FLOP, acc/GB primary) | G | COV | identical FLOP/GB accounting against LW+dummy on a fixed micro-run |
| C9 Held-out-diagnostic verifier | G | COV | flags any config whose objective references a diagnostic tensor |

### D. Model artifacts

| Artifact | G/P | Role | Acceptance gate |
|---|---|---|---|
| D1 Graph encoders (GraphSAGE, GAT, TransE, RotatE) for the ladder | P→G wrap | N | procure PyG/DGL (GNNs) + PyKEEN (TransE/RotatE); **reproduce a published sanity metric** (e.g., RotatE hits@10 on a standard KG benchmark within tolerance) to confirm correct integration |
| D2 Encoder-ladder rungs (G / G+probe / G+frozen-LLM) | G | N | linear probe + frozen-LLM wrappers; gate = frozen weights provably unchanged |
| D3 Trained checkpoints (all configs × tiers × seeds × resolution) | G | BB | each records data ID + config + seed + commit; matched-compute confirmed |
| D4 Ablation-arm components (ASEntmax; ADiT+TAR+BES; AGG; embedding-LR; KV-shifting) | P | ABL | procure reference implementations **where they exist and are verified**; each quarantined with its forbidden-diagnostic ledger; **ADiT availability must be confirmed before its ablation is committed** (canonical status uncertain — verify, else descope A2-ablation) |

### E. Compute & infrastructure

| Artifact | G/P | Role | Notes |
|---|---|---|---|
| E1 GPU compute (A100 80GB / B200 180GB cluster or cloud) | P | N | size from the matched-FLOP budget; reserve B200 for 1M-tier KV-cache headroom |
| E2 Object storage (datasets + checkpoints, content-addressed) | P | N | order-of-magnitude: tokenized walk corpora are tens–hundreds of GB per family·tier·seed; total TB-scale — compute exactly from matched-token budget |
| E3 Experiment tracking & config management | P | N | W&B/MLflow; immutable run config; seed registry |
| E4 Edge-canonical regeneration runtime (Node.js) | G | N | runs B1–B5 generators with zero infra to re-derive any dataset from manifest |

### F. Provenance & governance

| Artifact | G/P | Role | Notes |
|---|---|---|---|
| F1 Dataset datasheets (per data artifact) | G | N | generation spec, seed, distributions, known limitations |
| F2 Model cards (per checkpoint family) | G | N | data ID, config, compute, intended use, ablation status |
| F3 Regeneration manifests (`spec_hash + seed + commit → ID`) | G | N | one per dataset/checkpoint; enables bitwise re-derivation |
| F4 Pre-registration record (frozen hypotheses + decision rules + Tier-0 gates) | G | N | timestamped, immutable, pre-data |
| F5 Tier-0 evidence bundle (per run) | G | N | update-count match, geometry bounds, LW+dummy efficiency basis, held-out-diagnostic pass |

---

## 2. Build Specifications for Generated Artifacts

**Determinism contract.** Every generator takes `(spec, seed)` and is a pure function: identical inputs → bitwise-identical output, verified by hashing. No wall-clock, no nondeterministic parallelism in the data path.

**Topology generation (A1/B1).** BA with γ-target via attachment exponent; WS with rewiring probability tuned to target clustering at the constant ⟨k⟩; ER at $p=\langle k\rangle/N$. Generate at all three tiers per family; ER serves as the uniform-degree control isolating degree effects from topology-family effects.

**Frequency-matched walks (A3/B2) — the load-bearing bespoke artifact.** Standard random walks over $G_{SF}$ produce Pareto token frequencies (hub over-training, peripheral starvation) — the ST3 confound. The sampler instead equalizes **per-node visit counts** via degree-corrected transition probabilities or stratified uniform restarts, emitting a per-node visit log. Acceptance: visit-count coefficient of variation ≤ tolerance across nodes *and across families*, so a peripheral $G_{SF}$ node receives the same update budget as a typical $G_{SW}$ node. Imbalance is retained as a *reported* profile, never as the training regime.

**Reserved vocabularies & dummy prefix (A2/B3/B5).** Three provably disjoint partitions. Dummy tokens: reserved pool, **resampled per sequence**, **frozen random embeddings**, never targets — contributing length, KV-cache, and attention-sink anchoring and nothing else. Acceptance includes a runtime assertion that dummy-row gradients are exactly zero.

**Anti-statistical probes (A6/C6) — order-dependent.** These cannot be built until the corpus exists and is profiled, because each item must be constructed *against measured* co-occurrence: select/compose items where the corpus's lexical co-occurrence statistic points to the wrong entity while the ground-truth path points to the right one. Both conditions are verified per item; a yield report records how many are constructible at each scale (it may be that very small graphs cannot satisfy the constraint for some relations — that is a reported limitation, not a silent gap).

**NL corpus (A4/B4).** Relation-matched D/A templates over the synthetic facts (no causal framing — the relations are static). D/A pairs reference identical underlying facts so decision-invariance is measurable; negatives via controlled-plausibility edge corruption.

---

## 3. Procurement Specifications

**Wrap-don't-fork discipline.** Each procured dependency sits behind a thin adapter exposing only what the experiment needs, so (a) the bare baseline stays auditable, (b) versions can be pinned and swapped, (c) procured defaults can be stripped.

| Component | Candidate | Build-vs-procure rationale | At procurement: verify |
|---|---|---|---|
| Graph primitives | NetworkX / igraph / graph-tool | mature, correct, well-tested; perf-critical at 1M → igraph/graph-tool | version; 1M-scale memory/time; license |
| Training scaffold | nanoGPT-class minimal | inspectable, matched-compute controllable; heavyweight frameworks hide compute and inject defaults | strip sparse/fused defaults; determinism |
| Trainer/infra | Lightning / Accelerate / raw | reduces harness risk | deterministic seeding; checkpoint integrity |
| KG encoders | PyKEEN (TransE/RotatE), PyG/DGL (GraphSAGE/GAT) | standard, benchmarked; reimplementation is wasted risk | reproduce a published sanity metric |
| Sparse attention (ablation) | `entmax` library | only needed for A1 ablation; quarantined | maintenance status; correctness vs reference |
| ADiT + TAR + BES (ablation) | research code | only needed for A2 ablation | **availability + canonical status uncertain — confirm before committing; else descope A2-ablation** |
| Statistics | statsmodels / pymer4 | standard mixed-effects + FDR/Holm | mixed-effects convergence on the design |
| Tracking | W&B / MLflow | provenance + config immutability | data-retention/privacy settings |
| Compute | A100/B200 cluster or cloud | training is necessarily cluster-bound | quota; B200 KV-cache headroom at 1M |

---

## 4. Acceptance / Validation Gates (summary)

No artifact is used downstream until its gate passes. Gates are falsifiable, not subjective:

- **Graphs:** empirical degree distribution matches target within tolerance (γ-fit/KS for SF; clustering threshold for SW; Poisson for ER); exact N, M; connectivity.
- **Corpora:** per-node visit-count CoV ≤ tolerance across nodes and families; full coverage; bitwise re-derivation from seed.
- **Vocab/dummy:** provable partition disjointness; per-sequence resampling; frozen-embedding zero-gradient assertion.
- **Probes:** anti-statistical items verified against *both* measured corpus statistics and ground-truth paths; alias classes verified against their defining property.
- **Diagnostic instruments (C1–C9):** each passes a known-answer unit test on A11 fixtures (IPI extremes on equivariant vs memorizer; geometry on collapsed vs isotropic; decomposer recovers injected splits; shuffled-signal returns null on position-blind fixture; held-out verifier flags a planted violation).
- **Encoders (procured):** reproduce a published benchmark metric within tolerance — confirms integration, independent of the synthetic experiment.
- **Checkpoints:** matched-compute confirmed; full provenance recorded.

---

## 5. Dependency Graph and Build Sequence

The build order is constrained by two facts: instruments must precede runs, and probes depend on a profiled corpus.

**Phase α — Foundations & instruments (no GPU).**
1. F4 pre-registration frozen; F3 manifest scheme defined.
2. C1–C9 diagnostic library + A11 fixtures, fully unit-tested. *(Gate: every instrument passes known-answer tests before any training.)*
3. B1–B5 generators (edge-canonical, deterministic) + E4 regeneration runtime.

**Phase β — Data (CPU-heavy).**
4. A1 graphs (all tiers/families) → A2 vocab → A3 frequency-matched corpora (+ visit logs) → profile corpus.
5. A4 NL corpus; A5 alias sets; **then** A6 anti-statistical probes (require profiled A3); A7 swap/mask sets; A8 holdouts; A9 transfer pairs; A10 noise grid.
6. F1 datasheets + F3 manifests for all data.

**Phase γ — Vertical slice at the 10k tier (cheap GPU).**
7. B6 bare baseline + B7 variants + B8 harness; D1–D2 encoder ladder.
8. Run the **entire pipeline end-to-end at 10k only** — all configs, all instruments, all Tier-0 gates, the full decision-rule path — on a few seeds. *(Gate: the slice produces a complete, interpretable result set and every Tier-0 gate fires correctly. Fix the pipeline here, where runs are cheap, before scaling.)*

**Phase δ — Scale (expensive GPU).**
9. 100k then 1M tiers; scaling-divergence slope; cross-topology under frequency-matching; full seed count.
10. D3 checkpoints + F2 model cards + F5 evidence bundles per run.

**Phase ε — Ablations (quarantined).**
11. D4 ablation arms (A1–A7 register), each reporting only its permitted diagnostics; ADiT arm contingent on availability confirmation.

---

## 6. Provenance & Reproducibility Artifacts

- **Regeneration manifest (F3):** for each dataset/checkpoint, `spec_hash + seed + code_commit → content-addressed ID`. A reviewer reconstructs any corpus bit-for-bit from the manifest using the edge-canonical runtime (E4) — no cluster needed for data.
- **Datasheets (F1) / model cards (F2):** generation spec, distributions, limitations / data ID, config, compute, ablation status and forbidden-diagnostic ledger.
- **Pre-registration (F4):** frozen hypotheses, decision rules, Tier-0 gates, timestamped pre-data.
- **Tier-0 evidence bundle (F5), per run:** update-count match, geometry bounds, LW+dummy efficiency basis, held-out-diagnostic pass — the proof that the run is interpretable under the guardrail.

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ADiT (and TAR/BES) not cleanly available / not canonical | Medium | Low (ablation-only) | Confirm availability before committing; if absent, **descope the A2 ablation** and note it — the bare-baseline cross-topology result is unaffected |
| Anti-statistical probes unsatisfiable at some scale/relation | Medium | Medium | Yield report per scale; report coverage honestly; rely on edge-swap/path-mask where anti-statistical items are sparse |
| Frequency-matching tolerance unachievable on extreme hubs | Low–Med | High (ST3 reopens) | Importance-reweighting + restart tuning; if residual imbalance remains, carry per-node-norm covariate and report the residual rather than claim a clean match |
| Storage blow-up at 1M/25M (TB-scale corpora × seeds) | Medium | Medium | Compute exact token budget from matched-FLOP; stream/shard; store seeds+manifests, regenerate on demand rather than persisting all corpora |
| Contamination leak via a procured tokenizer/pretrained default | Low | High | From-scratch training only; tokenizer built in-house (B3); no pretrained checkpoints anywhere in the baseline |
| Procured component injects a robustness default into the baseline | Medium | High | Wrap-don't-fork; strip to standard softmax/single LR; B6 audit gate |
| Diagnostic discovered missing after expensive runs | Low | Very High | **Instruments-before-runs** sequencing (Phase α); vertical-slice gate (Phase γ) before scaling |
| Mixed-effects model fails to converge on the full design | Medium | Medium | Cluster-robust paired fallback pre-registered; validate on stats fixtures (B9) before real data |

---

## 8. Minimal First Milestone (the vertical slice)

Before any spend at scale, produce the smallest artifact set that exercises the **entire** pipeline:

- $G_{SF}$, $G_{SW}$, $G_{ER}$ at the **10k tier** only (A1); vocab (A2); frequency-matched corpora + visit logs (A3); a small NL set (A4); one alias class + a handful of anti-statistical probes (A5/A6).
- The **full diagnostic library** (C1–C9) passing fixtures.
- Bare baseline + all six configs + the encoder ladder (B6/B7/D1–D2), trained a few seeds.
- One complete pass through every Tier-0 gate and the full decision-rule mapping.

**Exit criterion:** the slice yields a complete, interpretable result table with every validity gate firing and the guardrail enforced — at which point scaling to 100k/1M is a matter of compute, not of unresolved design. This is the cheapest place to discover that an instrument, a probe generator, or a gate is wrong.

---

## Appendix — Build/Procure Decision Summary

**Generate (load-bearing for validity):** all data (contamination-free, regenerable); the frequency-matched walk sampler; the reserved-vocab/dummy machinery; the bare baseline and config variants; the entire diagnostic-instrument library; all provenance records.

**Procure + wrap (mature, neutral infra):** graph primitives; a minimal stripped training scaffold; KG-embedding encoders for the ladder; statistics; tracking; compute.

**Procure + quarantine (ablation-only):** ASEntmax, ADiT/TAR/BES, AGG, embedding-LR, KV-shifting — each admitted solely as a labeled ablation under the construct-validity guardrail, reporting only its permitted diagnostics, with ADiT contingent on availability confirmation.
