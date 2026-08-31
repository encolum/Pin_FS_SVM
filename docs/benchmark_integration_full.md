# Full benchmark integration — implementation report

This completes the remaining **code scope** of the supplied implementation plan
and technical specification after Milestones 0–2. Experimental approval and
full-size pilot evidence are separate and are not claimed complete.

## Implemented

- **Sparse core/preprocessing:** CSR-safe validation, prediction, Pin-FS MILP rows,
  full/restricted solving, cache/data hashes and kernel engine; the MILP matrix
  remains sparse. Standard, sparse standard (no centering), MaxAbs, none and
  upstream-normalized passthrough policies fit only on training observations.
  Densification needs explicit authorization and a byte cap, applied at both fit
  and transform. Active support still uses `abs(w) > 1e-3`.
- **Signals:** sparse Fisher moments; explicit discrete/continuous MI handling;
  two-dimensional correlation chunks and chunked support redundancy. No full
  feature-by-feature correlation matrix is allocated. Disabled/skipped signals
  carry reasons. Budget checks are made between work units; online signal and
  solver model-build costs are included. Optional SciPy Pin signals are skipped
  in timed routes because SLSQP lacks an enforceable wall-clock limit.
- **Instance integration:** `kind: benchmark` uses verified original loaders;
  `dataset`/`legacy_dataset` retain the separate manuscript path. Source policy,
  research split and outer fold have separate metadata. All six benchmarks can
  become `PolicyInstance`s without calling a solver. Official holdout never
  enters fitting/tuning. Source hashes, labels, stable IDs and preprocessing
  parameters travel into each run manifest.
- **Synthetic/corruption:** clean base generator; the old embedded-corruption API
  remains explicitly legacy. Experimental corruption is generated only after
  splitting and train-only preprocessing. Mixed noise, actual feature outliers,
  their combination and an explicitly named optional legacy label attack are
  supported. Masks, parameters, seeds, train/test hashes and source row IDs are
  persisted. Routes share the same prepared training object. Reused clean source
  data across distinct research groups is rejected.
- **Classification:** nested stratified 5 × 3 protocol for scientific benchmark
  runs; smaller folds only for explicit provisional QA. Inner Balanced Accuracy
  selects `B/C/tau` with a full/reduced Pin-FS solver, without ADKS/VeraPin tuning.
  Parameters and corrupted outer-training input are frozen before the three
  routes run. Each route's incumbent is evaluated on clean outer-test data.
- **Fair reporting:** full-model-only bounds/gaps remain separate from restricted
  bounds. Common best-known objective and a shared horizon are used for post-hoc
  route primal integrals, including early stopping and online overhead. Build
  time, nodes, runtime, memory, selected features, status and trajectories are
  retained. License failures are recorded and never labeled nontrivial hardness.
- **Profiles/gates:** complete provisional real/synthetic hardness and ADKS configs;
  explicit instance selection for sequential real pilots; `--validate-only`
  without fitting/solving; scientific evolution/final templates updated for the
  clean/new-benchmark paths. Evidence and author-decision gates prevent automatic
  ADKS, evolution or full classification execution.

The Pin-FS objective/constraints, ADKS scoring formula, policy DSL/sandbox and
VeraPin search mathematics were not changed. No raw dataset files were changed
or deleted and no archived results were removed.

## Verification and actual execution

Baseline: 208 passed / 3 skipped. Full integration: **256 passed / 3 skipped**.
Full-suite checks cover dense/CSR equivalence
with SciPy and CPLEX, all six real adapters, safe preprocessing, blocked
densification, sparse signal equivalence, nested fold isolation, deterministic
corruption, research-group separation and three-route end-to-end final QA.
The three legacy manuscript-data tests remain skipped because their external
files are absent; retained-benchmark tests are not skipped.

A real local CPLEX **software pilot**, not a paper benchmark, ran on a clean
40-observation / 20-feature synthetic instance, with provisional `B=5`, `C=1`,
`tau=0.5`, bounds `[-2, 2]`, one thread and a five-second cap:

| Measurement | Observed |
| --- | ---: |
| Solver status | optimal |
| Objective / bound | 16.1591510116 |
| Final gap | 0 |
| Nodes | 521 |
| Total route runtime | 0.22735 s |
| Model build time | 0.07882 s |
| First incumbent | 0.08758 s |
| Selected features | 5 |

Local run: `results_verapin/20260831-202702-hardness/`. Its config snapshot,
source-generation parameters, preprocessing/corruption manifests, progress,
selected features and solver diagnostics are saved. This timing is one local
observation, not a performance/generalization claim.

The installed CPLEX **22.1.2 Community Edition** rejected a bounded 1001-variable
license probe with **error 1016**. Therefore full-size Hill-Valley/Madelon and
later real pilots are blocked by license capacity; no subsampling was used to
manufacture a pass. No full benchmark or LLM evolution/API call was launched.
Only one tiny hardness instance was measured, so the two-hard-instance ADKS gate
is not satisfied.

## Milestone status

| Scope | Status |
| --- | --- |
| 0–2: baseline, registry, adapter, validation | Complete |
| 3–6: sparse support/signals, integration, train-only corruption | Code and tests complete |
| 7: pilot profiles and decision tracking | Implemented; scientific choices still provisional/unapproved |
| 8: validation and hardness execution | Tiny synthetic pilot passed; full-size real pilots license-blocked |
| 9: ADKS pilot workflow and gates | Implemented and fixture-tested; real pilot not authorized by evidence yet |
| 10: evolution readiness/final configuration | Implemented; author decisions/frozen artifacts required |
| Final nested classification/corruption protocol | Implemented and tiny end-to-end tested; full study not run |

## Decisions still required

Obtain a suitable CPLEX license and decide version parity; approve bounds,
hyperparameter grids, seeds, solver budgets, preprocessing overrides and any
corruption profiles; identify at least two genuinely hard instances; review/freeze
ADKS and the final method; choose/authorize any LLM provider and offline budget.
See `AUTHOR_DECISIONS_REQUIRED.md`. The implementation does not make these author
decisions on the user's behalf.

Generated verification reports remain in ignored `artifacts_v2/`; dataset source
files and their original manifest remain untouched. The README documents the
exact validation and sequential-pilot commands.
