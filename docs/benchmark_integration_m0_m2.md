# Benchmark integration: Milestones 0–2

> Historical milestone report. The duplicate manuscript pipeline described as
> retained at this stage was later archived and removed from active `main` on
> 2026-09-02. See `docs/main_cleanup.md`.

Implemented against the supplied VeraPin Benchmark Integration Technical
Specification and Implementation Plan. The plan's section 18 explicitly limits
the immediate work to Milestones 0–2; the technical specification's section 21
also lists later preprocessing/experiment integration. This change follows the
plan's narrower staged scope, not the complete roadmap.

## Changes

- Added a strict six-dataset YAML registry, rejecting duplicate keys, unknown
  fields, invalid mappings and unsupported partition/preprocessing combinations.
- Added `SolverReadyBenchmark` and an adapter reusing existing raw parsers. It
  verifies file hashes and runtime measurements against `dataset/manifest.json`.
- Explicitly maps native labels into a new int64 `{-1, +1}` array. Original
  labels, feature values, dtypes and file bytes remain unchanged.
- Added deterministic `merge_labeled`, `official_holdout` and already-pooled
  `pool` handling. No partition is guessed. API overrides are recorded.
- Added IDs `dataset:source_partition:zero_based_row_index`, source and registry
  hashes, storage/memory metadata and provenance warnings.
- Added the read-only `validate-benchmarks` command with optional JSON output.
  Raw validation remains separate; no scaler, solver or LLM is invoked by this command.
- Added 70 tests covering all six real datasets, native value preservation,
  source checksums, mappings, split separation, malformed/stale metadata,
  non-finite inputs, sparse handling and CLI/report safety.

No files were removed. No dataset, manuscript loader, existing experiment config,
search algorithm, solver formulation or author-confirmation gate was modified.

## Validation results

Baseline commit: `b29c538d84e699f0b33474887782bd48b63ceba6`.
Baseline tests: **138 passed, 3 skipped**.
After integration: **208 passed, 3 skipped** (10.17 seconds).
The three skips require unavailable legacy manuscript files under
`Dataset/Dataset`; all retained-benchmark tests ran.

All six default solver-facing views passed. Density means the fraction of
feature entries numerically nonzero, not allocated sparse entries.

| Dataset | Samples × features | Negative | Positive | Storage | Density | Partition policy |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Hill-Valley | 1,212 × 100 | 600 | 612 | dense | 100.000000% | merge_labeled |
| Madelon | 2,600 × 500 | 1,300 | 1,300 | dense | 99.999923% | merge_labeled |
| GINA | 3,468 × 970 | 1,763 | 1,705 | dense | 30.945582% | pool |
| HIVA | 4,229 × 1,617 | 4,080 | 149 | dense | 9.081360% | pool |
| Colon | 62 × 2,000 | 40 | 22 | CSR | 100.000000% | pool |
| BASEHOCK | 1,993 × 4,862 | 994 | 999 | CSR | 1.385485% | pool |

No missing or infinite feature/label values were found. Optional official holdout
views also passed: Hill-Valley 606 training / 606 test rows; Madelon 2,000 training /
600 validation rows. Train and holdout IDs are disjoint, with original row order.

JSON reports and test logs remain in ignored `artifacts_v2/`, not mixed into the
original `dataset/` directory:

- `pre_integration_test_report.txt`
- `pre_integration_dataset_report.json`
- `milestone_0_report.md`
- `benchmark_integration_test_report.txt`
- `benchmark_integration_validation.json`
- `post_integration_dataset_report.json`

## Contract details and discrepancies resolved

The specification's table describes BASEHOCK as native sparse. The supplied MAT
actually contains dense uint8 X. The adapter records an explicit dense-to-CSR
conversion in memory and preserves every numerical value; the MAT is untouched.

For `official_holdout`, the specification does not prescribe a separate return
field. This implementation adds `SolverReadyPartition` as `benchmark.holdout`.
Top-level `X`, `y` and `sample_ids` contain training rows only. Metadata identifies
both source partitions and the matrix role. Callers cannot accidentally receive
the holdout concatenated into training through this policy.

The registry's `pool` value applies only to already-pooled sources. For those
sources, an explicit `merge_labeled` request is also a single-pool view, with no
invented rows or splits. `official_holdout` fails when split indices are absent.

## Deferred work and author decisions

"Solver-ready" denotes the label/storage/partition interface, not readiness for
scientific runs. Standard/MaxAbs/passthrough policies are **declarations only**.
Sparse-safe train-only preprocessing, sparse solver changes, `kind: benchmark`
dispatch, clean hardness pilot configuration and subsequent roadmap milestones
are not implemented here. No hardness, ADKS, evolution or full experiment was run.

Before later integration/runs:

- Confirm extending scope beyond Milestone 2 and review the adapter contract.
- Choose the experimental split policy. Merging labeled official partitions
  requires new outer splits; pooled GINA/HIVA exports have no original split IDs.
- Confirm coefficient bounds, `C`, `tau`, budget derivation, solver limits/gaps,
  split seeds/folds and the CPLEX runtime/license for the intended model sizes.
- Keep Colon's upstream normalization caveat; do not claim it was raw input to a
  train-only preprocessing protocol.
- Decide sparse preprocessing/memory limits before integrating solver support.
- Resolve corruption/evolution settings separately in the active configurations;
  no scientific values were guessed here.

Reproduce the checks using the project's Python environment:

```bash
python -m pytest -q -rs
python main.py validate-datasets
python main.py validate-benchmarks --registry configs/benchmark_registry.yaml
```
