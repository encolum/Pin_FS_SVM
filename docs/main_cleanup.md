# Main-branch cleanup inventory

Cleanup date: 2026-09-02. Pre-cleanup commit:
`2b1a72b9ee22862cd8b336c6468139b686e371c6`.

The complete former tree is preserved on GitHub in branch
`archive/manuscript-v2` and tag `pre-verapin-cleanup`. No generated/archived
result directory and no file under `dataset/` was deleted or changed.

## Retained on main

- all 10 original files under `dataset/`, plus its README and manifest;
- six raw benchmark loaders, registry, solver adapter and validations;
- train-only preprocessing, deterministic corruption and synthetic generation;
- Pin-FS-SVM, L1-SVM, Pin-SVM, Budgeted MILP-SVM and shared solver base/CPLEX code;
- the full `src/search/` tree: restricted solver, MIP starts, progress, signals,
  Static KS, Handcrafted ADKS and VeraPin evolution/DSL/cache/replay;
- classification metrics, feature stability, route reporting and solver profiles;
- VeraPin preparation, selection, readiness gates and orchestration;
- all VeraPin/hardness/Static-KS configs, documentation and active tests.

`src/models/corrected/` keeps its existing name to avoid an unrelated import-path
migration. Only legacy baseline files inside it were removed.

## Removed from main

Legacy configurations:

- `configs/ablation.yaml`, `configs/datasets.yaml`, `configs/full.yaml`,
  `configs/pilot.yaml`, `configs/sensitivity.yaml`.

Legacy manuscript data/evaluation/orchestration:

- `src/data/loaders.py`;
- `src/evaluation/nested_cv.py`, `src/evaluation/predictions.py`;
- `src/experiments/config.py`, `registry.py`, `runner.py`, `search.py`,
  `sensitivity.py`.

Legacy baselines and ablations:

- `src/models/corrected/ablations.py`, `fisher_l1_svm.py`,
  `l1_svm_rfe.py`, `l2_svm.py`.

Legacy reporting/statistics and runner-only utilities:

- `src/reporting/aggregate.py`, `plots.py`, `tables.py`;
- `src/statistics/__init__.py`, `wilcoxon.py`;
- `src/utils/logging.py`, `manifests.py`.

Legacy-only tests:

- `tests/test_loaders.py`, `test_nested_cv.py`, `test_runner.py`,
  `test_search.py`, `test_wilcoxon.py`.

The old CLI commands `validate`, `pilot`, `run`, `sensitivity`, `ablation`,
`statistics`, `analyze` and `plot` were removed with their implementation.
Unused direct dependencies `statsmodels` and `matplotlib` were also removed.

## Required refactors completed first

- General config loading moved to `src/utils/config.py`.
- The active support/B/parameter tie-break moved to
  `src/experiments/selection.py`.
- VeraPin accepts only `kind: benchmark` and `kind: synthetic`; its legacy
  manuscript-loader branch was removed.
- Package exports, active tests and documentation were updated before validation.

## Post-cleanup verification

- `compileall` passed for `main.py` and every retained `src/` module;
- all active modules imported successfully;
- **282 tests passed, 0 skipped**;
- original-data validation passed **8/8 partitions across 6 dataset groups**;
- solver-ready validation passed **6/6 benchmarks**;
- the tracked `dataset/` tree has no diff;
- `ruff` and `vulture` were not installed in the configured environment, so they
  were reported as unavailable rather than silently installed or claimed as run.

No hardness, ADKS, evolution, final experiment or LLM call was launched during
cleanup validation.
