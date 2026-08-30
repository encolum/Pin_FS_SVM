# Pin-FS-SVM: corrected paper-aligned implementation

This repository contains the refactored experimental pipeline for the Pin-FS-SVM
manuscript. The executable implementation is centered on `main.py`, the corrected
models in `src/models/corrected/`, and the experiment definitions in `configs/`.

This is a source-only checkout. Local environments, datasets, generated results,
plots, reports, and logs are deliberately excluded. Recreate them locally when
needed; `.gitignore` prevents those artifacts from being committed.

The mathematical formulations follow the manuscript, while the evaluation protocol
uses the corrected Phase 1 design: nested stratified cross-validation, clean
train-only scaling, then partition-local corruption of standardized training data.
This intentionally improves on the manuscript's non-nested evaluation and does not
claim numerical reproduction of its archived tables.

The paper does not provide numeric coefficient bounds or complete corruption rates
and distributions. Full, sensitivity, and ablation runs remain guarded until those
values are confirmed in `AUTHOR_DECISIONS_REQUIRED.md`. The pilot defaults to
SciPy/HiGHS for portable software QA. Full, sensitivity, and ablation configurations
use the optional DOcplex/CPLEX backend with one solver thread, matching the
manuscript's reported solver family.

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For manuscript-level solver parity, install the optional CPLEX environment and
ensure that the local IBM license permits the planned model sizes:

```bash
.venv/bin/python -m pip install -r requirements-cplex.txt
```

Restore the manuscript dataset files under `Dataset/Dataset/` using the names
declared in `src/data/loaders.py`. Then validate and run the pilot:

```bash
.venv/bin/python main.py validate --config configs/pilot.yaml
.venv/bin/python main.py pilot --config configs/pilot.yaml
```

Calling `main.py` without a command displays the available commands. Every run
prints its experiment matrix and estimated fit count before training.

## Commands

```bash
# Audit every dataset and validate a configuration
.venv/bin/python main.py validate --config configs/pilot.yaml

# Run the small end-to-end verification experiment
.venv/bin/python main.py pilot --config configs/pilot.yaml

# Resume an interrupted experiment
.venv/bin/python main.py pilot --config configs/pilot.yaml --resume results_v2/<run_id>

# Run the full benchmark after resolving all author-confirmation gates
.venv/bin/python main.py run --config configs/full.yaml --confirm-full-run

# Generate analysis artifacts from a completed run
.venv/bin/python main.py analyze --run-dir results_v2/<run_id>
.venv/bin/python main.py plot --run-dir results_v2/<run_id>
.venv/bin/python main.py statistics --run-dir results_v2/<run_id>
```

Sensitivity and ablation experiments use `configs/sensitivity.yaml` and
`configs/ablation.yaml`, respectively.

## Experimental protocol

- Full experiments use 5 outer and 3 inner stratified folds.
- Scaling is fit only on each training partition.
- Inner model selection maximizes Balanced Accuracy.
- Ties are resolved by fewer active features, smaller `B`, then parameter order.
- Fisher thresholds (25th, 50th, and 75th percentiles) are selected inside inner CV.
- Outer test folds remain clean unless a configuration explicitly says otherwise.
- Predictions are restricted to `{-1, +1}`.
- Reported metrics are Balanced Accuracy, weighted F1, Accuracy, and G-mean.
- Seeds, fold indices, corruption manifests, hyperparameter searches, predictions,
  scores, coefficients, selected features, and solver diagnostics are saved.
- Optional Wilcoxon tests with Benjamini-Hochberg correction run after training and
  are not part of model selection.

## Repository layout

```text
main.py                         command-line entry point
configs/                        pilot, full, sensitivity, ablation, and dataset specs
src/data/                       loading, validation, partitioning, and corruption
src/models/corrected/           corrected proposed and baseline estimators
src/evaluation/                 nested cross-validation and metrics
src/experiments/                configuration, search, registry, and run orchestration
src/statistics/                 post-hoc statistical analysis
src/reporting/                  tables and plots from saved results
src/utils/                      manifests, seeds, serialization, and logging
tests/                          automated validation of the corrected pipeline
```

Generated run directories are created under `results_v2/` and are intentionally
untracked. Cross-run reports and logs may likewise be written under `artifacts_v2/`
and `logs_v2/` without becoming part of the source repository.

## Verification

After restoring the datasets and installing the dependencies:

```bash
PYTHONPYCACHEPREFIX=/tmp/pinfs-pycache .venv/bin/python -m pytest -q
```

Before launching the full experiment, resolve every `null` corruption parameter in
`configs/full.yaml`, review the preserved `[-2, 2]` coefficient-bound assumption,
and set `coefficient_bounds.author_confirmed` to `true` only after confirmation.
