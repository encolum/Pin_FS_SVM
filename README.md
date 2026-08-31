# Pin-FS-SVM: corrected paper-aligned implementation

This repository contains the refactored experimental pipeline for the Pin-FS-SVM
manuscript. The executable implementation is centered on `main.py`, the corrected
models in `src/models/corrected/`, and the experiment definitions in `configs/`.

The checkout includes compact public benchmark inputs under `dataset/`, with
provenance, checksums, and explicit partitions in [dataset/README.md](dataset/README.md).
The original manuscript data under `Dataset/Dataset/`, local environments,
generated results, plots, reports, and logs remain excluded from version control.

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

The newly bundled benchmarks can be checked immediately, without a solver or
experiment run:

```bash
.venv/bin/python main.py validate-datasets
```

Use `src.data.load_benchmark_dataset(name, partition=...)` for these inputs.
They are deliberately separate from the original six-dataset manuscript loader;
existing experiment configs have not been changed to silently substitute them.
The LIBSVM Colon export is already normalized upstream and must not be represented
as raw input for a strict train-only-preprocessing reproduction.

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

VeraPin-KS infrastructure commands are separate from the manuscript benchmark:

```bash
# Cold-CPLEX hardness profiling
.venv/bin/python main.py hardness --config configs/hardness.yaml

# Static KS and handcrafted ADKS pilots
.venv/bin/python main.py kernel-search --config configs/static_ks_pilot.yaml
.venv/bin/python main.py adks --config configs/adks_pilot.yaml

# Train-only evolution, validation-only freezing, and replay audit
.venv/bin/python main.py evolve-verapin --config configs/verapin_evolution.yaml
.venv/bin/python main.py replay-evolution --run-dir results_verapin/<run_id>

# Held-out comparison; this path never creates an LLM provider
.venv/bin/python main.py evaluate-verapin \
  --config configs/verapin_final.yaml --confirm-full-run
```

The distributed VeraPin configs deliberately contain `null` author-decision gates.
The CLI lists every unresolved field before creating a run directory. Record the
choices in `AUTHOR_DECISIONS_REQUIRED.md` and the relevant config before running;
the repository does not silently invent scientific parameters.

Final classification metrics use deterministic stratified outer folds. Every
route is optimized on the outer-training partition, preprocessing is fitted only
there, and Balanced Accuracy/F1/Accuracy/G-mean are computed on the untouched
outer-test partition. In-sample optimization data are not reported as
classification test results.

## VeraPin-KS solver and search API

`src.search` exposes the paper formulation as reusable solver data and supports
feature-kernel restrictions without changing the original objective or constraints.
Features outside the kernel are fixed with `v_j = 0`; active features are always
derived from `abs(w_j) > 1e-3`, not from the binary selector alone.

```python
from src.search import (
    build_pin_fs_problem,
    result_to_mip_start,
    solve_restricted_pin_fs,
)

restricted = solve_restricted_pin_fs(
    X,
    y,
    kernel={0, 2, 5},
    B=2,
    C=1.0,
    tau=0.5,
    coefficient_bounds=(-2.0, 2.0),
    backend="cplex",
    time_limit=60.0,
    mip_gap=0.01,
    threads=1,
    collect_progress=True,
)

full_problem = build_pin_fs_problem(
    X,
    y,
    B=2,
    C=1.0,
    tau=0.5,
    lower_bound=-2.0,
    upper_bound=2.0,
)
warm_start = result_to_mip_start(restricted, full_problem)
```

The returned result includes the complete decision vector split into `w`, `b`,
`z`, `xi`, and `v`, solver diagnostics, CPLEX MIP-start status, and a timestamped
incumbent/bound/gap/node trajectory. Malformed starts raise a validation error and
are never silently ignored. The original `PinFSSVM.fit(X, y)` API remains unchanged.

The same `run_kernel_search` engine executes Static KS, handcrafted ADKS, and a
frozen VeraPin policy. Policy formulas live outside the engine, incumbent support
is retained, and signal, LP-relaxation, policy, MIP-start, restricted-solve, and
final-refinement time all count against one wall-clock budget. Static and ADKS
weights are deterministic and config-driven.

Restricted-kernel bounds and MIP gaps are retained only as per-iteration local
diagnostics; they are never spliced into the full-model trajectory. Route-level
gap fields remain empty until a final unrestricted refinement supplies a comparable
full-model bound. After all comparison routes finish, primal integrals are
recomputed with the same best-known feasible objective for that instance.

VeraPin candidates are typed JSON expression trees. The bounded interpreter allows
only arithmetic, clipping, and conditionals over an explicit signal allowlist. It
does not execute generated Python or expose imports, files, network, subprocesses,
reflection, or loops. Evolution prompts contain training summaries only; policy
selection uses validation instances; `evaluate-verapin` loads the frozen JSON and
makes no LLM call. Prompts, responses, token usage, latency, estimated cost,
checkpoints, and cache keys are retained for audit/replay.

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
dataset/                        curated numeric benchmarks, metadata, and inventory
scripts/curate_datasets.py       lossless conversion from original uploaded files
src/data/                       loading, validation, partitioning, and corruption
src/models/corrected/           corrected proposed and baseline estimators
src/search/                     restricted Pin-FS builder, MIP starts, and progress
src/search/policies/            Static KS, handcrafted ADKS, and frozen VeraPin
src/search/llm_evolution/       safe DSL, providers, evaluation, cache, evolution
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
VeraPin runs and frozen-policy artifacts use `results_verapin/` and
`artifacts_verapin/`, which are also intentionally untracked.

## Verification

After installing the dependencies (only legacy data integration tests require
restoring the original manuscript datasets):

```bash
PYTHONPYCACHEPREFIX=/tmp/pinfs-pycache .venv/bin/python -m pytest -q
```

Before launching the full experiment, resolve every `null` corruption parameter in
`configs/full.yaml`, review the preserved `[-2, 2]` coefficient-bound assumption,
and set `coefficient_bounds.author_confirmed` to `true` only after confirmation.
