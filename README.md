# Pin-FS-SVM and VeraPin-KS

This repository contains the active implementation for:

- the paper-aligned Pin-FS-SVM formulation;
- Handcrafted Adaptive Kernel Search (ADKS);
- VeraPin-KS policy evolution and frozen-policy evaluation.

The former manuscript-reproduction pipeline has been removed from `main` to
avoid maintaining two conflicting experiment paths. Its complete source remains
recoverable from branch `archive/manuscript-v2` and tag
`pre-verapin-cleanup`.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

A full-size CPLEX license is still required for the intended benchmark models.
The previously observed Community Edition limit must not be bypassed through
unreported row or feature reduction.

## Retained benchmark data

`dataset/` contains the original selected uploads, their provenance and SHA-256
checksums. No loader rewrites these files. Hill-Valley retains the original
without-noise train/test partitions; Madelon retains labeled train/validation;
BASEHOCK, Colon, GINA and HIVA retain their supplied pooled source.

Run both read-only audits before any experiment:

```bash
.venv/bin/python main.py validate-datasets
.venv/bin/python main.py validate-benchmarks \
  --registry configs/benchmark_registry.yaml
```

`validate-datasets` verifies the 10 source files and eight native partitions.
`validate-benchmarks` checks the six solver-facing views, explicit label maps,
partition policies, storage and declared preprocessing. Optional JSON reports
must be written outside `dataset/`; the tools refuse to overwrite an original
input, its manifest or the registry.

The solver-ready API requires an explicit policy:

```python
from src.data import load_solver_ready_benchmark

pool = load_solver_ready_benchmark("basehock", partition_policy="pool")
holdout = load_solver_ready_benchmark(
    "hill_valley", partition_policy="official_holdout"
)
```

Original labels are mapped into a new int64 `{-1, +1}` vector. Stable sample IDs
preserve dataset, source partition and original row index. BASEHOCK is converted
from its dense MAT representation to CSR in memory; Colon remains CSR. Source
bytes, native labels and feature values are unchanged.

## Commands

The CLI exposes only the active pipeline:

```text
validate-datasets
validate-benchmarks
hardness
adks
evolve-verapin
evaluate-verapin
verify-policy
replay-evolution
```

Configuration validation is solver-free and creates no run directory:

```bash
.venv/bin/python main.py hardness \
  --config configs/hardness_real_pilot.yaml --validate-only
.venv/bin/python main.py hardness \
  --config configs/hardness_synthetic_pilot.yaml --validate-only
.venv/bin/python main.py adks \
  --config configs/adks_real_pilot.yaml --validate-only
```

The real hardness pilot requires an explicit single instance. After reviewing
the provisional settings and installing an adequate CPLEX license:

```bash
.venv/bin/python main.py hardness \
  --config configs/hardness_real_pilot.yaml --instance hill-valley-clean
```

Continue with other IDs separately. Do not launch ADKS/evolution/final runs
until every unresolved `null` gate and provisional setting in the selected
configuration has been reviewed and resolved.

Core workflow commands are:

```bash
# Cold full Pin-FS hardness profiling
.venv/bin/python main.py hardness \
  --config configs/hardness_real_pilot.yaml --instance hill-valley-clean

# Handcrafted ADKS
.venv/bin/python main.py adks --config configs/adks_real_pilot.yaml

# Train-only evolution, validation-only policy freezing, and offline replay
.venv/bin/python main.py evolve-verapin --config configs/verapin_evolution.yaml
.venv/bin/python main.py replay-evolution --run-dir results_verapin/<run_id>

# Held-out comparison; this route never constructs an LLM provider
.venv/bin/python main.py evaluate-verapin \
  --config configs/verapin_final.yaml --confirm-full-run
```

## Scientific protocol

- Benchmark and synthetic inputs are distinct typed instance kinds.
- Scientific final evaluation uses 5 outer × 3 inner stratified folds.
- Preprocessing is fitted only on the applicable training partition.
- Corruption is generated only after splitting and only on training data.
- Inner Balanced Accuracy selects `B/C/tau` using full/reduced Pin-FS only.
- Ties prefer fewer active features (`abs(w) > 1e-3`), then smaller `B`,
  then deterministic parameter order.
- Cold CPLEX, ADKS and frozen VeraPin share the same prepared outer-training
  input and are evaluated on the same untouched outer-test partition.
- Restricted MILP gaps remain local diagnostics; they are not connected across
  different feasible regions. Comparable route gaps require full refinement.
- Classification metrics are Balanced Accuracy, weighted F1, Accuracy and
  G-mean on held-out data, never the solver training matrix.

Evolution precomputes a policy-independent reference for every train/validation
instance from cold full Pin-FS and fixed Handcrafted ADKS. Both baselines receive
the same `solver.total_time_limit` as every candidate. The best feasible
in-budget objective is frozen in `fitness_references.json`; absence of a valid
reference aborts before any LLM provider is created. All policies use the same
horizon, including the tail after early stopping. Reference, horizon, scoring
protocol, target gap and failure normalization participate in cache/checkpoint
identity. Resume requires the original reference artifact.

Corruption protocol v2 supports `label_noise`, `mixed`, `feature_outlier`,
`combined`, and an explicitly named optional high-margin label attack. Sparse
feature noise samples only numerical nonzeros and never fills structural zeros.
Additive and multiplicative masks are disjoint. Manifests distinguish selected
cells from actual changes and report effective rates. Outlier magnitude is
`scale × std_j`, with population standard deviation fitted on clean preprocessed
training data; constant features remain unchanged and combined corruption freezes
the scale before its first stage.

## Pin-FS and Kernel Search APIs

`src.search` exposes the full/restricted formulation, MIP-start conversion,
progress trajectories and the shared kernel engine:

```python
from src.search import build_pin_fs_problem, result_to_mip_start, solve_restricted_pin_fs

restricted = solve_restricted_pin_fs(
    X, y,
    kernel={0, 2, 5},
    B=2, C=1.0, tau=0.5,
    coefficient_bounds=(-2.0, 2.0),
    backend="cplex",
    time_limit=60.0,
    mip_gap=0.01,
    threads=1,
    collect_progress=True,
)

full_problem = build_pin_fs_problem(
    X, y, B=2, C=1.0, tau=0.5,
    lower_bound=-2.0, upper_bound=2.0,
)
warm_start = result_to_mip_start(restricted, full_problem)
```

VeraPin candidates are typed JSON expression trees interpreted by a bounded DSL.
Generated policies cannot execute Python, imports, files, network, subprocesses,
reflection or loops. Prompts receive training summaries only; freezing uses the
validation research group; held-out final evaluation loads frozen JSON and does
not call an LLM.

## Repository layout

```text
main.py                         active command-line interface
configs/                        VeraPin, hardness, ADKS and registry configs
dataset/                        original uploads, provenance and checksums
src/data/                       benchmark adapters, preprocessing, corruption
src/evaluation/                 classification metrics and feature stability
src/experiments/                preparation, tie-break, gates and orchestration
src/models/                     retained Pin/L1/MILP model implementations
src/reporting/                  route tables and solver profiles
src/search/                     restricted solver and shared kernel engine
src/search/policies/            Static KS debug policy, ADKS and frozen VeraPin
src/search/llm_evolution/       safe policy DSL, evaluation, cache and evolution
src/utils/                      config, matrix, seed and serialization helpers
tests/                          active-pipeline regression tests
```

Generated results and validation reports remain ignored under `results_verapin/`
and `artifacts_verapin/`.

## Verification

```bash
PYTHONPYCACHEPREFIX=/tmp/pinfs-pycache .venv/bin/python -m compileall -q main.py src
PYTHONPYCACHEPREFIX=/tmp/pinfs-pycache .venv/bin/python -m pytest -q
.venv/bin/python main.py validate-datasets
.venv/bin/python main.py validate-benchmarks
```

These checks do not authorize or launch hardness, ADKS, evolution or final
experiments.
