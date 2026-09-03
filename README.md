# Pin-FS-SVM and VeraPin-KS

## A reproducible framework for feature-selective pinball-loss SVMs and adaptive kernel search

### Research status

This repository contains the active research implementation of Pin-FS-SVM,
Handcrafted Adaptive Kernel Search (ADKS), and VeraPin-KS policy evolution. It
defines the optimization models, experimental protocol, validation gates, and
machine-readable reporting needed for a reproducible study.

## Abstract

Feature-selective support vector machines can improve interpretability and
reduce prediction cost, but their mixed-integer formulations become difficult
as the number of candidate features grows. This project studies a
pinball-loss SVM with an explicit cardinality budget and bounded coefficients.
It compares a cold full mixed-integer solve with restricted kernel-search
strategies that optimize over smaller feature subsets. The implemented methods
include a deterministic Static Kernel Search baseline, a handcrafted adaptive
policy, and VeraPin-KS: a policy represented by a bounded JSON expression
language and evolved using training instances only. The evaluation design uses
nested stratified cross-validation, train-only preprocessing and corruption,
shared solver budgets, fixed anytime-performance references, and held-out
classification metrics.

## 1. Research questions

The implementation is designed to investigate three questions:

1. Can restricted kernel search reach competitive Pin-FS solutions faster than
   a cold full CPLEX solve under the same wall-clock budget?
2. Does adaptive feature scoring improve anytime optimization performance over
   a fixed feature ranking?
3. Can a policy evolved on synthetic training/validation groups generalize to
   held-out real benchmarks without access to their outcomes during evolution?

The principal implementation contributions are:

- a paper-aligned Pin-FS-SVM mixed-integer formulation;
- one shared search engine for Static KS, ADKS, and VeraPin-KS;
- sparse-preserving, train-only preprocessing and corruption;
- safe policy evolution through a non-executable JSON DSL;
- common-budget anytime evaluation with persisted reference objectives;
- deterministic manifests, hashes, checkpoints, and result tables.

## 2. Pin-FS-SVM formulation

Let training observations be `(x_i, y_i)`, with `y_i in {-1, +1}`. The model
uses coefficients `w`, intercept `b`, absolute-value auxiliaries `z`, pinball
slacks `xi`, and binary selectors `v`. Given feature budget `B`, penalty `C`,
pinball parameter `tau`, and coefficient bounds `L < 0 < U`, the implemented
problem is:

```math
\begin{aligned}
\min_{w,b,z,\xi,v}\quad
    & \sum_{j=1}^{p} z_j + C\sum_{i=1}^{n}\xi_i \\
\text{s.t.}\quad
    & y_i(x_i^\top w+b)+\xi_i \ge 1, && i=1,\ldots,n, \\
    & y_i(x_i^\top w+b)-\xi_i/\tau \le 1, && i=1,\ldots,n, \\
    & -z_j \le w_j \le z_j, && j=1,\ldots,p, \\
    & L v_j \le w_j \le U v_j, && j=1,\ldots,p, \\
    & \sum_{j=1}^{p} v_j \le B, \\
    & z_j \ge 0,\; \xi_i \ge 0,\; v_j \in \{0,1\}.
\end{aligned}
```

Prediction uses `sign(x^T w + b)`, with an exact zero assigned to class `+1`.
The public implementation is in `src/models/pin_fs_svm.py`; reusable full and
restricted model construction is in `src/search/restricted_solver.py`.

## 3. Kernel-search methods

For a kernel `K` contained in the full feature set, the restricted model fixes
`v_j = 0` for every feature outside `K`. All kernel methods use the same engine,
solver interface, wall-clock accounting, incumbent handling, and optional final
full-model refinement.

### 3.1 Cold full Pin-FS

The reference route exposes every feature from the beginning and solves the full
mixed-integer problem directly. Its CPLEX progress trajectory provides incumbent
objectives, bounds, gaps, node counts, and time-to-target measurements.

### 3.2 Static Kernel Search

Static KS ranks features once using a configured train-only signal. It grows the
kernel through deterministic prefixes of that ranking and serves as a debugging
and fixed-ranking baseline.

### 3.3 Handcrafted ADKS

ADKS combines static and dynamic signals, including Fisher score, mutual
information, LP-relaxation activation, incumbent coefficient magnitude,
selection frequency, slack association, redundancy, inactivity, and kernel age.
After improvement it focuses the kernel; after sustained stagnation it expands
the kernel. Every weight and size parameter must be supplied explicitly.

### 3.4 VeraPin-KS

VeraPin policies define initial, add, keep, and target-size expressions. A policy
is stored as typed JSON and interpreted by a bounded DSL. Candidate policies
cannot execute Python, imports, files, network requests, subprocesses,
reflection, or loops. A frozen policy is deterministic and can be evaluated
without constructing an LLM provider.

Restricted feasible solutions can be converted into validated full-vector CPLEX
MIP starts. Gaps from different restricted feasible regions are treated as local
diagnostics; route-level comparable gaps require full-model refinement.

## 4. Policy evolution and fitness

Evolution and final evaluation are separated by research group:

```text
synthetic training group -> candidate evolution
synthetic validation group -> final policy selection and freezing
held-out benchmark group -> final comparison only
```

Before any provider is created, the pipeline computes a policy-independent
reference for each training/validation instance using cold full Pin-FS and fixed
ADKS under the same total solver budget as candidate policies. The best feasible
in-budget objective becomes the persisted reference objective.

Candidate fitness is a configured weighted combination of:

- normalized primal integral;
- final relative gap;
- failure rate;
- policy/search overhead.

Every route is scored over the same horizon, including the unused tail after an
early stop. Reference objectives, horizons, normalization, target gap, problem
parameters, and failure rules participate in cache and checkpoint identity.
Resume requires the original reference artifact.

## 5. Experimental protocol

The scientific final route implements the following protocol:

1. Use five outer stratified folds for held-out evaluation.
2. Use three inner stratified folds for parameter selection.
3. Fit preprocessing only on the applicable training partition.
4. Generate corruption only after splitting and only on training observations.
5. Select `B`, `C`, and `tau` by inner Balanced Accuracy using full/reduced
   Pin-FS only; ADKS and VeraPin do not participate in hyperparameter selection.
6. Break score ties by fewer active coefficients (`abs(w) > 1e-3`), then smaller
   `B`, then deterministic parameter order.
7. Give cold CPLEX, ADKS, and frozen VeraPin the same prepared outer-training
   input, untouched outer-test partition, and total wall-clock budget.
8. Evaluate predictions only on held-out observations.

Reported classification metrics are:

- Balanced Accuracy;
- weighted F1;
- Accuracy;
- geometric mean of sensitivity and specificity (G-mean).

## 6. Corruption protocol

Supported deterministic training-only conditions are:

- `label_noise`;
- `mixed` label, additive, and multiplicative noise;
- `feature_outlier`;
- `combined` mixed noise plus feature outliers.

Sparse feature corruption samples numerical nonzeros only and never fills
structural zeros. Additive and multiplicative masks are disjoint. Outlier size is
defined as `scale * training_std_j`, with population standard deviation measured
on the clean preprocessed training data. Constant features remain unchanged.
Every corruption produces a manifest containing parameters, selected cells,
effective changes, hashes, and replay information.

## 7. Benchmark data

The repository retains six binary-classification benchmarks and does not rewrite
their source files.

| Dataset | Solver-facing samples | Features | Storage | Partition policy | Preprocessing |
| --- | ---: | ---: | --- | --- | --- |
| Hill-Valley | 1,212 | 100 | dense | merge labeled train/test | standardization |
| Madelon | 2,600 | 500 | dense | merge labeled train/validation | standardization |
| GINA | 3,468 | 970 | dense | pooled upload | standardization |
| HIVA | 4,229 | 1,617 | dense | pooled upload | standardization |
| Colon Cancer | 62 | 2,000 | CSR | pooled upload | upstream-normalized passthrough |
| BASEHOCK | 1,993 | 4,862 | CSR | pooled upload | max-absolute scaling |

Native labels are mapped explicitly into a new `int64 {-1, +1}` vector. Stable
sample identifiers retain dataset, source partition, and original row index.
The registry is stored in `configs/benchmark_registry.yaml`; detailed provenance,
rights, dimensions, hashes, and original class counts are documented in
`dataset/README.md` and `dataset/manifest.json`.

## 8. Reproducible installation

### 8.1 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency contract accepts CPLEX Python API versions `>=22.1.1,<23` and
DOcplex versions `>=2.25,<3`.

### 8.2 Full CPLEX runtime

The intended benchmark models require a full-size IBM ILOG CPLEX installation
and license. The Community Edition size limit must not be bypassed through
unreported row or feature reduction. For a standard CPLEX Studio 22.1.2 Linux
installation, connect the Python environment to the full runtime with:

```bash
docplex config --upgrade /opt/ibm/ILOG/CPLEX_Studio2212
```

Verify the runtime and license capacity:

```bash
python -c 'import cplex; m=cplex.Cplex(); print(m.get_version()); m.end()'
python -c 'from src.experiments.readiness import cplex_environment_report; print(cplex_environment_report(probe_size_limit=True))'
```

A usable full-size environment reports `large_model_probe_passed: True`.

## 9. Data and software validation

Run both read-only data audits before an experiment:

```bash
python main.py validate-datasets
python main.py validate-benchmarks --registry configs/benchmark_registry.yaml
```

`validate-datasets` verifies the ten retained source files and eight native
partitions against the manifest. `validate-benchmarks` audits six solver-facing
views, label maps, storage formats, partition policies, and declared
preprocessing. Neither command fits a model or changes source data.

Run the software regression suite with:

```bash
PYTHONPYCACHEPREFIX=/tmp/pinfs-pycache python -m compileall -q main.py src
PYTHONPYCACHEPREFIX=/tmp/pinfs-pycache python -m pytest -q
```

## 10. Experiment workflow

The command-line interface exposes only the active research pipeline:

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
python main.py hardness --config configs/hardness_real_pilot.yaml --validate-only
python main.py hardness --config configs/hardness_synthetic_pilot.yaml --validate-only
python main.py adks --config configs/adks_real_pilot.yaml --validate-only
```

Running `--validate-only` against the supplied evolution or final configuration
is expected to exit nonzero and list unresolved author decisions until those
templates have been completed.

The real hardness pilot allows one explicitly selected instance per run:

```bash
python main.py hardness \
  --config configs/hardness_real_pilot.yaml \
  --instance hill-valley-clean
```

After pilot evidence and every author gate have been resolved, the intended
sequence is:

```bash
python main.py adks --config configs/adks_real_pilot.yaml
python main.py evolve-verapin --config configs/verapin_evolution.yaml
python main.py replay-evolution --run-dir results_verapin/<run_id>
python main.py evaluate-verapin \
  --config configs/verapin_final.yaml \
  --confirm-full-run
```

The supplied evolution and final configurations intentionally contain `null`
values and `false` readiness gates. These are unresolved scientific decisions,
not software defaults. They must be reviewed rather than filled automatically.

## 11. Outputs and audit trail

Runs are written below `results_verapin/` by default. Depending on the route,
the output includes:

- `manifest.json`: configuration identity, status, and run metadata;
- `solver_profiles.csv` and `hardness_summary.json`: cold-solver hardness data;
- `route_results.csv`, `iteration_results.csv`, and `search_details.json`:
  route-level and iteration-level kernel-search results;
- `fitness_references.json`: fixed evolution scoring anchors;
- `checkpoint.json`: resumable evolution state;
- `provider_records.json` and `offline_llm_summary.json`: replay and cost audit;
- `validation_selection.json`: validation-only freeze decision;
- `policies/frozen_verapin_policy.json`: immutable selected policy.

Generated results and policies are ignored under `results_verapin/` and
`artifacts_verapin/` so that scientific artifacts are reviewed before deliberate
publication.

## 12. Repository organization

```text
main.py                         command-line interface
configs/                        benchmark and experiment protocols
dataset/                        original uploads, provenance, and checksums
src/data/data_loader.py         raw file parsing and checksum verification
src/data/benchmark_data.py      registry, validation, solver-ready views
src/data/preprocessing.py       train-only dense/sparse transformations
src/data/synthetic.py           reproducible synthetic instances
src/data/corruptions.py         deterministic train-only perturbations
src/evaluation/                 held-out classification metrics
src/experiments/                preparation, selection, gates, orchestration
src/models/                     Pin-FS-SVM model and solver support
src/reporting/                  solver and kernel-search result schemas
src/search/                     restricted solver and shared search engine
src/search/policies/            Static KS, handcrafted ADKS, frozen VeraPin
src/search/llm_evolution/       safe DSL, evaluation, replay, and evolution
src/utils/                      configuration, matrices, serialization
tests/                          regression and scientific-invariant tests
```

## 13. Scope and limitations

- The repository currently documents methods and planned experiments, not final
  empirical findings.
- Solver performance depends on hardware, CPLEX version, license, and explicitly
  recorded solver parameters.
- The real benchmarks are binary classification datasets with heterogeneous
  sample sizes, dimensionality, sparsity, and class balance.
- The Colon Cancer upload is already normalized upstream and therefore cannot
  support a claim of raw-data train-only scaling.
- LLM-generated policies are heuristics. Safety of the DSL prevents arbitrary
  code execution but does not guarantee scientific quality or generalization.
- Any change to data inventory, preprocessing, folds, corruption, fitness
  references, solver budgets, or frozen policies defines a different experiment
  and must be reported as such.

## 14. Citation and provenance

No repository-level citation metadata or final paper citation has yet been
provided. Before publication, add the authors, paper title, venue or preprint,
year, DOI, and a versioned software archive identifier. Dataset-specific sources,
licenses, and links are maintained in `dataset/README.md`; dataset rights remain
separate from any license applied to the source code.
