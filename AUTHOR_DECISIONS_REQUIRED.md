# Author decisions required before scientific runs

The corrected code must not guess the following manuscript-level choices.

## Coefficient bounds

The formulation requires valid bounds `l_j < 0 < u_j`, but the manuscript does
not give numeric values. Confirm the bounds used by each active VeraPin config
before treating any run as a paper result.

## Perturbation generation

Confirm every value used by the generated-data protocol:

- label-flip rate;
- additive-noise cell rate and standard deviation;
- multiplicative-noise cell rate and distribution/standard deviation;
- high-margin label-flip rate;
- reference L1-SVM `C`;
- number of independent corruption seeds.

A generated-data study is a corrected extension, not an exact numerical
reproduction of archived manuscript variants.

## Solver parity

The manuscript reports DOcplex with IBM ILOG CPLEX 22.1.1 and one CPLEX thread
for Pin-SVM, Pin-FS-SVM, and Budgeted MILP-SVM. Confirm the licensed CPLEX
runtime and whether strict 22.1.1 parity is required before the scientific pilot.

## Execution gate

Do not mark `coefficient_bounds.author_confirmed` as true, fill the full
corruption profiles, or launch hardness/ADKS/evolution/final experiments until
the choices above have been reviewed by an author.

## VeraPin-KS scientific configuration

The full VeraPin infrastructure is implemented, but the following values are
intentionally `null` in `configs/hardness.yaml`, `configs/static_ks_pilot.yaml`,
`configs/adks_pilot.yaml`, `configs/verapin_evolution.yaml`, and
`configs/verapin_final.yaml` until an author records a decision:

- synthetic sample counts, informative and redundant ratios, correlation,
  imbalance, label noise, outlier severity,
  and feature-budget ratios;
- Pin-FS `B` derivation, `C`, `tau`, and coefficient bounds for these studies;
- CPLEX total/subproblem limits and target MIP gap;
- Static-KS kernel/bucket sizes and every handcrafted ADKS weight/adaptation value;
- evolution population size, generations, parent/candidate counts, diversity
  threshold, fitness weights/scales, and target gap;
- outer classification fold count and deterministic split seed for final evaluation;
- real LLM provider/model, temperature, and offline token/API cost budget.

The CLI validates these gates before creating a run directory. Held-out evaluation
also requires `--confirm-full-run` and loads only a frozen JSON policy; it never
constructs an LLM provider.

## Full benchmark integration: implementation versus research approval

The six-benchmark, sparse-safe, post-split-corruption and nested-classification
code is now implemented. This does not approve scientific hyperparameters or
justify ADKS/VeraPin's empirical usefulness.

The new `hardness_real_pilot.yaml`, `hardness_synthetic_pilot.yaml` and
`adks_real_pilot.yaml` contain no unresolved structural parameters. All their
numbers are explicitly **provisional software-QA choices**, including bounds,
budgets, time limits, seeds, signal switches and ADKS weights. They use
`execution.purpose: provisional_pilot`, `parameters_provisional: true`, and
`coefficient_bounds.author_confirmed: false`. Do not cite their outcomes as
paper results without reviewing these settings.

Confirmed local limitation on 2026-08-31: CPLEX 22.1.2 Community Edition rejected
a 1001-variable license probe with error 1016. A 40 × 20 clean synthetic CPLEX
pilot succeeded; full-size Hill-Valley/Madelon pilots were not run. A full-size
license and a decision on 22.1.1 versus 22.1.2 parity remain necessary. Inputs
must not be reduced silently to work around this restriction.

Before scientific execution, review:

- numerical bounds and whether they remain valid after each preprocessing policy;
- real-data feature budgets, `C`, `tau`, fixed route budgets and target gaps;
- nested `B/C/tau` grids, inner tuning solver limits and explicit inner/outer seeds;
- merged-pool versus official-holdout sensitivity protocols;
- Colon's upstream normalization and any explicit preprocessing override;
- sparse memory caps, signal costs and corruption memory caps;
- every experimental corruption rate/severity and repeated seed;
- two or more genuinely hard instances before approving the ADKS pilot;
- a frozen ADKS baseline, independent research groups, fixed budgets and affordable
  signals before approving evolution;
- a reviewed frozen method/policy before final 5 × 3 classification evaluation.

Real benchmarks are excluded from evolution by default. Any explicit override
is recorded, and overlapping source observations across research groups are
rejected. Active configs must use `kind: benchmark` for the six retained datasets
or `kind: synthetic`; legacy manuscript kinds are available only from the archive.
