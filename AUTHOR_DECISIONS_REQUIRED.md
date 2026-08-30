# Author decisions required before scientific runs

The corrected code must not guess the following manuscript-level choices.

## Coefficient bounds

The manuscript requires valid bounds `l_j < 0 < u_j` and says that the
experimental values are reported, but it does not give numeric values. The pilot
configuration preserves the legacy `[-2, 2]` assumption only for software QA.
Confirm or replace these bounds before treating any run as a manuscript result.

## Perturbation generation

Confirm every value used by the generated-data protocol:

- label-flip rate;
- additive-noise cell rate and standard deviation;
- multiplicative-noise cell rate and distribution/standard deviation;
- high-margin label-flip rate;
- reference L1-SVM `C`;
- number of independent corruption seeds.

The manuscript reports one archived processed file per dataset and condition. A
new generated-data study is a corrected extension, not an exact numerical
reproduction of those archived variants.

## Solver parity

The manuscript reports DOcplex with IBM ILOG CPLEX 22.1.1 and one CPLEX thread
for Pin-SVM, Pin-FS-SVM, and Budgeted MILP-SVM. Confirm the licensed CPLEX
runtime and whether strict 22.1.1 parity is required before the scientific pilot.

## Statistical reporting

The Wilcoxon/BH implementation is exploratory. Confirm the final pairing unit,
correction family, alternative hypothesis, and whether inferential results will
appear in the manuscript.

## Execution gate

Do not mark `coefficient_bounds.author_confirmed` as true, fill the full
corruption profiles, or launch `main.py run --confirm-full-run` until the choices
above have been reviewed by an author.

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
