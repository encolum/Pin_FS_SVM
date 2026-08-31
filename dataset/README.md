# Six benchmarks — original values, minimal layout

Only filenames and locations have changed. The 10 retained input files keep their
original bytes, formats, feature values, labels, and row order. No scaling, dtype
conversion, feature selection, split merging, or noise generation is performed.

```text
dataset/
├── BASEHOCK.mat
├── colon-cancer.bz2
├── gina.npz
├── hiva.npz
├── hill_valley/
│   ├── train.data
│   └── test.data
├── madelon/
│   ├── train.data
│   ├── train.labels
│   ├── valid.data
│   └── valid.labels
├── README.md
└── manifest.json
```

Hill-Valley contains **only the original without-noise** training and testing
files. Madelon retains its labeled train and validation files separately; they
have **not** been merged for CV. The unlabeled Madelon test, upstream noisy
Hill-Valley variants, auxiliary documentation/examples, and completed download
script are removed from this tree. They remain recoverable in external backups
and the original-upload Git history. Future noise generation is a separate step.

## One manifest for provenance, integrity, and validation

`manifest.json` contains:

- `files`: current paths, original upload paths, unchanged sizes and SHA-256 hashes;
- `removed_files`: the nine user-requested exclusions, original hashes and reasons;
- `validation`: measured shape, dtype, original class counts, missing/infinite
  values, sparsity, and validation results for all eight retained partitions.

The old separate validation report is no longer kept in this directory.
`source_inventory_sha256` hashes the canonical manifest metadata **excluding**
the `validation` block, avoiding a circular self-hash.

```bash
# Read-only checks: no input or metadata writes, no solver.
python main.py validate-datasets
python -m pytest tests/test_dataset_originals.py tests/test_benchmark_loaders.py -q

# Explicitly refresh only manifest.json's validation block.
python main.py validate-datasets --update-manifest
```

## Six read-only loaders

`src/data/benchmark_loaders.py` exposes `load_basehock`, `load_colon`,
`load_gina`, `load_hiva`, `load_hill_valley`, and `load_madelon`, plus a
`load_benchmark_dataset` dispatcher. They return `RawBenchmarkDataset` objects
with `X`, `y`, source hashes and split identity.

```python
from src.data import load_basehock, load_gina, load_hill_valley, load_madelon

basehock = load_basehock()
gina = load_gina()
hill_train = load_hill_valley(partition="train")
hill_test = load_hill_valley(partition="test")
madelon_train = load_madelon(partition="train")
madelon_valid = load_madelon(partition="validation")  # reads valid.data/valid.labels
```

Hill-Valley rejects `variant="with_noise"`; its only variant is `without_noise`.
Madelon rejects `partition="test"` and never concatenates train/validation
automatically. BASEHOCK, Colon, GINA and HIVA expose their supplied files as pools,
without inventing original split indices.

Stored MAT/NPZ dtypes and native labels are retained. BASEHOCK's `Y` column is
exposed as a one-dimensional view; its original `(1993, 1)` shape is recorded.
Text files have no stored NumPy dtype; `parsed_float64` identifies their parser
dtype. Colon's LIBSVM matrix remains CSR. GINA/HIVA object/string labels are read
as literal strings from the known serialization format, without executing pickle
and without an unsafe fallback.

## Measured retained inputs

| Dataset / partition | X shape | X dtype | Original class counts | X sparsity |
| --- | --- | --- | --- | ---: |
| BASEHOCK / pool | 1993 × 4862 | uint8 | 1: 994; 2: 999 | 98.614515% |
| Colon / pool | 62 × 2000 | float64 (CSR) | -1: 40; +1: 22 | 0% |
| GINA / pool | 3468 × 970 | int64 | string -1: 1763; string 1: 1705 | 69.054418% |
| HIVA / pool | 4229 × 1617 | int64 | string -1: 4080; string 1: 149 | 90.918640% |
| Hill-Valley / train | 606 × 100 | float64 | 0: 305; 1: 301 | 0% |
| Hill-Valley / test | 606 × 100 | float64 | 0: 295; 1: 311 | 0% |
| Madelon / train | 2000 × 500 | float64 | -1: 1000; +1: 1000 | 0.000100% |
| Madelon / validation | 600 × 500 | float64 | -1: 300; +1: 300 | 0% |

All retained features and labels have zero measured missing/infinite values.
Sparsity is the fraction of feature entries exactly zero (including implicit
sparse zeros), not the fraction of unallocated storage. Original `y` dtypes are
uint8 for BASEHOCK, object/string for GINA/HIVA, and parsed float64 for the others.

Validation does not start hardness profiling. Native 1/2, 0/1 and string labels
still require an explicit reviewed adapter to the solver's `{-1, +1}` contract;
sparse/dense handling must also be explicit. The existing experiment configs and
author-confirmation gates are unchanged.

## Provenance and source rights

Dataset rights are separate from repository code. Source metadata were checked
on 2026-08-31; this cleanup does not relicense the files.

- **BASEHOCK**: [ASU/scikit-feature](https://jundongl.github.io/scikit-feature/datasets.html),
  derived from [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/). The original
  MAT matched the public download. The checked collection page stated no
  dataset-specific license; no additional license is asserted here.
- **Colon**: Alon et al. (1999),
  [LIBSVM distribution](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#colon-cancer).
  The bzip2 matched the public download. Upstream already normalized these values;
  preserving this export does not make it raw for a train-only-scaling claim.
  The checked page stated no dataset-specific license.
- **GINA**: Isabelle Guyon, Agnostic Learning vs. Prior Knowledge Challenge,
  [OpenML 1038](https://www.openml.org/d/1038), `gina_agnostic`, version 1.
- **HIVA**: IJCNN 2007 Workshop on Agnostic Learning vs. Prior Knowledge,
  [OpenML 1039](https://www.openml.org/d/1039), `hiva_agnostic`, version 1.
  The original download script used these IDs. Metadata lists `Public`, not an
  inferred SPDX/public-domain license. Both exports combine original training
  and validation samples; no boundaries are inferred here.
- **Hill-Valley**: Graham, L. & Oppacher, F. (2008),
  [UCI](https://archive.ics.uci.edu/dataset/166/hill+valley),
  [DOI 10.24432/C5JC8P](https://doi.org/10.24432/C5JC8P),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Only the without-noise train/test subset is retained; filenames are shortened,
  file contents unchanged. Original labels: 0 = valley, 1 = hill.
- **Madelon**: Guyon, I. (2004),
  [UCI](https://archive.ics.uci.edu/dataset/171/madelon),
  [DOI 10.24432/C5602H](https://doi.org/10.24432/C5602H),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Only labeled train/validation are retained; filenames are shortened, contents
  unchanged. The removed [technical report](https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/Dataset.pdf)
  remains available upstream and in the original backup.
