# Curated benchmark inputs

Ten labeled partitions from six uploaded benchmark groups. Each `.npz` contains
only numeric `X` and `y`, readable with `numpy.load(..., allow_pickle=False)`.
Feature values and row/column order are unchanged; there is no new scaling,
feature selection, subsampling, or split merging. Integer storage is reduced
losslessly. Labels are explicitly mapped to `{-1, +1}`.

| Loader name | Partition(s) and rows | Features | Source labels → stored labels |
| --- | --- | ---: | --- |
| `basehock` | `pool`: 1,993 | 4,862 | 1 → -1; 2 → +1 |
| `colon_libsvm` | `pool`: 62 | 2,000 | -1/+1 unchanged |
| `gina` | `pool`: 3,468 | 970 | string -1/+1 → integer -1/+1 |
| `hiva` | `pool`: 4,229 | 1,617 | string -1/+1 → integer -1/+1 |
| `hill_valley_clean` | `train`: 606; `test`: 606 | 100 | 0 (valley) → -1; 1 (hill) → +1 |
| `hill_valley_noisy` | `train`: 606; `test`: 606 | 100 | 0 (valley) → -1; 1 (hill) → +1 |
| `madelon` | `train`: 2,000; `validation`: 600 | 500 | -1/+1 unchanged |

`manifest.json` is the complete kept/excluded inventory: original relative paths,
sizes and SHA-256 checksums, curated destinations, transformations, class counts,
and hashes of the committed inputs. The original source bundles are backed up
outside this repository, not destroyed or committed alongside duplicate copies.

## Loading and validation

```python
from src.data import load_benchmark_dataset

X, y = load_benchmark_dataset("gina", partition="pool")
X_train, y_train = load_benchmark_dataset("hill_valley_noisy", partition="train")
X_test, y_test = load_benchmark_dataset("hill_valley_noisy", partition="test")
```

The loader checks file hashes, shape, label counts, dtypes, and finite values.
It returns float64 features and integer labels without preprocessing. The
partition argument is mandatory; no loader automatically combines train/test.

```bash
python main.py validate-datasets
python -m pytest -q
```

These commands do not launch the experiment suite. The original manuscript
loader and configs remain separate; this collection does not restore its other
five datasets or its archived corruption variants.

## Evaluation cautions

- GINA/HIVA OpenML version 1 already concatenate original training and validation
  observations. The uploads contain no split indices, so they are kept as pools;
  do not invent split boundaries from the published sample counts. Create and
  record an appropriate held-out split before training or policy selection.
- Keep official Hill-Valley test partitions untouched during tuning. Clean and
  noisy versions are separate benchmarks, not redundant files or generated
  corruption variants. Do not treat their similarly positioned rows as pairs.
- Madelon's `validation` is not an independent final test if used for selection.
  The uploaded 1,800-row test matrix has no labels and is retained only in the
  original backup, not in this supervised subset. It can be restored for future
  challenge submissions; labels are never fabricated.
- `colon_libsvm` was normalized by the upstream distributor, including feature-wise
  normalization across the dataset. This cannot be undone by train-only scaling.
  Obtain raw data for a strict leakage-free preprocessing claim. It is intentionally
  not an alias for the manuscript loader's `colon.csv`.
- Fit any new scaling or feature selection on training data only. Solver budgets,
  policy splits and author-gated scientific configs are not chosen by this curation.

## Provenance and dataset rights

The following source metadata were checked on 2026-08-31. Dataset rights are
separate from repository code; this repository does not relicense third-party data.
The BASEHOCK/Colon source files and the Hill-Valley/Madelon numeric and metadata
files were compared byte-for-byte with the public downloads. Every GINA/HIVA
feature value, label, and row position was also compared with the named OpenML
exports after conversion; all comparisons matched.

- **BASEHOCK**: [ASU/scikit-feature collection](https://jundongl.github.io/scikit-feature/datasets.html),
  derived from [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/).
  The uploaded MAT is byte-identical to the collection's
  [public download](https://jundongl.github.io/scikit-feature/files/datasets/BASEHOCK.mat).
  The checked dataset page does not state a dataset-specific license; no additional
  license or unrestricted-reuse claim is asserted here.
- **Colon**: Alon et al. (1999), distributed by
  [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#colon-cancer).
  The uploaded bzip2 is byte-identical to its public download. The source documents
  instance-wise followed by feature-wise normalization. The checked dataset page
  does not state a dataset-specific license; the LIBSVM software license must not
  be assumed to license the data.
- **GINA**: Isabelle Guyon, Agnostic Learning vs. Prior Knowledge Challenge;
  [OpenML 1038, gina_agnostic, version 1](https://www.openml.org/d/1038).
  [OpenML metadata](https://www.openml.org/api/v1/json/data/1038) lists licence
  `Public` (verbatim metadata, not an inferred SPDX/public-domain license).
- **HIVA**: IJCNN 2007 Workshop on Agnostic Learning vs. Prior Knowledge;
  [OpenML 1039, hiva_agnostic, version 1](https://www.openml.org/d/1039).
  [OpenML metadata](https://www.openml.org/api/v1/json/data/1039) also lists `Public`.
  These are molecular descriptors and activity labels, not patient records.
  The uploaded `download.py` identifies the two OpenML IDs. Both ARFF exports
  credit TunedIT conversion. This curation changes storage and label dtype only.
- **Hill-Valley**: Graham, L. & Oppacher, F. (2008).
  [UCI dataset](https://archive.ics.uci.edu/dataset/166/hill+valley),
  [DOI: 10.24432/C5JC8P](https://doi.org/10.24432/C5JC8P),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  The original `hill_valley.names` is retained. Changes: CSV to numeric NPZ,
  class 0 mapped to -1; no feature or partition changes.
- **Madelon**: Guyon, I. (2004).
  [UCI dataset](https://archive.ics.uci.edu/dataset/171/madelon),
  [DOI: 10.24432/C5602H](https://doi.org/10.24432/C5602H),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  The original `madelon.param` is retained. Changes: labeled train/validation
  matrices and labels packed into NPZ; unlabeled test omitted from this subset.
  The [original technical report](https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/Dataset.pdf)
  remains available upstream and in the source backup.

## Kept versus excluded files

- Kept as numeric inputs: BASEHOCK X/Y; Colon features/labels; GINA/HIVA X/y;
  all four Hill-Valley data files; Madelon train and validation data **and labels**.
- Kept as metadata: Hill-Valley names and Madelon parameters; this guide and the
  machine-readable inventory preserve provenance and transformation details.
- Excluded from GitHub, retained in the original backup: Madelon unlabeled test,
  reference PDF, Hill-Valley example ARFF and illustration, original raw duplicate
  files, and the superseded root download script. `.DS_Store` is also omitted.

Rebuild into a **new** directory from the preserved upload bundle:

```bash
python scripts/curate_datasets.py --source-root /path/to/original/uploads \
  --output /path/to/new/curated-inputs
```

The script refuses to overwrite existing output, does not execute the uploaded
script, and does not unpickle object labels. It extracts only literal binary
labels from the known export format, verifies exact numeric round trips, and
rejects unsupported formats. This guide is maintained separately from generated
data and the manifest.
