# Original dataset uploads — no transformations

These files are copied byte-for-byte from the original uploads. Original names,
directory structure, formats, dtypes, feature values, labels, row order, and
train/test/validation files are preserved. Nothing is converted, normalized,
re-encoded, relabeled, merged, sampled, or filtered.

```text
dataset/
  BASEHOCK.mat
  colon-cancer.bz2
  gina.npz
  hiva.npz
  download.py
  hill+valley/       all original data, names, ARFF example, and illustration
  madelon/          all original matrices, labels, parameters, and reference PDF
  README.md         this repository note
  manifest.json     original sizes and SHA-256 checksums
```

All 19 uploaded data/script/documentation files are retained, including Madelon's
unlabeled test matrix. Only `.DS_Store` (operating-system metadata) is excluded
from Git; it remains in the original local backup. Both that backup and the
previous converted version are retained outside this repository.

The converted NPZ copies, curation script, and converted-data loader introduced
in commit `c460e39` have been removed from the current tree. The experiment code
and configs do not automatically load or transform these uploads.

## Integrity check only

`manifest.json` records the checksums measured **before** any conversion. Verify
that the original bytes remain unchanged without parsing or executing any file:

```bash
python -m pytest tests/test_dataset_originals.py -q
```

The NPZ exports from the uploaded `download.py` include object-typed labels.
They are intentionally left untouched; the integrity check does not unpickle
them. The download script is retained for provenance and is not executed by the
tests or pipeline. Future loading/preprocessing is a separate implementation
decision, not part of storing these originals.

## Source notes

Dataset rights are separate from repository code; these files are not relicensed.
Source metadata below were checked on 2026-08-31.

- **BASEHOCK**: [ASU/scikit-feature](https://jundongl.github.io/scikit-feature/datasets.html),
  derived from [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/). The uploaded
  MAT matches the public download. No dataset-specific license was stated on
  the checked collection page; no additional license is asserted here.
- **Colon**: Alon et al. (1999),
  [LIBSVM distribution](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#colon-cancer).
  The uploaded bzip2 matches the public download. Its distributor already applied
  normalization; preserving this export does not make it unprocessed source data.
  No dataset-specific license was stated on the checked page.
- **GINA**: Isabelle Guyon, Agnostic Learning vs. Prior Knowledge Challenge;
  [OpenML 1038](https://www.openml.org/d/1038), `gina_agnostic`, version 1.
- **HIVA**: IJCNN 2007 Workshop on Agnostic Learning vs. Prior Knowledge;
  [OpenML 1039](https://www.openml.org/d/1039), `hiva_agnostic`, version 1.
  OpenML metadata for both datasets lists `Public` (not an inferred SPDX or
  public-domain license). Both exports combine original training and validation
  samples; this repository does not invent split boundaries. The uploaded
  `download.py` records the OpenML IDs.
- **Hill-Valley**: Graham, L. & Oppacher, F. (2008).
  [UCI dataset](https://archive.ics.uci.edu/dataset/166/hill+valley),
  [DOI 10.24432/C5JC8P](https://doi.org/10.24432/C5JC8P),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All original files
  are retained without modification, including both clean and noisy variants.
- **Madelon**: Guyon, I. (2004).
  [UCI dataset](https://archive.ics.uci.edu/dataset/171/madelon),
  [DOI 10.24432/C5602H](https://doi.org/10.24432/C5602H),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All original files
  are retained without modification. The test matrix has no supplied labels;
  no labels are inferred or fabricated.
