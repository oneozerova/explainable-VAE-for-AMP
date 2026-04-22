# Reproducibility Guide

One-page guide to regenerating every artefact in this repository from raw data,
and to submitting the resulting peptides to the AIPAMPDS oracle.

## 1. Environment

- Python 3.10 or 3.11.
- Install dependencies: `pip install -r requirements.txt` (add
  `iterative-stratification` for true multi-label stratified splits; otherwise
  `src/amp_vae/utils/splits.py` falls back to a deterministic groupby-based
  split).
- Deterministic cuDNN is enabled inside the training scripts (set in
  `amp_vae.utils.seed.set_seed`).
- Fixed seed is `42` for every stage. Override with `make all SEED=123`.

## 2. One-shot run

```bash
make all
```

rebuilds everything in the correct order:

```
data -> split -> train-cvae + train-classifier -> generate -> eval
```

To submit generations to the AIPAMPDS oracle after training:

```bash
make oracle
```

## 3. Stage-by-stage

| Target            | Script                                    | Approx runtime (CPU / single GPU) |
|-------------------|-------------------------------------------|-----------------------------------|
| `data`            | `scripts/build_dataset.py`                | < 1 min                           |
| `split`           | `amp_vae.utils.splits.load_or_create_splits(..., force=True)` | seconds                |
| `train-cvae`      | `scripts/train_cvae.py --seed $SEED`      | 20-40 min GPU / hours CPU         |
| `train-classifier`| `scripts/train_classifier.py --seed $SEED`| 10-20 min GPU                     |
| `generate`        | `scripts/generate_candidates.py`          | 1-2 min                           |
| `eval`            | `evaluate_generated.py`, `concept_vectors.py`, `latent_analysis.py`, `projection_plots.py`, `interpolate.py`, `saliency_examples.py` | 5-15 min total |
| `oracle`          | `scripts/generate_for_oracle.py` + `scripts/submit_aipampds.py` | 5 min generation + oracle-dependent |
| `clean`           | removes split files, cVAE checkpoint, classifier artefacts | instant            |

## 4. Split determinism

Train / validation / test partitions are built once by
`src/amp_vae/utils/splits.load_or_create_splits` and persisted as positional
index arrays:

```
data/processed/split_train.npy
data/processed/split_val.npy
data/processed/split_test.npy
```

Both `scripts/train_cvae.py` and `scripts/train_classifier.py` load these
indices (via `load_splits`) so the classifier never sees cVAE training rows at
evaluation time. To regenerate the splits (e.g. after changing the seed or the
dataset), run `make split` or pass `--force` through the helper.

## 5. Schema

The active condition vector is 6-dimensional (`src/amp_vae/config.py`
`CONDITION_COLUMNS`):

1. `is_anti_gram_positive`
2. `is_anti_gram_negative`
3. `is_antifungal`
4. `is_antiviral`
5. `is_antiparasitic`
6. `is_anticancer`

There is no `is_antibacterial` label - it was dropped because its near-constant
positive signal diluted gradients for rarer labels. Historical 7-label tables
under `docs/tables/` and `data/external/aipampds/` are kept for comparison
only; they do not match the current checkpoint.

## 6. AIPAMPDS hit-rate formula

`scripts/submit_aipampds.py` writes a per-condition metrics CSV using one
uniform rule, unlike the historical mix of criteria flagged in
`docs/pipeline_issues.md` #19:

| Condition           | Hit rule                           |
|---------------------|------------------------------------|
| `gram_pos`          | `saureus_score > 0.5`              |
| `gram_neg`          | `ecoli_score   > 0.5`              |
| `antifungal`        | `max(ecoli, saureus) > 0.5`        |
| `antiviral`         | `max(ecoli, saureus) > 0.5`        |
| `antiparasitic`     | `max(ecoli, saureus) > 0.5`        |
| `anticancer`        | `max(ecoli, saureus) > 0.5`        |
| `unconditional`     | `NA` (no target label to hit)      |

Additional derived metrics:

- `non_haemolytic_rate = mean(haemolytic_score < 0.5)`
- `safe_hit_rate       = mean(hit AND non_haem)`

The CSV carries a `#` comment header restating this formula inline so the file
is self-documenting. `mean_offtarget_score` is not defined under this rule and
is therefore not emitted (the historical column was computed inconsistently).

Rationale for the `max(...)` rule on non-bacterial conditions: AIPAMPDS only
exposes two bactericidal scoring endpoints, so this row acts as a generic
bactericidal proxy for any non-Gram condition. It is *not* a claim that the
peptide has the requested activity - it is a claim that the oracle would mark
it active against at least one tested bacterium.

## 7. Checkpoint integrity

Every artefact records:

- `checkpoint_sha256`: `hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()` -
  written by `scripts/generate_for_oracle.py` per row and in the companion
  `submission_YYYYMMDD_summary.json`.
- `git_sha`: `git rev-parse HEAD` at the time of generation.

This lets any downstream consumer verify they are reading the CSV produced by
a known checkpoint + source tree, and catches the class of drift described in
`docs/pipeline_issues.md` P0 #1 (7-label AIPAMPDS artefacts against a 6-label
model).

## 8. Known limitations

- Tiny dataset (~6k peptides after cleaning) - per-class support for rare
  labels (antiparasitic, antiviral, anticancer) is under 500 rows each.
- Single trained model per stage, no cross-validation or ensembling.
- AIPAMPDS is an external oracle with its own model-of-models biases;
  `safe_hit_rate` is a proxy for real wet-lab success, not a substitute.
- See the "Limits" section of `docs/blogpost.md` for a fuller discussion.
