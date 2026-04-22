# Makefile for explainable-VAE-for-AMP
# Stage order: data -> split -> train-cvae / train-classifier -> generate -> eval
# Oracle submission is a separate target because it depends on an external API.
#
# Run the whole pipeline end-to-end:
#     make all
# Submit generations to AIPAMPDS:
#     make oracle
# Override the seed:
#     make all SEED=123

SEED ?= 42
PYTHON ?= python3
DATE := $(shell date +%Y%m%d)

.PHONY: all data split train-cvae train-classifier generate eval oracle clean

all: data split train-cvae train-classifier generate eval

data:
	$(PYTHON) scripts/build_dataset.py

split: data
	$(PYTHON) -c "import pandas as pd; from amp_vae.utils.splits import load_or_create_splits; from amp_vae.config import CONDITION_COLUMNS; df = pd.read_csv('data/processed/master_dataset.csv'); load_or_create_splits(df, seed=$(SEED), stratify_cols=CONDITION_COLUMNS, force=True)"

train-cvae: split
	$(PYTHON) scripts/train_cvae.py --seed $(SEED)

train-classifier: split
	$(PYTHON) scripts/train_classifier.py --seed $(SEED)

generate: train-cvae
	$(PYTHON) scripts/generate_candidates.py --checkpoint models/best_cvae.pt --seed $(SEED) --output docs/tables/generated_examples.csv

eval: generate train-classifier
	$(PYTHON) scripts/evaluate_generated.py --generated docs/tables/generated_examples.csv --reference data/processed/master_dataset.csv --output docs/tables/generation_metrics.json
	$(PYTHON) scripts/concept_vectors.py --walk-seed $(SEED)
	$(PYTHON) scripts/latent_analysis.py
	$(PYTHON) scripts/projection_plots.py --pair-seed $(SEED)
	$(PYTHON) scripts/interpolate.py
	$(PYTHON) scripts/saliency_examples.py

oracle: train-cvae
	$(PYTHON) scripts/generate_for_oracle.py --ckpt models/best_cvae.pt --seed $(SEED) --n-per-condition 250 --out data/external/aipampds/submission_$(DATE).csv
	$(PYTHON) scripts/submit_aipampds.py --in data/external/aipampds/submission_$(DATE).csv --out data/external/aipampds/scored_$(DATE).csv

clean:
	rm -f data/processed/split_train.npy data/processed/split_val.npy data/processed/split_test.npy
	rm -f models/best_cvae.pt models/external_amp_classifier_artifacts.json
