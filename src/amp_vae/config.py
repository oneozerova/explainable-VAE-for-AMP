"""Project-wide constants."""

from __future__ import annotations

CONDITION_COLUMNS = [
    "is_anti_gram_positive",
    "is_anti_gram_negative",
    "is_antifungal",
    "is_antiviral",
    "is_antiparasitic",
    "is_anticancer",
]

GENERATION_CONDITION_COLUMNS = list(CONDITION_COLUMNS)

SPECIAL_TOKENS = ("<PAD>", "<SOS>", "<EOS>", "<UNK>")

VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

DEFAULT_CVAE_CHECKPOINT = "models/best_cvae.pt"
DEFAULT_CLASSIFIER_CHECKPOINT = "models/best_external_amp_classifier.pt"
DEFAULT_VOCAB_PATH = "models/vocab.pkl"
