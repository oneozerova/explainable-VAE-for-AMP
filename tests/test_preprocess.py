from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_vae.data.preprocess import (
    clean_dataset,
    infer_condition_flags,
    normalize_whitespace,
)
from amp_vae.data.schema import SEQUENCE_COLUMN


class PreprocessTest(unittest.TestCase):
    def test_normalize_whitespace(self):
        self.assertEqual(normalize_whitespace("A B\n\n\nC"), "A B\n\nC")

    def test_infer_condition_flags(self):
        flags = infer_condition_flags("active against Gram-positive bacteria and antifungal targets")
        self.assertEqual(flags["is_anti_gram_positive"], 1)
        self.assertEqual(flags["is_antifungal"], 1)
        self.assertEqual(flags["is_antiviral"], 0)

    def test_gram_regex_does_not_leak(self):
        flags = infer_condition_flags("diagram positive shapes in some unrelated study")
        # \bgram[- ]?(positive|...) requires word boundary at g.
        self.assertEqual(flags["is_anti_gram_positive"], 0)

    def test_dedup_on_uppercase(self):
        frame = pd.DataFrame(
            {
                SEQUENCE_COLUMN: ["KWKSFLKT", "kwksflkt", "LLRRR"],
                "is_anti_gram_positive": [1, 0, 1],
                "is_anti_gram_negative": [0, 1, 0],
                "is_antifungal": [0, 0, 0],
                "is_antiviral": [0, 0, 0],
                "is_antiparasitic": [0, 0, 0],
                "is_anticancer": [0, 0, 0],
            }
        )
        cleaned = clean_dataset(frame, max_len=64)
        self.assertEqual(len(cleaned), 2)
        merged = cleaned[cleaned[SEQUENCE_COLUMN] == "KWKSFLKT"].iloc[0]
        # dedup uses max over condition cols -> both labels OR'd
        self.assertEqual(int(merged["is_anti_gram_positive"]), 1)
        self.assertEqual(int(merged["is_anti_gram_negative"]), 1)

    def test_non_standard_aa_filtered(self):
        frame = pd.DataFrame(
            {
                SEQUENCE_COLUMN: ["KWKSFLKT", "KWXFLKT", "KWUFLKT", "KWaFLKT"],
                "is_anti_gram_positive": [1, 1, 1, 1],
                "is_anti_gram_negative": [0, 0, 0, 0],
                "is_antifungal": [0, 0, 0, 0],
                "is_antiviral": [0, 0, 0, 0],
                "is_antiparasitic": [0, 0, 0, 0],
                "is_anticancer": [0, 0, 0, 0],
            }
        )
        cleaned = clean_dataset(frame, max_len=64)
        # 'a' is uppercased first then passes (A valid) -> keeps
        # X, U are non-canonical -> dropped
        seqs = set(cleaned[SEQUENCE_COLUMN].tolist())
        self.assertIn("KWKSFLKT", seqs)
        self.assertIn("KWAFLKT", seqs)
        self.assertNotIn("KWXFLKT", seqs)
        self.assertNotIn("KWUFLKT", seqs)

    def test_overlength_dropped(self):
        short = "A" * 10
        exact = "A" * 62  # max_len=64 -> limit 62 kept
        over = "A" * 63  # dropped
        frame = pd.DataFrame(
            {
                SEQUENCE_COLUMN: [short, exact, over],
                "is_anti_gram_positive": [1, 1, 1],
                "is_anti_gram_negative": [0, 0, 0],
                "is_antifungal": [0, 0, 0],
                "is_antiviral": [0, 0, 0],
                "is_antiparasitic": [0, 0, 0],
                "is_anticancer": [0, 0, 0],
            }
        )
        cleaned = clean_dataset(frame, max_len=64)
        lengths = set(cleaned[SEQUENCE_COLUMN].str.len().tolist())
        self.assertIn(10, lengths)
        self.assertIn(62, lengths)
        self.assertNotIn(63, lengths)


if __name__ == "__main__":
    unittest.main()
