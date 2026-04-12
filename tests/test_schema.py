from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_vae.config import CONDITION_COLUMNS, GENERATION_CONDITION_COLUMNS, SPECIAL_TOKENS, VALID_AMINO_ACIDS
from amp_vae.data.schema import TARGET_COLUMNS


class SchemaTest(unittest.TestCase):
    def test_condition_columns_are_canonical(self):
        self.assertEqual(TARGET_COLUMNS, CONDITION_COLUMNS)
        self.assertEqual(
            TARGET_COLUMNS,
            [
                "is_anti_gram_positive",
                "is_anti_gram_negative",
                "is_antifungal",
                "is_antiviral",
                "is_antiparasitic",
                "is_anticancer",
            ],
        )

    def test_special_tokens(self):
        self.assertEqual(SPECIAL_TOKENS, ("<PAD>", "<SOS>", "<EOS>", "<UNK>"))
        self.assertIn("A", VALID_AMINO_ACIDS)

    def test_generation_columns_match_condition_columns(self):
        self.assertEqual(GENERATION_CONDITION_COLUMNS, CONDITION_COLUMNS)
        self.assertEqual(len(GENERATION_CONDITION_COLUMNS), 6)


if __name__ == "__main__":
    unittest.main()
