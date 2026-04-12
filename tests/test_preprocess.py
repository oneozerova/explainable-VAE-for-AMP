from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_vae.data.preprocess import infer_condition_flags, normalize_whitespace


class PreprocessTest(unittest.TestCase):
    def test_normalize_whitespace(self):
        self.assertEqual(normalize_whitespace("A\u00a0B\n\n\nC"), "A B\n\nC")

    def test_infer_condition_flags(self):
        flags = infer_condition_flags("active against Gram-positive bacteria and antifungal targets")
        self.assertEqual(flags["is_anti_gram_positive"], 1)
        self.assertEqual(flags["is_antifungal"], 1)
        self.assertEqual(flags["is_antiviral"], 0)


if __name__ == "__main__":
    unittest.main()

