from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_vae.data.tokenizer import build_vocab_bundle, detokenize_sequence, tokenize_sequence


class TokenizerTest(unittest.TestCase):
    def test_roundtrip(self):
        vocab = build_vocab_bundle(["ACD"], condition_cols=["is_anti_gram_positive"], max_len=8)
        tokens, seq_len = tokenize_sequence("ACD", vocab)
        self.assertGreater(seq_len, 0)
        seq = detokenize_sequence(tokens, vocab)
        self.assertEqual(seq, "ACD")


if __name__ == "__main__":
    unittest.main()
