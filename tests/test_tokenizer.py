from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_vae.config import SPECIAL_TOKENS, VALID_AMINO_ACIDS
from amp_vae.data.tokenizer import (
    build_vocab_bundle,
    detokenize_sequence,
    tokenize_sequence,
)


class TokenizerTest(unittest.TestCase):
    def test_roundtrip(self):
        vocab = build_vocab_bundle(["ACD"], condition_cols=["is_anti_gram_positive"], max_len=8)
        tokens, seq_len = tokenize_sequence("ACD", vocab)
        self.assertGreater(seq_len, 0)
        seq = detokenize_sequence(tokens, vocab)
        self.assertEqual(seq, "ACD")

    def test_vocab_is_fixed_alphabet_plus_specials(self):
        # Training data uses only 3 AAs but vocab must still cover all 20.
        vocab = build_vocab_bundle(["ACD"], max_len=8)
        self.assertEqual(
            len(vocab.vocab_list),
            len(SPECIAL_TOKENS) + len(VALID_AMINO_ACIDS),
        )
        self.assertEqual(len(vocab.vocab_list), 24)
        for token in SPECIAL_TOKENS:
            self.assertIn(token, vocab.char2idx)
        for aa in VALID_AMINO_ACIDS:
            self.assertIn(aa, vocab.char2idx)

    def test_build_rejects_non_aa_chars(self):
        with self.assertRaises(ValueError):
            build_vocab_bundle(["ACDX"], max_len=8)
        with self.assertRaises(ValueError):
            build_vocab_bundle(["ACDU"], max_len=8)


if __name__ == "__main__":
    unittest.main()
