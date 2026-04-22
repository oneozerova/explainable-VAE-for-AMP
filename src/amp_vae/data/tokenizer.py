"""Vocabulary helpers used by the generator and classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable, Sequence

from ..config import DEFAULT_MAX_LEN, SPECIAL_TOKENS, VALID_AMINO_ACIDS


@dataclass(frozen=True)
class VocabBundle:
    char2idx: dict[str, int]
    idx2char: dict[int, str]
    max_len: int
    vocab_list: list[str]
    condition_cols: list[str]

    @property
    def pad_idx(self) -> int:
        return self.char2idx["<PAD>"]

    @property
    def sos_idx(self) -> int:
        return self.char2idx["<SOS>"]

    @property
    def eos_idx(self) -> int:
        return self.char2idx["<EOS>"]

    @property
    def unk_idx(self) -> int:
        return self.char2idx["<UNK>"]

    @classmethod
    def from_dict(cls, payload: dict) -> "VocabBundle":
        return cls(
            char2idx=dict(payload["char2idx"]),
            idx2char={int(k): v for k, v in payload["idx2char"].items()},
            max_len=int(payload["max_len"]),
            vocab_list=list(payload.get("vocab_list", [])),
            condition_cols=list(payload.get("condition_cols", [])),
        )

    def to_dict(self) -> dict:
        return {
            "char2idx": self.char2idx,
            "idx2char": self.idx2char,
            "max_len": self.max_len,
            "vocab_list": self.vocab_list,
            "condition_cols": self.condition_cols,
        }


def load_vocab_bundle(path: str | Path) -> VocabBundle:
    with open(Path(path), "rb") as f:
        payload = pickle.load(f)
    return VocabBundle.from_dict(payload)


def dump_vocab_bundle(bundle: VocabBundle, path: str | Path) -> None:
    with open(Path(path), "wb") as f:
        pickle.dump(bundle.to_dict(), f)


def build_vocab_bundle(
    sequences: Sequence[str],
    condition_cols: Iterable[str] | None = None,
    max_len: int | None = None,
    alphabet: Iterable[str] = VALID_AMINO_ACIDS,
) -> VocabBundle:
    """Build vocab with a fixed alphabet (default: 20 canonical AAs) + specials.

    Asserts every char in every sequence belongs to ``alphabet``. Raises on
    violation so that cleaning bugs upstream surface here instead of silently
    producing ``<UNK>`` tokens.
    """
    alpha_set = frozenset(alphabet)
    for seq in sequences:
        s = str(seq).strip().upper()
        bad = set(s) - alpha_set
        if bad:
            raise ValueError(
                f"build_vocab_bundle: sequence contains chars outside alphabet: {sorted(bad)}"
            )
    sorted_alpha = sorted(alpha_set)
    vocab_list = list(SPECIAL_TOKENS) + sorted_alpha
    char2idx = {ch: i for i, ch in enumerate(vocab_list)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    max_len = int(max_len or DEFAULT_MAX_LEN)
    return VocabBundle(
        char2idx=char2idx,
        idx2char=idx2char,
        max_len=max_len,
        vocab_list=vocab_list,
        condition_cols=list(condition_cols or []),
    )


def tokenize_sequence(
    seq: str,
    vocab: VocabBundle,
    add_sos_eos: bool = True,
) -> tuple[list[int], int]:
    seq = str(seq).strip().upper()
    tokens = [vocab.char2idx.get(ch, vocab.unk_idx) for ch in seq]
    if add_sos_eos:
        tokens = [vocab.sos_idx] + tokens + [vocab.eos_idx]
    seq_len = min(len(tokens), vocab.max_len)
    tokens = tokens[: vocab.max_len]
    if len(tokens) < vocab.max_len:
        tokens = tokens + [vocab.pad_idx] * (vocab.max_len - len(tokens))
    return tokens, seq_len


def detokenize_sequence(tokens: Iterable[int], vocab: VocabBundle) -> str:
    chars: list[str] = []
    for idx in tokens:
        token = vocab.idx2char.get(int(idx), "<UNK>")
        if token == "<EOS>":
            break
        if token in {"<PAD>", "<SOS>"}:
            continue
        chars.append(token)
    return "".join(chars)

