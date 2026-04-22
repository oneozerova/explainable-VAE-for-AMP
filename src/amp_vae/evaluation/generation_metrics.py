"""Basic generation metrics.

Denominator convention (uniqueness / novelty)
---------------------------------------------
Two conventions exist in the codebase history:

1. Old notebook (``notebooks/cvae_analysis.ipynb``, deleted):
   ``uniqueness = |unique(valid)| / |valid|``
   ``novelty    = |valid and not in reference| / |valid|``
   i.e. valid-only denominator.

2. Current code (this module):
   ``uniqueness = |unique(generated)| / |generated|``
   ``novelty    = |generated and not in reference| / |generated|``
   i.e. full denominator over all generated sequences (including invalids).

We keep the **full denominator** as the primary definition -- it is closer to
the product question "what fraction of what the model emitted is novel/unique"
and does not paper over a bad validity rate. For backward-compatibility with
the older CSV artefacts we additionally expose ``uniqueness_valid_only`` and
``novelty_valid_only`` which reproduce the old notebook numbers.

All reference and generated sequences are upper-cased and stripped before any
set membership check, so case-variant duplicates do not produce false novels.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def _normalize(seq: str) -> str:
    return str(seq).strip().upper()


def is_valid_sequence(seq: str, min_len: int = 5, max_len: int = 50, valid_aa: Iterable[str] = "ACDEFGHIKLMNPQRSTVWY") -> bool:
    valid_aa = set(valid_aa)
    s = _normalize(seq)
    return bool(s and min_len <= len(s) <= max_len and set(s).issubset(valid_aa))


def validity_rate(sequences: Sequence[str], min_len: int = 5, max_len: int = 50, valid_aa: Iterable[str] = "ACDEFGHIKLMNPQRSTVWY") -> float:
    valid_aa = set(valid_aa)
    if not sequences:
        return 0.0
    valid = 0
    for seq in sequences:
        s = _normalize(seq)
        if s and min_len <= len(s) <= max_len and set(s).issubset(valid_aa):
            valid += 1
    return valid / len(sequences)


def uniqueness_rate(sequences: Sequence[str]) -> float:
    """Full-denominator uniqueness: |unique(generated)| / |generated|."""
    if not sequences:
        return 0.0
    normed = [_normalize(s) for s in sequences]
    return len(set(normed)) / len(normed)


def uniqueness_valid_only(
    sequences: Sequence[str],
    min_len: int = 5,
    max_len: int = 50,
    valid_aa: Iterable[str] = "ACDEFGHIKLMNPQRSTVWY",
) -> float:
    """Valid-only uniqueness used by the old notebook: |unique(valid)| / |valid|."""
    valid_aa = set(valid_aa)
    valids = [
        _normalize(s)
        for s in sequences
        if _normalize(s) and min_len <= len(_normalize(s)) <= max_len and set(_normalize(s)).issubset(valid_aa)
    ]
    if not valids:
        return 0.0
    return len(set(valids)) / len(valids)


def novelty_rate(sequences: Sequence[str], reference_sequences: Sequence[str]) -> float:
    """Full-denominator novelty over all generated sequences.

    Reference and generated are both upper-cased and stripped before lookup so
    that mixed-case reference entries do not produce false positives.
    """
    if not sequences:
        return 0.0
    reference = {_normalize(s) for s in reference_sequences}
    novel = sum(1 for seq in sequences if _normalize(seq) not in reference)
    return novel / len(sequences)


def novelty_valid_only(
    sequences: Sequence[str],
    reference_sequences: Sequence[str],
    min_len: int = 5,
    max_len: int = 50,
    valid_aa: Iterable[str] = "ACDEFGHIKLMNPQRSTVWY",
) -> float:
    """Valid-only novelty used by the old notebook."""
    valid_aa = set(valid_aa)
    reference = {_normalize(s) for s in reference_sequences}
    valids = [
        _normalize(s)
        for s in sequences
        if _normalize(s) and min_len <= len(_normalize(s)) <= max_len and set(_normalize(s)).issubset(valid_aa)
    ]
    if not valids:
        return 0.0
    novel = sum(1 for s in valids if s not in reference)
    return novel / len(valids)
