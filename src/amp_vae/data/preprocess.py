"""Dataset cleaning and activity-label inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..config import CONDITION_COLUMNS, DEFAULT_MAX_LEN, VALID_AMINO_ACIDS
from .schema import ACTIVITY_COLUMN, LENGTH_COLUMN, SEQUENCE_COLUMN, TARGET_COLUMNS

GRAM_POS_RE = re.compile(r"\bgram[- ]?(?:positive|pos|\+)", re.IGNORECASE)
GRAM_NEG_RE = re.compile(r"\bgram[- ]?(?:negative|neg)|\bgram-(?![a-zA-Z+])", re.IGNORECASE)
ANTIFUNGAL_RE = re.compile(
    r"\b(antifungal|anti-fungal|antimycotic|fungicidal|fungicide|"
    r"anti\s+(fungal|mycotic|candida)|candidacidal|"
    r"fluconazole|amphotericin|echinocandin|azole|polyene)\b",
    re.IGNORECASE,
)
ANTIVIRAL_RE = re.compile(
    r"\b(antiviral|anti-viral|anti\s+viral|virucidal|"
    r"anti[- ]?(hiv|herpes|influenza)|anti\s+(coronavirus|influenza|herpes))\b",
    re.IGNORECASE,
)
ANTIPARASITIC_RE = re.compile(
    r"\b(antiparasitic|anti-parasitic|antiparasital|"
    r"anti\s+(parasitic|protozoal|helminth|worm|helminthi?cidal)|"
    r"anthelmintic|antihelminthic|vermifuge)\b",
    re.IGNORECASE,
)
ANTICANCER_RE = re.compile(
    r"\b(anticancer|anti-cancer|anti-tumor|antitumor|"
    r"anti-cancerous|antineoplastic|tumoricidal|anti-tumour|antitumour|"
    r"anti\s+tumou?r|anticarcinogenic|oncolytic)\b",
    re.IGNORECASE,
)

# Map filename substrings -> forced condition column
SOURCE_FILE_LABEL_MAP: dict[str, str] = {
    "anticancer": "is_anticancer",
    "antiparasitic": "is_antiparasitic",
    "viral": "is_antiviral",
    "antiviral": "is_antiviral",
    "antifungal": "is_antifungal",
}


def normalize_whitespace(text: object) -> str:
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_condition_flags(activity_text: object) -> dict[str, int]:
    text = normalize_whitespace(activity_text)
    flags = {name: 0 for name in TARGET_COLUMNS}
    flags["is_anti_gram_positive"] = int(bool(GRAM_POS_RE.search(text)))
    flags["is_anti_gram_negative"] = int(bool(GRAM_NEG_RE.search(text)))
    flags["is_antifungal"] = int(bool(ANTIFUNGAL_RE.search(text)))
    flags["is_antiviral"] = int(bool(ANTIVIRAL_RE.search(text)))
    flags["is_antiparasitic"] = int(bool(ANTIPARASITIC_RE.search(text)))
    flags["is_anticancer"] = int(bool(ANTICANCER_RE.search(text)))
    return flags


def infer_condition_frame(records: Sequence[Mapping[str, object]], activity_key: str = ACTIVITY_COLUMN) -> list[dict[str, int]]:
    return [infer_condition_flags(row.get(activity_key, "")) for row in records]


def ensure_binary_condition_columns(frame, columns: Iterable[str] = CONDITION_COLUMNS):
    for col in columns:
        if col in frame.columns:
            frame[col] = frame[col].fillna(0).astype(int)
    return frame


def source_file_label(path: str | Path) -> str | None:
    """Return condition column implied by source filename, or None."""
    name = Path(path).stem.lower()
    for key, col in SOURCE_FILE_LABEL_MAP.items():
        if key in name:
            return col
    return None


def clean_dataset(
    frame,
    drop_all_zero_labels: bool = True,
    deduplicate: bool = True,
    max_len: int = DEFAULT_MAX_LEN,
):
    if SEQUENCE_COLUMN not in frame.columns:
        raise KeyError(f"Missing required column: {SEQUENCE_COLUMN}")

    frame = frame.dropna(subset=[SEQUENCE_COLUMN]).copy()
    # 1. Uppercase + strip
    frame[SEQUENCE_COLUMN] = frame[SEQUENCE_COLUMN].astype(str).str.strip().str.upper()
    frame = frame[frame[SEQUENCE_COLUMN] != ""]

    # 2. Drop rows with non-standard AA chars
    valid = set(VALID_AMINO_ACIDS)
    before = len(frame)
    mask_aa = frame[SEQUENCE_COLUMN].map(lambda s: set(s).issubset(valid))
    frame = frame[mask_aa].copy()
    dropped_aa = before - len(frame)
    if dropped_aa:
        print(f"[clean_dataset] dropped {dropped_aa} rows with non-standard AA chars")

    # 3. Drop over-length rows (lose EOS if kept)
    limit = int(max_len) - 2
    before = len(frame)
    frame = frame[frame[SEQUENCE_COLUMN].str.len() <= limit].copy()
    dropped_len = before - len(frame)
    if dropped_len:
        print(f"[clean_dataset] dropped {dropped_len} rows longer than {limit} (max_len-2)")

    if LENGTH_COLUMN not in frame.columns:
        frame[LENGTH_COLUMN] = frame[SEQUENCE_COLUMN].str.len()
    else:
        # recompute length after uppercasing/stripping to stay consistent
        frame[LENGTH_COLUMN] = frame[SEQUENCE_COLUMN].str.len()

    ensure_binary_condition_columns(frame)

    # 4. Dedup on uppercase sequence
    if deduplicate:
        before = len(frame)
        agg = {col: "max" for col in TARGET_COLUMNS if col in frame.columns}
        for col in frame.columns:
            if col not in agg and col != SEQUENCE_COLUMN:
                agg[col] = "first"
        frame = frame.groupby(SEQUENCE_COLUMN, as_index=False).agg(agg)
        print(f"[clean_dataset] dedup on uppercase sequence: {before} -> {len(frame)}")

    if drop_all_zero_labels:
        present = [col for col in TARGET_COLUMNS if col in frame.columns]
        if present:
            frame = frame[frame[present].sum(axis=1) > 0].copy()

    return frame.reset_index(drop=True)


def read_csvs(paths: Iterable[str | Path]):
    import pandas as pd

    frames = []
    for path in paths:
        p = Path(path)
        sub = pd.read_csv(p)
        sub["_source_file"] = p.name
        frames.append(sub)
    if not frames:
        raise ValueError("No input files provided")
    return pd.concat(frames, ignore_index=True)


def build_labeled_dataset(
    source_paths: Iterable[str | Path],
    output_path: str | Path | None = None,
    drop_all_zero_labels: bool = True,
    deduplicate: bool = True,
    max_len: int = DEFAULT_MAX_LEN,
):
    import pandas as pd

    frame = read_csvs(source_paths)

    # Activity-regex inferred flags
    if ACTIVITY_COLUMN in frame.columns:
        labels = frame[ACTIVITY_COLUMN].fillna("").map(infer_condition_flags).apply(pd.Series)
        for col in TARGET_COLUMNS:
            frame[col] = labels[col].astype(int)
    else:
        for col in TARGET_COLUMNS:
            if col not in frame.columns:
                frame[col] = 0

    # OR-in source-file implied labels (rows from anticancer.csv etc. without Activity)
    if "_source_file" in frame.columns:
        implied = frame["_source_file"].map(lambda x: source_file_label(x) if isinstance(x, str) else None)
        for col in TARGET_COLUMNS:
            mask = implied == col
            if mask.any():
                frame.loc[mask, col] = (frame.loc[mask, col].fillna(0).astype(int) | 1).astype(int)
        frame = frame.drop(columns=["_source_file"])

    frame = clean_dataset(
        frame,
        drop_all_zero_labels=drop_all_zero_labels,
        deduplicate=deduplicate,
        max_len=max_len,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

    return frame
