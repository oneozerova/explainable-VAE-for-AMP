"""Dataset cleaning and activity-label inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..config import CONDITION_COLUMNS
from .schema import ACTIVITY_COLUMN, LENGTH_COLUMN, SEQUENCE_COLUMN, TARGET_COLUMNS

GRAM_POS_RE = re.compile(r"(gram\+|gram positive|gram[- ]?[+pP]|g[+pP])", re.IGNORECASE)
GRAM_NEG_RE = re.compile(r"(gram-|gram negative|gram[- ]?[nN]|g[-nN])", re.IGNORECASE)
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


def clean_dataset(frame, drop_all_zero_labels: bool = True, deduplicate: bool = True):
    if SEQUENCE_COLUMN not in frame.columns:
        raise KeyError(f"Missing required column: {SEQUENCE_COLUMN}")

    frame = frame.dropna(subset=[SEQUENCE_COLUMN]).copy()
    frame[SEQUENCE_COLUMN] = frame[SEQUENCE_COLUMN].astype(str).str.strip().str.upper()
    frame = frame[frame[SEQUENCE_COLUMN] != ""]

    if LENGTH_COLUMN not in frame.columns:
        frame[LENGTH_COLUMN] = frame[SEQUENCE_COLUMN].str.len()

    ensure_binary_condition_columns(frame)

    if deduplicate:
        agg = {col: "max" for col in TARGET_COLUMNS if col in frame.columns}
        for col in frame.columns:
            if col not in agg and col != SEQUENCE_COLUMN:
                agg[col] = "first"
        frame = frame.groupby(SEQUENCE_COLUMN, as_index=False).agg(agg)

    if drop_all_zero_labels:
        present = [col for col in TARGET_COLUMNS if col in frame.columns]
        if present:
            frame = frame[frame[present].sum(axis=1) > 0].copy()

    return frame.reset_index(drop=True)


def read_csvs(paths: Iterable[str | Path]):
    import pandas as pd

    frames = [pd.read_csv(Path(path)) for path in paths]
    if not frames:
        raise ValueError("No input files provided")
    return pd.concat(frames, ignore_index=True)


def build_labeled_dataset(
    source_paths: Iterable[str | Path],
    output_path: str | Path | None = None,
    drop_all_zero_labels: bool = True,
    deduplicate: bool = True,
):
    import pandas as pd

    frame = read_csvs(source_paths)
    if ACTIVITY_COLUMN in frame.columns:
        labels = frame[ACTIVITY_COLUMN].fillna("").map(infer_condition_flags).apply(pd.Series)
        for col in TARGET_COLUMNS:
            frame[col] = labels[col].astype(int)
    frame = clean_dataset(frame, drop_all_zero_labels=drop_all_zero_labels, deduplicate=deduplicate)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

    return frame

