from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_CLASSIFIER_CHECKPOINT, DEFAULT_VOCAB_PATH
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.saliency import (
    _tokenize_batch,
    integrated_gradients_per_residue,
    saliency_to_heatmap,
)
from amp_vae.models.classifier import load_external_classifier_checkpoint
from amp_vae.utils.paths import repo_path


DEFAULT_SEQUENCES = [
    "FLRRIRAWLRR",
    "RGGRLYFNGRFRRKRRRV",
    "FLPLLAGLLANFL",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Classifier saliency via integrated gradients.")
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--classifier", default=str(repo_path(DEFAULT_CLASSIFIER_CHECKPOINT)))
    parser.add_argument("--artifacts", default=str(repo_path("models", "external_amp_classifier_artifacts.json")))
    parser.add_argument("--sequences", default="", help="Comma-separated override list.")
    parser.add_argument("--sequences-file", default="", help="Optional CSV with a 'sequence' column.")
    parser.add_argument("--labels", default="is_anti_gram_positive,is_antifungal")
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--fig-out", default=str(repo_path("docs", "figures", "saliency_examples.png")))
    parser.add_argument("--table-out", default=str(repo_path("docs", "tables", "saliency_examples.csv")))
    return parser.parse_args()


def _load_sequences(args) -> list[str]:
    if args.sequences_file:
        frame = pd.read_csv(args.sequences_file)
        if "sequence" not in frame.columns:
            raise ValueError("sequences-file must contain a 'sequence' column")
        return [str(s).strip() for s in frame["sequence"].tolist() if str(s).strip()]
    if args.sequences:
        return [s.strip() for s in args.sequences.split(",") if s.strip()]
    return list(DEFAULT_SEQUENCES)


def _load_target_cols(path: str, num_labels: int) -> list[str]:
    with open(path, "r") as f:
        payload = json.load(f)
    # Prefer explicit v2 schema field, fall back to legacy key.
    raw = payload.get("target_labels", payload.get("target_cols", []))
    artifact_labels = [str(c) for c in raw]
    current = list(CONDITION_COLUMNS)
    if artifact_labels != current:
        raise RuntimeError(
            f"classifier schema drift: artifact={artifact_labels} vs CONDITION_COLUMNS={current}"
        )
    if len(artifact_labels) != num_labels:
        raise RuntimeError(
            f"classifier head dim ({num_labels}) does not match artifact labels ({len(artifact_labels)}); "
            f"retrain classifier with current CONDITION_COLUMNS."
        )
    return artifact_labels


def _infer_num_labels(path: str) -> int:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return int(sd["head.3.weight"].shape[0])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab_bundle(args.vocab)
    num_labels = _infer_num_labels(args.classifier)
    target_cols = _load_target_cols(args.artifacts, num_labels)
    model = load_external_classifier_checkpoint(
        args.classifier,
        vocab,
        num_labels=num_labels,
        device=device,
    )
    model.eval()

    label_names = [s.strip() for s in args.labels.split(",") if s.strip()]
    for name in label_names:
        if name not in target_cols:
            raise ValueError(f"label '{name}' not in target_cols={target_cols}")
    label_indices = [target_cols.index(name) for name in label_names]

    sequences = _load_sequences(args)
    tokens, lengths = _tokenize_batch(sequences, vocab)

    saliency_by_label: dict[str, torch.Tensor] = {}
    for name, idx in zip(label_names, label_indices):
        sal = integrated_gradients_per_residue(
            model,
            tokens=tokens,
            lengths=lengths,
            target_label_idx=idx,
            n_steps=args.n_steps,
            device=device,
        )
        saliency_by_label[name] = sal.cpu()

    rows = []
    flat_saliencies: list[np.ndarray] = []
    for seq_i, seq in enumerate(sequences):
        for name in label_names:
            sal_row = saliency_by_label[name][seq_i].numpy()
            offset = 1
            per_residue = sal_row[offset : offset + len(seq)]
            if per_residue.size < len(seq):
                pad = np.zeros(len(seq) - per_residue.size)
                per_residue = np.concatenate([per_residue, pad])
            flat_saliencies.append(per_residue)
            for pos, (aa, value) in enumerate(zip(seq, per_residue)):
                rows.append({
                    "sequence": seq,
                    "label": name,
                    "position": pos,
                    "residue": aa,
                    "saliency": float(value),
                })

    frame = pd.DataFrame(rows)
    Path(args.table_out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table_out, index=False)
    print(f"wrote {args.table_out}")

    saliency_to_heatmap(sequences, flat_saliencies, label_names, args.fig_out)
    print(f"wrote {args.fig_out}")

    print("per-label max saliency residue:")
    for name in label_names:
        for seq_i, seq in enumerate(sequences):
            sal_row = saliency_by_label[name][seq_i].numpy()[1 : 1 + len(seq)]
            if sal_row.size == 0:
                continue
            top = int(np.argmax(sal_row))
            print(f"  {name} | {seq}: pos={top} aa={seq[top]} sal={sal_row[top]:.4f}")


if __name__ == "__main__":
    main()
