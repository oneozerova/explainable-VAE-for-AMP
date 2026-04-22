"""Linear probe on cVAE mu per condition label.

Trains one logistic regression per label on train-split mu and reports
AUROC / AUPRC on the held-out test split. If AUROC < ~0.7 for a label,
no 2D projection will separate that label because the latent doesn't
encode it linearly.

Usage::

    python3 scripts/latent_probe.py
"""

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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from amp_vae.config import DEFAULT_CVAE_CHECKPOINT, DEFAULT_VOCAB_PATH
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.concept_vectors import (
    build_condition_dataset,
    collect_latent_embeddings,
)
from amp_vae.models.cvae import load_cvae_checkpoint
from amp_vae.utils.paths import repo_path
from amp_vae.utils.seed import set_seed
from amp_vae.utils.splits import load_splits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(repo_path("data", "processed", "master_dataset.csv")))
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--checkpoint", default=str(repo_path(DEFAULT_CVAE_CHECKPOINT)))
    parser.add_argument("--output", default=str(repo_path("artifacts", "latent_probe.json")))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Logistic regression inverse regularization strength.",
    )
    return parser.parse_args()


def _collect(model, vocab, data_path, idx, batch_size, device):
    ds, cols = build_condition_dataset(data_path, vocab, indices=idx.tolist())
    mu, cond = collect_latent_embeddings(model, ds, batch_size=batch_size, device=device)
    return mu.cpu().numpy().astype(np.float32), cond.cpu().numpy().astype(np.int32), cols


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    vocab = load_vocab_bundle(args.vocab)
    model = load_cvae_checkpoint(args.checkpoint, vocab, device=device)

    train_idx, val_idx, test_idx = load_splits()

    X_train, y_train, cols = _collect(model, vocab, args.data, train_idx, args.batch_size, device)
    X_test, y_test, _ = _collect(model, vocab, args.data, test_idx, args.batch_size, device)
    print(f"train mu={X_train.shape} test mu={X_test.shape} labels={cols}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    per_label: dict[str, dict] = {}
    aurocs: list[float] = []
    auprcs: list[float] = []
    for k, name in enumerate(cols):
        yt = y_train[:, k]
        ye = y_test[:, k]
        n_pos_train = int(yt.sum())
        n_pos_test = int(ye.sum())
        if n_pos_train < 2 or n_pos_test < 2 or n_pos_train == len(yt):
            per_label[name] = {
                "auroc": None,
                "auprc": None,
                "n_pos_train": n_pos_train,
                "n_pos_test": n_pos_test,
                "note": "insufficient positives",
            }
            continue
        clf = LogisticRegression(
            C=args.c,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=args.seed,
        )
        clf.fit(X_train_s, yt)
        p = clf.predict_proba(X_test_s)[:, 1]
        auroc = float(roc_auc_score(ye, p))
        auprc = float(average_precision_score(ye, p))
        aurocs.append(auroc)
        auprcs.append(auprc)
        prevalence = float(ye.mean())
        per_label[name] = {
            "auroc": auroc,
            "auprc": auprc,
            "test_prevalence": prevalence,
            "n_pos_train": n_pos_train,
            "n_pos_test": n_pos_test,
        }
        print(
            f"{name:28s}  auroc={auroc:.3f}  auprc={auprc:.3f}  "
            f"prev={prevalence:.3f}  n_pos_train={n_pos_train}  n_pos_test={n_pos_test}"
        )

    summary = {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "latent_dim": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "C": args.c,
        "per_label": per_label,
    }
    print(f"macro_auroc={summary['macro_auroc']:.3f}  macro_auprc={summary['macro_auprc']:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
