from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment fallback
    torch = None
    TORCH_AVAILABLE = False

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_CLASSIFIER_CHECKPOINT, DEFAULT_CVAE_CHECKPOINT, DEFAULT_VOCAB_PATH
from amp_vae.config import GENERATION_CONDITION_COLUMNS
from amp_vae.data.schema import SEQUENCE_COLUMN
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.generation_metrics import uniqueness_rate, validity_rate
from amp_vae.evaluation.rank_candidates import rank_candidates
from amp_vae.inference.generate import generation_condition_vector, generate_batch, is_valid_sequence
from amp_vae.inference.score import predict_probabilities
from amp_vae.models.classifier import load_external_classifier_checkpoint
from amp_vae.models.cvae import load_cvae_checkpoint
from amp_vae.utils.paths import repo_path


st.set_page_config(page_title="AMP VAE Studio", layout="wide")
st.title("AMP VAE Studio")
st.caption("Streamlit layer for generation, screening, and candidate triage.")


@st.cache_data
def load_json_artifact(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource
def load_models(vocab_path: str, cvae_path: str, classifier_path: str):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available in this environment")
    vocab = load_vocab_bundle(vocab_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae = load_cvae_checkpoint(cvae_path, vocab, device=device)
    classifier = load_external_classifier_checkpoint(
        classifier_path,
        vocab=vocab,
        num_labels=len(CONDITION_COLUMNS),
        device=device,
    )
    return vocab, cvae, classifier, device


def sequences_to_frame(sequences, vocab, classifier=None, device=None):
    frame = pd.DataFrame({SEQUENCE_COLUMN: sequences})
    frame["length"] = frame[SEQUENCE_COLUMN].astype(str).str.len()
    frame["valid"] = frame[SEQUENCE_COLUMN].astype(str).map(is_valid_sequence)
    frame["unique_flag"] = ~frame[SEQUENCE_COLUMN].duplicated()

    if classifier is not None:
        probs = predict_probabilities(classifier, frame[SEQUENCE_COLUMN].tolist(), vocab, device=device)
        prob_cols = [f"p_{c}" for c in CONDITION_COLUMNS]
        prob_df = pd.DataFrame(probs, columns=prob_cols)
        frame = pd.concat([frame, prob_df], axis=1)

    return frame


with st.sidebar:
    st.header("Artifacts")
    vocab_path = st.text_input("Vocabulary", value=str(repo_path(DEFAULT_VOCAB_PATH)))
    cvae_path = st.text_input("cVAE checkpoint", value=str(repo_path(DEFAULT_CVAE_CHECKPOINT)))
    classifier_path = st.text_input("Classifier checkpoint", value=str(repo_path(DEFAULT_CLASSIFIER_CHECKPOINT)))
    classifier_artifacts_path = st.text_input(
        "Classifier artifacts JSON",
        value=str(repo_path("models", "external_amp_classifier_artifacts.json")),
    )

    st.divider()
    st.header("Generation")
    selected_labels = []
    for label in GENERATION_CONDITION_COLUMNS:
        if st.checkbox(label, value=(label == "is_anti_gram_positive"), key=f"cond_{label}"):
            selected_labels.append(label)
    n_samples = st.slider("Samples", min_value=8, max_value=512, value=64, step=8)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.05)
    alpha = st.slider("Condition strength", min_value=0.0, max_value=2.0, value=1.0, step=0.05)


tabs = st.tabs(["Generate", "Score", "Overview"])


with tabs[2]:
    artifacts = load_json_artifact(classifier_artifacts_path)
    cols = st.columns(3)
    cols[0].metric("Torch", "available" if TORCH_AVAILABLE else "missing")
    cols[1].metric("Condition labels", str(len(GENERATION_CONDITION_COLUMNS)))
    cols[2].metric("Checkpoint ready", "yes" if Path(cvae_path).exists() else "no")
    st.markdown(
        """
        This app is intentionally thin:

        - `src/amp_vae` holds the reusable logic.
        - Streamlit only orchestrates generation, classification, and ranking.
        - notebooks remain available for exploratory work.
        """
    )
    if artifacts:
        st.subheader("Classifier artifacts")
        st.json(
            {
                "macro_auroc": artifacts.get("test_macro_default", {}).get("macro_auroc"),
                "macro_f1": artifacts.get("test_macro_default", {}).get("macro_f1"),
                "best_thresholds": artifacts.get("best_thresholds", {}),
            }
        )


try:
    vocab, cvae, classifier, device = load_models(vocab_path, cvae_path, classifier_path)
    st.sidebar.success(f"Loaded models on {device}")
except Exception as exc:  # pragma: no cover - UI fallback
    vocab = cvae = classifier = device = None
    st.sidebar.error(f"Could not load artifacts: {exc}")

top_cols = st.columns(4)
top_cols[0].metric("Vocabulary", "loaded" if vocab is not None else "missing")
top_cols[1].metric("cVAE", "loaded" if cvae is not None else "missing")
top_cols[2].metric("Classifier", "loaded" if classifier is not None else "missing")
top_cols[3].metric("Runtime", str(device) if device is not None else "n/a")


with tabs[0]:
    st.subheader("Generate peptides")
    if cvae is None or vocab is None:
        st.info("Load a valid vocabulary and cVAE checkpoint to generate sequences.")
    elif not TORCH_AVAILABLE:
        st.warning("torch is not available, so generation is disabled in this environment.")
    else:
        condition_map = {name: int(name in selected_labels) for name in GENERATION_CONDITION_COLUMNS}
        condition = generation_condition_vector(**condition_map)
        if st.button("Generate"):
            sequences = generate_batch(
                cvae,
                vocab=vocab,
                condition=condition,
                n_samples=n_samples,
                temperature=temperature,
                alpha=alpha,
                device=device,
            )
            frame = sequences_to_frame(sequences, vocab=vocab, classifier=classifier, device=device)
            frame["valid"] = frame[SEQUENCE_COLUMN].astype(str).map(is_valid_sequence)
            frame["length"] = frame[SEQUENCE_COLUMN].astype(str).str.len()
            st.metric("Validity", f"{validity_rate(frame[SEQUENCE_COLUMN].tolist()):.2f}")
            st.metric("Uniqueness", f"{uniqueness_rate(frame[SEQUENCE_COLUMN].tolist()):.2f}")
            st.dataframe(frame, use_container_width=True)
            st.download_button(
                "Download CSV",
                frame.to_csv(index=False).encode("utf-8"),
                file_name="generated_peptides.csv",
                mime="text/csv",
            )


with tabs[1]:
    st.subheader("Score peptides")
    source = st.radio("Input source", ["Paste sequences", "Upload CSV"], horizontal=True)
    sequences: list[str] = []

    if source == "Paste sequences":
        raw_text = st.text_area("One sequence per line")
        sequences = [line.strip().upper() for line in raw_text.splitlines() if line.strip()]
    else:
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            upload_df = pd.read_csv(uploaded)
            seq_col = SEQUENCE_COLUMN if SEQUENCE_COLUMN in upload_df.columns else upload_df.columns[0]
            sequences = upload_df[seq_col].astype(str).str.strip().str.upper().tolist()

    if st.button("Score") and sequences:
        if classifier is None or vocab is None:
            st.warning("Load a classifier checkpoint to score sequences.")
        else:
            frame = sequences_to_frame(sequences, vocab=vocab, classifier=classifier, device=device)
            artifacts = load_json_artifact(classifier_artifacts_path) or {}
            thresholds = artifacts.get("best_thresholds", {})
            for label, thr in thresholds.items():
                col = f"p_{label}"
                if col in frame.columns:
                    frame[f"{label}_hit"] = frame[col] >= float(thr)
            if selected_labels:
                requested_cols = [f"p_{label}" for label in selected_labels]
                ranked = rank_candidates(frame, requested_cols=requested_cols, off_target_cols=[])
                st.dataframe(ranked, use_container_width=True)
            else:
                st.dataframe(frame, use_container_width=True)
