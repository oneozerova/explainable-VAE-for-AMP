# explainable-VAE-for-AMP

## Layout

- `src/amp_vae`: reusable library code for data, models, inference, evaluation, and training
- `scripts`: CLI entrypoints for dataset build, training, generation, and evaluation
- `app/streamlit_app.py`: Streamlit UI for generation and screening
- `data/raw`: immutable source inputs
- `data/processed`: cleaned datasets and generated tabular outputs, with `master_dataset.csv` as the default training input
- `data/external/aipampds`: external screening artifacts
- `models`: checkpoints and model artifacts
- `notebooks`: removed from the refactored branch

## Streamlit

Run the app after installing dependencies:

```bash
streamlit run app/streamlit_app.py
```

The app expects:

- `models/vocab.pkl`
- `models/best_cvae.pt`
- `models/best_external_amp_classifier.pt`
- `models/external_amp_classifier_artifacts.json`

## Docker

Run the Streamlit app locally in Docker:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

The compose file mounts the repository into the container, so the app reads your local `data/` and `models/` directories directly.
