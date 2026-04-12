# Architecture

## Goal

Keep the current notebook workflow intact while moving reusable logic into `src/amp_vae`.

## Layers

1. `src/amp_vae`
   - reusable library code
   - data schema and preprocessing
   - model definitions
   - inference helpers
   - evaluation helpers
   - training loops and checkpointing

2. `data`
   - `raw`: source inputs
   - `processed`: cleaned datasets and generated tables
   - `master_dataset.csv`: default training dataset
   - `external/aipampds`: external screening outputs

3. `models`
   - checkpoints and serialized artifacts

4. `scripts`
   - thin command-line entry points
   - no business logic

5. `app/streamlit_app.py`
   - visualization and manual triage
   - generation and scoring UI
   - no training loops

6. `notebooks`
   - exploratory analysis
   - plots
   - research notes

## Migration rule

Do not remove the notebook code until the equivalent function exists in `src/amp_vae` and the new path is validated.
