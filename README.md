# Customer churn prediction

Flask web app and notebooks for predicting telecom customer churn using a cleaned Telco dataset and an XGBoost model.

## Prerequisites

- Python 3.10+ (tested with 3.10)
- `pip`

## Setup

```bash
cd customer-churn-prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate the venv with `source .venv/bin/activate`.

## Cloning from Git (what others get)

Nothing important was removed from the **pipeline source**:

- **Data cleaning and feature code** live in `notebooks/` — e.g. `02_data_cleaning.ipynb`, `03_feature_engineering.ipynb`, `04_model_training.ipynb`. Those files stay in the repo; only **`.ipynb_checkpoints/`** autosave copies were deleted (they are not the real notebooks and should not be committed).
- **`data/`** is **not** ignored by `.gitignore`, so commit `Telco_customer_churn.csv` and `cleaned_churn.csv` if you want clones to run the app without re-running notebooks. If someone starts from raw data only, they run the notebooks in order to rebuild `cleaned_churn.csv`.
- **`model/churn_model.pkl`** is ignored (large binary). After clone, run `python train_model.py` once (needs `data/cleaned_churn.csv`).

**Minimal path for a new clone:** install deps → ensure `data/cleaned_churn.csv` exists (from git or from notebooks) → `python train_model.py` → `python app.py`.

## Train the model

The app loads `model/churn_model.pkl`. If the file is missing or invalid, train it from the cleaned CSV:

```bash
python train_model.py
```

This reads `data/cleaned_churn.csv` and writes `model/churn_model.pkl`.

## Run the app

From the project root (same folder as `app.py`):

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The form sends **tenure**, **monthly charges**, and **contract**; other features are set to a neutral baseline for a quick partial-input score.

## Project layout

| Path | Purpose |
|------|--------|
| `app.py` | Flask routes and model loading |
| `app/templates/` | `index.html`, `result.html` |
| `app/static/` | `style.css` |
| `data/` | `Telco_customer_churn.csv`, `cleaned_churn.csv` |
| `model/` | Trained `churn_model.pkl` (gitignored; regenerate with `train_model.py`) |
| `notebooks/` | Data cleaning, features, training, evaluation |
| `train_model.py` | Script to fit and save the model |

## Notebooks

Work through `notebooks/` in numeric order (`01_` … `06_`) for the full pipeline: understanding, cleaning, features, training, evaluation, optional SHAP.

## License

Use and modify for learning or your own projects; attribute the original Telco churn dataset as appropriate.
