# Customer Churn Prediction System

An end-to-end supervised learning project for predicting telecom customer
churn. The workflow compares several classifiers with the same stratified
holdout split and preprocessing, tunes the strongest cross-validation
candidates, selects a probability threshold with a minimum churn-recall
requirement, and packages the selected preprocessing and model pipeline for
use in a Streamlit dashboard.

## Project layout

- `data/cleaned_data.csv` — cleaned customer dataset.
- `train.py` — preprocessing, model comparison, cross-validation, tuning, and
  model/metadata export.
- `evaluate.py` — held-out metrics, classification report, ROC curve, and
  permutation importance.
- `app.py` — interactive customer assessment and model-performance dashboard.
- `web/` — React dashboard interface that uses the generated evaluation,
  feature metadata, and ROC-curve artifacts. Its customer score is an
  illustrative UI preview and is not connected to the trained model.
- `models/` — generated joblib pipeline and feature metadata.
- `reports/` — generated comparison tables, metrics, plots, and summaries.

## Run locally

From this directory:

```powershell
python -m pip install -r requirements.txt
python train.py
python evaluate.py
python -m streamlit run app.py
```

`app.py` starts the separate Streamlit dashboard; it does not serve the React
interface. To run React from this project directory, use:

```powershell
npm --prefix web install
npm --prefix web run dev
```

Open the local URL printed by Vite (by default, `http://localhost:5173`).
Alternatively, change into `web` and run `npm install` followed by `npm run dev`.

The React customer assessment is currently a UI prototype: its assessment
score is illustrative and does not call the trained Python pipeline. The
model-performance view reads the latest local evaluation artifacts. The
Streamlit app remains the working model-connected dashboard.

Run the evaluation-data validation tests with
`python -m unittest discover -s tests -v`.

`train.py` uses a stratified 80/20 split with `random_state=42`. Numerical
features are median-imputed and standardized; categorical features are
most-frequent-imputed and one-hot encoded. `customerID` is excluded. Models
include a prior baseline, logistic regression, decision tree, random forest,
SVM, and gradient boosting. Class weights are used where supported.

Every model is compared with five-fold stratified cross-validation and on the
same held-out test set using accuracy, precision, churn recall, F1, ROC-AUC,
and false-positive/false-negative counts. The two strongest non-baseline
cross-validation F1 candidates are tuned with `GridSearchCV`. The final
threshold is selected from out-of-fold training probabilities to maximize F1
subject to at least 75% churn recall where feasible. The holdout remains
separate for final evaluation.

Training writes `models/churn_pipeline.joblib`,
`models/feature_metadata.json`, `reports/model_comparison.csv`, and
`reports/training_summary.json`. Evaluation writes
`reports/evaluation.json`, `reports/roc_curve.png`, and
`reports/feature_importance.csv`. The dashboard includes an adjustable alert
threshold, prediction probability, confusion matrix, ROC curve, classification
report, global permutation importance, and approximate local one-feature
sensitivity comparisons.

The local feature comparisons are model sensitivity indicators, not causal
explanations. Metrics describe one held-out split from this dataset; they do
not guarantee performance on future customers.

Evaluation checks that the input predictors match the trained model's feature
schema before scoring. The `customerID` identifier is excluded, and predictors
are reordered to the model's training order; missing or additional predictors
produce a clear error instead of being silently accepted.
