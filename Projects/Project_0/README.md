# Iris Intelligence

A polished Streamlit ML studio for exploring the Iris dataset and making
single or batch flower-variety predictions with five supervised classifiers.

## Run locally

From this directory, install the dependencies and start the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app compares Logistic Regression, Support Vector Machine, K-Nearest
Neighbors, Random Forest, and Gradient Boosting. It uses a stratified holdout
for test metrics and five-fold cross-validation on the training partition to
select the recommended model.

## Features

- Enter measurements directly, start from representative flower examples, and
  choose which trained classifier to use.
- Upload a CSV for batch predictions, with input validation and downloadable
  results. Use the in-app template to get the required column names.
- Explore feature relationships and per-variety summaries.
- Compare model scores and inspect the recommended model's confusion matrix.
- Switch between Aurora, Paper, and Midnight visual themes.

Batch CSV files must include `sepal length (cm)`, `sepal width (cm)`,
`petal length (cm)`, and `petal width (cm)`, with finite, non-negative numeric
values. Uploads are limited to 10,000 rows.
