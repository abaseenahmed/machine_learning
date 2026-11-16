# Supervised model comparison

This project compares eight supervised classifiers using scikit-learn's
built-in Wisconsin Diagnostic Breast Cancer dataset. It is a real-world
diagnostic dataset of 569 cases with 30 numeric measurements; the target is
malignant or benign. The dataset loader obtains the data from scikit-learn
without requiring a manual download.

## Run

From this directory:

```powershell
python -m pip install scikit-learn pandas joblib
python model_comparison.py
```

The script uses a stratified 80/20 train/test split. Model selection uses
five-fold stratified cross-validation on the training split only, ranking by
mean ROC-AUC. Scaling is part of each pipeline, so it is fit independently
inside each cross-validation fold. The test metrics are reported for final
comparison and do not drive model selection.

Malignant cases are treated as the positive class for malignant precision,
recall, and F1. ROC-AUC and malignant recall are reported because accuracy
alone does not describe the cost of missing a malignant case. This benchmark
is educational and the results do not establish clinical suitability.

## Outputs

- `models/selected_model.joblib` — fitted pipeline chosen by cross-validation
- `reports/model_comparison.csv` — cross-validation and held-out metrics for
  every candidate
- `reports/evaluation.json` — dataset, selection method, selected model, and
  held-out metrics

The evaluated candidates are a most-frequent baseline, logistic regression,
K-nearest neighbors, support vector machine, decision tree, random forest,
gradient boosting, and Gaussian naive Bayes.

## Simple training example

For a shorter example using the local `Datasets/classification.csv` file, run:

```powershell
python -m pip install scikit-learn pandas joblib
python train_simple.py
```

The script splits the data into stratified training and test sets, preprocesses
numeric and categorical features, trains logistic regression, reports test
accuracy, and saves the fitted pipeline to
`models/simple_classifier.joblib`.
