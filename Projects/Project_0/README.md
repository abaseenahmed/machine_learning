# Iris Intelligence

A small Streamlit app that trains and compares five supervised classifiers on
scikit-learn's Iris dataset, selects a model by cross-validation, and predicts
an Iris variety from four measurements.

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
