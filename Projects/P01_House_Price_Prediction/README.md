# House Price Prediction Studio

A Streamlit dashboard for estimating house prices from the supplied historical
sales dataset, exploring market patterns, reviewing model performance, and
making batch estimates from CSV files.

## Run locally

From this directory, install the dependencies and start the dashboard:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app loads `data/cleaned_data.csv`, trains a Random Forest on log-transformed
prices, reports metrics on a fixed 20% holdout split, and fits the prediction
model on all valid sales. The trained model is cached in memory while the app
is running; it is retrained when the cleaned dataset changes.

## Dashboard features

- Enter property details and receive an estimated sale price.
- Upload a CSV for batch estimates and download the results.
- Download a CSV template with the expected feature columns.
- Explore sale-price distributions, living-area relationships, and city
  summaries, with filters for city and bedroom count.
- Review holdout MAE, RMSE, R², actual-versus-predicted values, and feature
  importance.
- View the dashboard in a responsive, polished layout with consistent charts
  and metric cards.

## Batch CSV format

The CSV must include these columns:

`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`,
`view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`,
`yr_renovated`, `city`, and `statezip`.

Numeric feature values must be finite and non-negative. Additional columns are
ignored, and uploads are limited to 10,000 rows.
The app warns when uploaded city or ZIP-code values were not in the training
data, or when a city/ZIP-code combination was not observed.

## Important limitations

The supplied records are historical sales (dated 2014); estimates and charts
reflect that data and should not be treated as current market valuations,
appraisals, or financial advice. Model metrics describe a held-out sample from
the same dataset and do not guarantee performance on future sales or other
markets.
