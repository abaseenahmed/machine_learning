from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_utils import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    load_dataset,
    predict_prices,
    train_and_evaluate,
)


st.set_page_config(
    page_title="HomeValue | House Price Studio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    .block-container { max-width: 1440px; padding-top: 2rem; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e6eaf1;
        padding: 1rem 1.1rem; border-radius: 12px;
    }
    div[data-testid="stForm"] {
        background: white; padding: 1.1rem; border-radius: 12px;
        border: 1px solid #e6eaf1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the house-price model...")
def get_model_bundle(
    dataset_path: str, modified_time: float
) -> tuple[object, dict[str, float], pd.DataFrame, pd.DataFrame]:
    del modified_time
    data = load_dataset(Path(dataset_path))
    return train_and_evaluate(data)


st.title("🏠 HomeValue")
st.markdown(
    "### House price prediction studio\n"
    "Explore the dataset, estimate a property's sale price, or score a batch "
    "of properties."
)

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found at `{DATA_PATH}`. Make sure "
        "`data/cleaned_data.csv` is present in the project folder."
    )
    st.stop()

try:
    dataset = load_dataset(DATA_PATH)
    model, metrics, holdout_predictions, feature_importance = get_model_bundle(
        str(DATA_PATH), DATA_PATH.stat().st_mtime
    )
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not prepare the prediction model: {error}")
    st.stop()

st.caption(
    f"Model trained on {len(dataset):,} cleaned historical sales. "
    "Predictions are estimates from this dataset, not appraisals or financial advice."
)

tab_estimate, tab_market, tab_performance = st.tabs(
    ["💵 Estimate a home", "📊 Explore the market", "🧪 Model performance"]
)

with tab_estimate:
    estimate_tab, batch_tab = st.tabs(["Single property", "Batch CSV"])

    with estimate_tab:
        st.subheader("Property details")
        st.caption(
            "Enter property characteristics. Initial values use the dataset "
            "median; available ranges reflect the historical training data."
        )

        with st.form("property_estimate_form"):
            city_values = sorted(dataset["city"].dropna().astype(str).unique())
            city = st.selectbox("City", city_values)
            zip_values = sorted(
                dataset.loc[dataset["city"].astype(str) == city, "statezip"]
                .dropna()
                .astype(str)
                .unique()
            )
            statezip = st.selectbox("ZIP code", zip_values)

            columns = st.columns(3)
            property_values: dict[str, float | int] = {}
            for index, feature in enumerate(NUMERIC_FEATURES):
                feature_values = pd.to_numeric(dataset[feature], errors="coerce")
                minimum = float(feature_values.min())
                maximum = float(feature_values.max())
                default = float(feature_values.median())
                is_integer = feature not in {"bathrooms", "floors"}
                step = 1.0 if is_integer else 0.25
                with columns[index % len(columns)]:
                    value = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        step=step,
                        help=f"Observed training range: {minimum:g} to {maximum:g}.",
                        key=f"single_{feature}",
                    )
                property_values[feature] = int(value) if is_integer else value

            submitted = st.form_submit_button(
                "Estimate sale price", type="primary", width="stretch"
            )

        if submitted:
            property_row = pd.DataFrame(
                [{**property_values, "city": city, "statezip": statezip}],
                columns=FEATURE_COLUMNS,
            )
            estimate = float(predict_prices(model, property_row)[0])
            st.success("Estimate ready")
            st.metric("Estimated sale price", f"${estimate:,.0f}")
            st.caption(
                "The estimate is a model output. Actual sale prices can differ "
                "substantially based on timing, condition, and other factors."
            )

    with batch_tab:
        st.subheader("Estimate multiple properties")
        st.write(
            "Upload a CSV with the property feature columns shown in the "
            "downloadable template. The model ignores additional columns."
        )
        template = pd.DataFrame(
            [{feature: dataset.iloc[0][feature] for feature in FEATURE_COLUMNS}]
        )
        st.download_button(
            "Download CSV template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="house_price_template.csv",
            mime="text/csv",
            width="stretch",
        )
        upload = st.file_uploader(
            "Upload property CSV", type=["csv"], key="batch_property_csv"
        )
        if upload is not None:
            try:
                batch = pd.read_csv(upload)
                missing_columns = [
                    feature for feature in FEATURE_COLUMNS
                    if feature not in batch.columns
                ]
                if missing_columns:
                    raise ValueError(
                        "Missing required columns: " + ", ".join(missing_columns)
                    )
                if batch.empty:
                    raise ValueError("The uploaded CSV contains no property rows.")
                if len(batch) > 10_000:
                    raise ValueError("Please upload no more than 10,000 rows.")

                batch = batch.loc[:, FEATURE_COLUMNS].copy()
                for feature in NUMERIC_FEATURES:
                    batch[feature] = pd.to_numeric(batch[feature], errors="coerce")
                if batch[NUMERIC_FEATURES].isna().any().any():
                    raise ValueError(
                        "Numeric feature columns must contain valid numbers "
                        "and no blank values."
                    )
                if not np.isfinite(batch[NUMERIC_FEATURES].to_numpy()).all():
                    raise ValueError("Numeric values must be finite.")
                if (batch[NUMERIC_FEATURES] < 0).any().any():
                    raise ValueError("Numeric feature values cannot be negative.")
                for feature in CATEGORICAL_FEATURES:
                    batch[feature] = batch[feature].astype("string").str.strip()
                    if batch[feature].isna().any() or batch[feature].eq("").any():
                        raise ValueError(f"{feature} cannot be blank.")

                predictions = predict_prices(model, batch)
                results = batch.copy()
                results["estimated_price"] = np.round(predictions, 2)
                st.dataframe(
                    results.style.format({"estimated_price": "${:,.0f}"}),
                    width="stretch",
                    hide_index=True,
                )
                st.download_button(
                    "Download estimates",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="house_price_estimates.csv",
                    mime="text/csv",
                    width="stretch",
                )
                unknown_cities = sorted(
                    set(batch["city"].astype(str)) - set(dataset["city"].astype(str))
                )
                if unknown_cities:
                    st.warning(
                        "Some uploaded cities were not in the training data; "
                        "their location may have limited influence on estimates."
                    )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                st.error(f"Unable to process this CSV: {error}")

with tab_market:
    st.subheader("Historical sales at a glance")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Sales in dataset", f"{len(dataset):,}")
    metric_columns[1].metric("Median sale price", f"${dataset['price'].median():,.0f}")
    metric_columns[2].metric("Median living area", f"{dataset['sqft_living'].median():,.0f} sq ft")
    metric_columns[3].metric("Cities represented", f"{dataset['city'].nunique():,}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        price_chart = px.histogram(
            dataset,
            x="price",
            nbins=45,
            labels={"price": "Sale price (USD)", "count": "Sales"},
            title="Distribution of sale prices",
            color_discrete_sequence=["#4361ee"],
        )
        price_chart.update_layout(bargap=0.08)
        st.plotly_chart(price_chart, width="stretch")
    with chart_right:
        scatter = px.scatter(
            dataset,
            x="sqft_living",
            y="price",
            color="bedrooms",
            hover_data=["city", "statezip"],
            labels={
                "sqft_living": "Living area (sq ft)",
                "price": "Sale price (USD)",
                "bedrooms": "Bedrooms",
            },
            title="Living area and sale price",
            opacity=0.65,
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(scatter, width="stretch")

    city_summary = (
        dataset.groupby("city", as_index=False)
        .agg(median_price=("price", "median"), sales=("price", "size"))
        .query("sales >= 5")
        .nlargest(12, "median_price")
        .sort_values("median_price")
    )
    city_chart = px.bar(
        city_summary,
        x="median_price",
        y="city",
        orientation="h",
        color="median_price",
        labels={"median_price": "Median sale price (USD)", "city": "City"},
        title="12 highest-median-price cities (at least 5 sales)",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(city_chart, width="stretch")
    st.caption(
        "Market charts summarize the supplied historical data; they are not "
        "current market statistics."
    )

with tab_performance:
    st.subheader("Held-out evaluation")
    st.caption(
        "Metrics are calculated on a fixed 20% holdout split not used to fit "
        "that evaluation model."
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Mean absolute error", f"${metrics['mae']:,.0f}")
    metric_columns[1].metric("Root mean squared error", f"${metrics['rmse']:,.0f}")
    metric_columns[2].metric("R² score", f"{metrics['r2']:.3f}")

    actual_vs_predicted = px.scatter(
        holdout_predictions,
        x="actual",
        y="predicted",
        labels={"actual": "Actual price (USD)", "predicted": "Predicted price (USD)"},
        title="Actual vs. predicted prices",
        opacity=0.65,
        color_discrete_sequence=["#4361ee"],
    )
    maximum_price = max(
        holdout_predictions["actual"].max(),
        holdout_predictions["predicted"].max(),
    )
    actual_vs_predicted.add_trace(
        go.Scatter(
            x=[0, maximum_price],
            y=[0, maximum_price],
            mode="lines",
            name="Perfect estimate",
            line={"color": "#e63946", "dash": "dash"},
        )
    )
    st.plotly_chart(actual_vs_predicted, width="stretch")

    importance_chart = px.bar(
        feature_importance.head(12).sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Most influential model features",
        labels={"importance": "Random Forest importance", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(importance_chart, width="stretch")
    st.caption(
        "Feature importance describes how the fitted forest used inputs; it "
        "does not establish causation. The data contains historical sales only."
    )
