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
    :root {
        --ink: #17233b;
        --muted: #718096;
        --line: #e7edf5;
        --surface: #ffffff;
    }
    .stApp { background: #f4f7fb; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244, 247, 251, 0.88); }
    .block-container { max-width: 1440px; padding: 2.2rem 2.5rem 4rem; }
    html, body, [class*="css"] { font-family: 'Aptos', 'Segoe UI', sans-serif; }
    h1, h2, h3 { color: var(--ink); font-family: 'Aptos Display', 'Segoe UI', sans-serif; letter-spacing: -0.035em; }
    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.15rem !important; }
    .hero {
        position: relative; overflow: hidden; margin: 0 0 1.8rem; padding: 2rem 2.2rem;
        border: 1px solid rgba(255,255,255,.12); border-radius: 22px;
        color: #fff; background: linear-gradient(118deg, #172b50 0%, #243d70 58%, #4265c7 100%);
        box-shadow: 0 16px 38px rgba(28, 52, 94, .16);
    }
    .hero:after {
        position: absolute; top: -105px; right: -40px; width: 340px; height: 340px;
        border: 1px solid rgba(255,255,255,.12); border-radius: 50%;
        box-shadow: 0 0 0 34px rgba(255,255,255,.035), 0 0 0 70px rgba(255,255,255,.025);
        content: '';
    }
    .hero-eyebrow { color: #b8cbff; font-size: .73rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .hero h1 { margin: .55rem 0 .45rem; color: #fff; font-size: clamp(2rem, 4vw, 3rem); }
    .hero-copy { max-width: 640px; margin: 0; color: #d3def3; font-size: 1rem; line-height: 1.65; }
    .hero-note {
        position: absolute; z-index: 1; top: 2rem; right: 2rem; padding: .55rem .85rem;
        border: 1px solid rgba(255,255,255,.2); border-radius: 999px;
        color: #e8efff; background: rgba(255,255,255,.09); font-size: .78rem; font-weight: 600;
    }
    .stCaption { color: var(--muted); }
    [data-testid="stTabs"] [role="tablist"] {
        gap: .45rem; padding: .38rem; border: 1px solid var(--line); border-radius: 14px;
        background: #eaf0f8;
    }
    [data-testid="stTabs"] button[role="tab"] {
        min-height: 2.7rem; border-radius: 10px; color: #62718a; font-weight: 600;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--ink); background: #fff; box-shadow: 0 2px 8px rgba(30, 50, 80, .08);
    }
    [data-testid="stMetric"] {
        min-height: 112px; padding: 1.15rem 1.25rem; border: 1px solid var(--line);
        border-radius: 16px; background: var(--surface);
        box-shadow: 0 4px 14px rgba(31, 51, 82, .035);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Aptos Display', 'Segoe UI', sans-serif; }
    div[data-testid="stForm"] {
        padding: 1.35rem; border: 1px solid var(--line); border-radius: 18px;
        background: #fff; box-shadow: 0 8px 24px rgba(31, 51, 82, .045);
    }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); border-radius: 16px; }
    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
        min-height: 2.75rem; border-radius: 10px; font-weight: 600;
        transition: transform .16s ease, box-shadow .16s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px); box-shadow: 0 6px 16px rgba(45, 80, 170, .13);
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #c9d4e5; border-radius: 14px; background: #f9fbfe;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 3rem; }
        .hero { padding: 1.5rem; }
        .hero-note { position: static; display: inline-block; margin-top: 1.2rem; }
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


def style_chart(chart: go.Figure) -> go.Figure:
    chart.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Aptos, Segoe UI, sans-serif", "color": "#52627a", "size": 12},
        title={"font": {"family": "Aptos Display, Segoe UI, sans-serif", "size": 16, "color": "#17233b"}},
        margin={"l": 12, "r": 12, "t": 58, "b": 12},
        coloraxis_colorbar={"outlinewidth": 0},
        hoverlabel={"bgcolor": "#17233b", "font": {"color": "#ffffff"}},
    )
    chart.update_xaxes(showgrid=True, gridcolor="#edf1f7", zeroline=False)
    chart.update_yaxes(showgrid=True, gridcolor="#edf1f7", zeroline=False)
    return chart


st.markdown(
    """
    <section class="hero">
      <div class="hero-eyebrow">Real estate intelligence</div>
      <h1>HomeValue Studio</h1>
      <p class="hero-copy">A clearer view of home values. Explore historical
      sales, build an informed estimate, and understand what drives the model.</p>
      <div class="hero-note">● &nbsp; Market insights &amp; estimates</div>
    </section>
    """,
    unsafe_allow_html=True,
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
    st.subheader("Explore the market")
    st.caption(
        "Narrow the historical view by location or bedroom count. Leave a filter "
        "empty to include every value."
    )
    city_options = sorted(dataset["city"].dropna().astype(str).unique())
    filter_city, filter_bedrooms = st.columns(2)
    with filter_city:
        selected_cities = st.multiselect(
            "Cities", city_options, placeholder="All cities"
        )
    available_market = dataset
    if selected_cities:
        available_market = available_market.loc[
            available_market["city"].astype(str).isin(selected_cities)
        ]
    bedroom_options = sorted(available_market["bedrooms"].dropna().astype(int).unique())
    with filter_bedrooms:
        selected_bedrooms = st.multiselect(
            "Bedrooms", bedroom_options, placeholder="All bedroom counts"
        )
    market_data = available_market
    if selected_bedrooms:
        market_data = market_data.loc[market_data["bedrooms"].isin(selected_bedrooms)]

    st.markdown("#### Historical sales at a glance")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Sales in view", f"{len(market_data):,}")
    metric_columns[1].metric("Median sale price", f"${market_data['price'].median():,.0f}")
    metric_columns[2].metric("Median living area", f"{market_data['sqft_living'].median():,.0f} sq ft")
    metric_columns[3].metric("Cities represented", f"{market_data['city'].nunique():,}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        price_chart = px.histogram(
            market_data,
            x="price",
            nbins=45,
            labels={"price": "Sale price (USD)", "count": "Sales"},
            title="Distribution of sale prices",
            color_discrete_sequence=["#4361ee"],
        )
        price_chart.update_layout(bargap=0.08)
        st.plotly_chart(style_chart(price_chart), width="stretch")
    with chart_right:
        scatter = px.scatter(
            market_data,
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
        st.plotly_chart(style_chart(scatter), width="stretch")

    city_summary = (
        market_data.groupby("city", as_index=False)
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
    st.plotly_chart(style_chart(city_chart), width="stretch")
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
            line={"color": "#020202", "dash": "dash"},
        )
    )
    st.plotly_chart(style_chart(actual_vs_predicted), width="stretch")

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
    st.plotly_chart(style_chart(importance_chart), width="stretch")
    st.caption(
        "Feature importance describes how the fitted forest used inputs; it "
        "does not establish causation. The data contains historical sales only."
    )
