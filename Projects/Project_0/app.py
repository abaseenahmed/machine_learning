from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="Iris Intelligence | ML Studio",
    page_icon="✿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def train_models() -> dict[str, object]:
    """Compare classifiers on a stratified split and return the CV winner."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )

    candidates: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(probability=True, random_state=42)),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250, random_state=42, n_jobs=1
                    ),
                )
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                (
                    "classifier",
                    GradientBoostingClassifier(random_state=42),
                )
            ]
        ),
    }

    comparison: list[dict[str, float | str]] = []
    test_predictions_by_model: dict[str, np.ndarray] = {}
    for name, pipeline in candidates.items():
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="f1_weighted",
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        test_predictions_by_model[name] = predictions
        comparison.append(
            {
                "Model": name,
                "CV F1": float(cv_scores.mean()),
                "Accuracy": float(accuracy_score(y_test, predictions)),
                "Weighted F1": float(f1_score(y_test, predictions, average="weighted")),
            }
        )

    # Cross-validation on the training partition chooses the model; the held-out
    # test partition remains an unbiased comparison of the finalists.
    best_name = max(comparison, key=lambda row: (row["CV F1"], row["Weighted F1"]))[
        "Model"
    ]
    best_model = candidates[best_name]
    best_model.fit(iris.data, iris.target)

    return {
        "iris": iris,
        "comparison": pd.DataFrame(comparison).sort_values(
            ["CV F1", "Weighted F1"], ascending=False
        ),
        "best_name": best_name,
        "best_model": best_model,
        "y_test": y_test,
        "best_test_predictions": test_predictions_by_model[best_name],
    }


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root {
            --ink: #14213d; --muted: #71809a; --line: #e8edf5;
            --violet: #6657e8; --mint: #18a98b; --surface: #ffffff;
        }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: #f5f7fb; color: var(--ink); }
        .block-container { max-width: 1440px; padding-top: 2.1rem; padding-bottom: 3rem; }
        header[data-testid="stHeader"] { background: transparent; }
        h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; color: var(--ink); }
        h1 { font-size: 2.65rem !important; font-weight: 800 !important; }
        h2 { font-size: 1.35rem !important; font-weight: 750 !important; }
        p, label, span { color: inherit; }
        .hero {
            position: relative; overflow: hidden; border-radius: 24px; padding: 2.15rem 2.35rem;
            background: linear-gradient(115deg, #171d3a 0%, #292c64 56%, #574bd1 100%);
            color: white; margin-bottom: 1.45rem; box-shadow: 0 18px 50px rgba(30,35,84,.15);
        }
        .hero:after {
            content: ''; position: absolute; width: 310px; height: 310px; right: 5%; top: -190px;
            border: 1px solid rgba(255,255,255,.12); border-radius: 50%;
            box-shadow: 0 0 0 34px rgba(255,255,255,.035), 0 0 0 70px rgba(255,255,255,.025);
        }
        .eyebrow { color: #bcb7ff; font-size: .73rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
        .hero-title { font: 800 2.35rem 'Manrope', sans-serif; letter-spacing: -.045em; margin: .45rem 0 .48rem; }
        .hero-copy { color: #d7d9f0; font-size: 1rem; max-width: 640px; margin: 0; }
        .hero-badge {
            display: inline-flex; gap: 8px; align-items: center; margin-top: 1.25rem; padding: 8px 12px;
            border: 1px solid rgba(255,255,255,.2); border-radius: 999px; color: #eeeaff;
            background: rgba(255,255,255,.08); font-size: .78rem; font-weight: 600;
        }
        .panel {
            background: var(--surface); border: 1px solid var(--line); border-radius: 18px;
            padding: 1.3rem 1.4rem; box-shadow: 0 6px 20px rgba(27,42,75,.035);
        }
        .panel-kicker { color: var(--muted); font-size: .73rem; font-weight: 700; text-transform: uppercase; letter-spacing: .11em; }
        .prediction-name { color: #24205d; font: 800 2rem 'Manrope', sans-serif; letter-spacing: -.04em; }
        .prediction-note { color: var(--muted); font-size: .9rem; margin-top: .15rem; }
        .stat-card {
            background: white; border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.15rem;
            box-shadow: 0 5px 18px rgba(27,42,75,.025); min-height: 94px;
        }
        .stat-label { color: var(--muted); font-size: .78rem; font-weight: 600; }
        .stat-value { color: var(--ink); font: 800 1.5rem 'Manrope', sans-serif; margin-top: .3rem; }
        .section-caption { color: var(--muted); font-size: .88rem; margin-top: -.65rem; margin-bottom: 1rem; }
        div[data-testid="stSlider"] > label { font-weight: 600; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
        .stButton > button { border-radius: 10px; font-weight: 700; }
        .footer { text-align: center; color: #929db0; font-size: .76rem; padding-top: 1.8rem; }
        @media (max-width: 700px) {
            .block-container { padding: 1rem; }
            .hero { padding: 1.5rem; border-radius: 18px; }
            .hero-title { font-size: 1.8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(best_name: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">Machine learning · made clear</div>
          <div class="hero-title">Iris Intelligence</div>
          <p class="hero-copy">Explore a model that learns from flower measurements.
          Compare five supervised algorithms, then make a prediction in seconds.</p>
          <div class="hero-badge"><span>✦</span> Recommended model: {best_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()
    artifacts = train_models()
    iris = artifacts["iris"]
    comparison = artifacts["comparison"]
    model = artifacts["best_model"]
    best_name = artifacts["best_name"]
    y_test = artifacts["y_test"]

    render_hero(best_name)

    st.markdown("## Model snapshot")
    st.markdown(
        '<div class="section-caption">A quick look at the dataset and the strongest model.</div>',
        unsafe_allow_html=True,
    )
    best_row = comparison.iloc[0]
    metric_columns = st.columns(4)
    metric_values = [
        ("Flowers in dataset", f"{len(iris.data):,}"),
        ("Flower varieties", str(len(iris.target_names))),
        ("Best model", best_name),
        ("Held-out accuracy", f"{best_row['Accuracy']:.0%}"),
    ]
    for column, (label, value) in zip(metric_columns, metric_values):
        with column:
            render_metric(label, value)

    st.write("")
    prediction_column, analysis_column = st.columns([0.92, 1.08], gap="large")

    with prediction_column:
        st.markdown("## Make a prediction")
        st.markdown(
            '<div class="section-caption">Adjust the flower measurements below.</div>',
            unsafe_allow_html=True,
        )
        input_columns = st.columns(2)
        measurements: list[float] = []
        for index, feature_name in enumerate(iris.feature_names):
            values = iris.data[:, index]
            with input_columns[index % 2]:
                measurement = st.slider(
                    feature_name.replace(" (cm)", "").title() + " (cm)",
                    min_value=round(float(values.min()), 1),
                    max_value=round(float(values.max()), 1),
                    value=round(float(np.median(values)), 1),
                    step=0.1,
                    help=f"Observed range: {values.min():.1f}–{values.max():.1f} cm",
                )
            measurements.append(float(measurement))

        sample = np.array([measurements])
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        predicted_name = str(iris.target_names[prediction]).title()
        confidence = float(probabilities[prediction])

        st.markdown(
            f"""
            <div class="panel" style="margin-top:.7rem">
              <div class="panel-kicker">Predicted variety</div>
              <div class="prediction-name">{predicted_name}</div>
              <div class="prediction-note">The model's estimated confidence is {confidence:.1%}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        probability_frame = pd.DataFrame(
            {
                "Variety": [str(name).title() for name in iris.target_names],
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=True)
        probability_chart = px.bar(
            probability_frame,
            x="Probability",
            y="Variety",
            orientation="h",
            text=probability_frame["Probability"].map(lambda value: f"{value:.0%}"),
            color="Probability",
            color_continuous_scale=["#c9c5ff", "#6657e8"],
            range_x=[0, 1],
        )
        probability_chart.update_traces(textposition="outside", cliponaxis=False)
        probability_chart.update_layout(
            height=220,
            margin=dict(l=0, r=30, t=18, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="#edf0f6", tickformat=".0%", title=None),
            yaxis=dict(title=None, showgrid=False),
            font=dict(family="DM Sans, sans-serif", color="#52617a"),
        )
        st.plotly_chart(probability_chart, width="stretch", config={"displayModeBar": False})

    with analysis_column:
        st.markdown("## Algorithm comparison")
        st.markdown(
            '<div class="section-caption">Five models evaluated on identical data splits.</div>',
            unsafe_allow_html=True,
        )
        chart_data = comparison.melt(
            id_vars="Model",
            value_vars=["CV F1", "Accuracy", "Weighted F1"],
            var_name="Metric",
            value_name="Score",
        )
        comparison_chart = px.bar(
            chart_data,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_map={
                "CV F1": "#958bf3",
                "Accuracy": "#24b99a",
                "Weighted F1": "#4059a8",
            },
        )
        comparison_chart.update_layout(
            height=335,
            margin=dict(l=0, r=5, t=15, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.13, x=0, title=None),
            xaxis=dict(title=None, tickangle=-12, showgrid=False),
            yaxis=dict(title="Score", range=[0, 1.08], gridcolor="#edf0f6", tickformat=".0%"),
            font=dict(family="DM Sans, sans-serif", color="#52617a"),
        )
        st.plotly_chart(comparison_chart, width="stretch", config={"displayModeBar": False})

        display_comparison = comparison.rename(
            columns={
                "Model": "Algorithm",
                "CV F1": "Cross-validation F1",
                "Accuracy": "Test accuracy",
                "Weighted F1": "Test weighted F1",
            }
        ).copy()
        for column in display_comparison.columns[1:]:
            display_comparison[column] = display_comparison[column].map(
                lambda value: f"{value:.1%}"
            )
        st.dataframe(display_comparison, hide_index=True, width="stretch")
        st.caption(
            f"Selected by 5-fold cross-validation F1: **{best_name}**. "
            "The displayed test scores use a separate, stratified holdout."
        )

    with st.expander("How well does the selected model perform?"):
        test_predictions = artifacts["best_test_predictions"]
        matrix = confusion_matrix(y_test, test_predictions)
        matrix_figure = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[str(name).title() for name in iris.target_names],
                y=[str(name).title() for name in iris.target_names],
                colorscale=[[0, "#f0efff"], [1, "#6657e8"]],
                text=matrix,
                texttemplate="%{text}",
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        matrix_figure.update_layout(
            title="Confusion matrix · held-out test data",
            xaxis_title="Predicted variety",
            yaxis_title="Actual variety",
            height=330,
            margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color="#52617a"),
        )
        st.plotly_chart(matrix_figure, width="stretch", config={"displayModeBar": False})

    st.markdown(
        '<div class="footer">Built with scikit-learn · Iris dataset · For learning and demonstration</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
