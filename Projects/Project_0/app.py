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
    initial_sidebar_state="expanded",
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
    for pipeline in candidates.values():
        pipeline.fit(iris.data, iris.target)
    best_model = candidates[best_name]

    return {
        "iris": iris,
        "comparison": pd.DataFrame(comparison).sort_values(
            ["CV F1", "Weighted F1"], ascending=False
        ),
        "models": candidates,
        "best_name": best_name,
        "best_model": best_model,
        "y_test": y_test,
        "best_test_predictions": test_predictions_by_model[best_name],
    }


def apply_styles(theme: str) -> None:
    themes = {
        "Aurora": {
            "background": "#f5f7fb",
            "surface": "#ffffff",
            "ink": "#14213d",
            "muted": "#71809a",
            "line": "#e8edf5",
            "accent": "#6657e8",
            "hero": "linear-gradient(115deg, #171d3a 0%, #292c64 56%, #574bd1 100%)",
        },
        "Paper": {
            "background": "#f6f2eb",
            "surface": "#fffdf9",
            "ink": "#30291f",
            "muted": "#817565",
            "line": "#e9e1d5",
            "accent": "#b46a43",
            "hero": "linear-gradient(115deg, #34281f 0%, #654632 58%, #b46a43 100%)",
        },
        "Midnight": {
            "background": "#101522",
            "surface": "#1a2131",
            "ink": "#edf1fc",
            "muted": "#a1aec4",
            "line": "#303a4d",
            "accent": "#8f83ff",
            "hero": "linear-gradient(115deg, #12182a 0%, #202c4d 58%, #584bd0 100%)",
        },
    }
    palette = themes[theme]
    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root {
            --ink: __INK__; --muted: __MUTED__;
            --line: __LINE__; --accent: __ACCENT__;
            --surface: __SURFACE__;
        }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: __BACKGROUND__; color: var(--ink); }
        .block-container { max-width: 1440px; padding-top: 2.1rem; padding-bottom: 3rem; }
        header[data-testid="stHeader"] { background: transparent; }
        h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; color: var(--ink); }
        h1 { font-size: 2.65rem !important; font-weight: 800 !important; }
        h2 { font-size: 1.35rem !important; font-weight: 750 !important; }
        p, label, span { color: inherit; }
        .hero {
            position: relative; overflow: hidden; border-radius: 24px; padding: 2.15rem 2.35rem;
            background: __HERO__;
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
            background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.15rem;
            box-shadow: 0 5px 18px rgba(27,42,75,.025); min-height: 94px;
        }
        .stat-label { color: var(--muted); font-size: .78rem; font-weight: 600; }
        .stat-value { color: var(--ink); font: 800 1.5rem 'Manrope', sans-serif; margin-top: .3rem; }
        .section-caption { color: var(--muted); font-size: .88rem; margin-top: -.65rem; margin-bottom: 1rem; }
        div[data-testid="stSlider"] > label { font-weight: 600; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            background: var(--surface); color: var(--ink); border-color: var(--line);
        }
        div[data-baseweb="popover"] { color: var(--ink); }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
        .stButton > button { border-radius: 10px; font-weight: 700; }
        div[data-baseweb="tab-list"] { gap: .5rem; }
        button[data-baseweb="tab"] { border-radius: 10px 10px 0 0; }
        .stDownloadButton > button { border-radius: 10px; font-weight: 700; }
        .footer { text-align: center; color: #929db0; font-size: .76rem; padding-top: 1.8rem; }
        @media (max-width: 700px) {
            .block-container { padding: 1rem; }
            .hero { padding: 1.5rem; border-radius: 18px; }
            .hero-title { font-size: 1.8rem; }
        }
        </style>
        """
    for token, value in {
        "__INK__": palette["ink"],
        "__MUTED__": palette["muted"],
        "__LINE__": palette["line"],
        "__ACCENT__": palette["accent"],
        "__SURFACE__": palette["surface"],
        "__BACKGROUND__": palette["background"],
        "__HERO__": palette["hero"],
    }.items():
        css = css.replace(token, value)
    st.markdown(css, unsafe_allow_html=True)


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


def render_probability_chart(
    probabilities: np.ndarray, class_names: np.ndarray, theme: str
) -> None:
    probability_frame = pd.DataFrame(
        {
            "Variety": [str(name).title() for name in class_names],
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=True)
    figure = px.bar(
        probability_frame,
        x="Probability",
        y="Variety",
        orientation="h",
        text=probability_frame["Probability"].map(lambda value: f"{value:.0%}"),
        color="Probability",
        color_continuous_scale=["#c9c5ff", "#6657e8"],
        range_x=[0, 1],
    )
    figure.update_traces(textposition="outside", cliponaxis=False)
    text_color = "#dce3f2" if theme == "Midnight" else "#52617a"
    grid_color = "#384256" if theme == "Midnight" else "#edf0f6"
    figure.update_layout(
        height=220,
        margin=dict(l=0, r=30, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor=grid_color, tickformat=".0%", title=None),
        yaxis=dict(title=None, showgrid=False),
        font=dict(family="DM Sans, sans-serif", color=text_color),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def main() -> None:
    artifacts = train_models()
    iris = artifacts["iris"]
    comparison = artifacts["comparison"]
    best_name = artifacts["best_name"]
    y_test = artifacts["y_test"]

    with st.sidebar:
        st.markdown("### Studio settings")
        theme = st.selectbox(
            "Choose a layout",
            ["Aurora", "Paper", "Midnight"],
            help="Switch the visual style without changing your model or data.",
        )
        selected_model_name = st.selectbox(
            "Prediction model",
            list(artifacts["models"]),
            index=list(artifacts["models"]).index(best_name),
            help="Choose which trained classifier powers predictions.",
        )
        st.caption("Model scores use a separate, stratified test split.")

    apply_styles(theme)
    model = artifacts["models"][selected_model_name]
    render_hero(best_name)

    best_row = comparison.loc[comparison["Model"] == best_name].iloc[0]
    metric_columns = st.columns(4)
    for column, (label, value) in zip(
        metric_columns,
        [
            ("Flowers in dataset", f"{len(iris.data):,}"),
            ("Flower varieties", str(len(iris.target_names))),
            ("Recommended model", best_name),
            ("Held-out accuracy", f"{best_row['Accuracy']:.0%}"),
        ],
    ):
        with column:
            render_metric(label, value)

    prediction_tab, explore_tab, models_tab = st.tabs(
        ["Make a prediction", "Explore the data", "Model studio"]
    )

    with prediction_tab:
        st.markdown("## Your prediction workspace")
        st.markdown(
            '<div class="section-caption">Enter one flower or upload a CSV to score a whole batch.</div>',
            unsafe_allow_html=True,
        )
        input_mode = st.radio(
            "Input type",
            ["One flower", "Batch CSV"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if input_mode == "One flower":
            presets = {
                "Custom measurements": np.median(iris.data, axis=0),
                "Setosa example": iris.data[0],
                "Versicolor example": iris.data[50],
                "Virginica example": iris.data[100],
            }

            def load_preset() -> None:
                values = presets[st.session_state["sample_preset"]]
                for feature_index, value in enumerate(values):
                    st.session_state[f"measurement_{feature_index}"] = float(value)

            st.selectbox(
                "Start from an example",
                list(presets),
                key="sample_preset",
                on_change=load_preset,
            )
            with st.form("single_prediction_form"):
                for row_start in range(0, len(iris.feature_names), 2):
                    measurement_columns = st.columns(2)
                    for index in range(
                        row_start, min(row_start + 2, len(iris.feature_names))
                    ):
                        feature_name = iris.feature_names[index]
                        if f"measurement_{index}" not in st.session_state:
                            st.session_state[f"measurement_{index}"] = float(
                                np.median(iris.data[:, index])
                            )
                        with measurement_columns[index - row_start]:
                            st.number_input(
                                feature_name.replace(" (cm)", "").title() + " (cm)",
                                min_value=0.0,
                                step=0.1,
                                format="%.1f",
                                key=f"measurement_{index}",
                                help=(
                                    f"Training data range: "
                                    f"{iris.data[:, index].min():.1f}–"
                                    f"{iris.data[:, index].max():.1f} cm. "
                                    "Values outside this range are accepted."
                                ),
                            )
                submitted = st.form_submit_button(
                    "Predict variety", type="primary", width="stretch"
                )
            if submitted:
                measurements = [
                    float(st.session_state[f"measurement_{index}"])
                    for index in range(len(iris.feature_names))
                ]
                st.session_state["prediction_result"] = {
                    "measurements": measurements,
                    "model": selected_model_name,
                }

            saved_prediction = st.session_state.get("prediction_result")
            if saved_prediction and saved_prediction["model"] == selected_model_name:
                sample = np.array([saved_prediction["measurements"]])
                predicted_class = int(model.predict(sample)[0])
                probabilities = model.predict_proba(sample)[0]
                predicted_name = str(iris.target_names[predicted_class]).title()
                confidence = float(probabilities[predicted_class])
                st.markdown(
                    f"""
                    <div class="panel">
                      <div class="panel-kicker">Predicted variety · {selected_model_name}</div>
                      <div class="prediction-name">{predicted_name}</div>
                      <div class="prediction-note">Estimated model confidence: {confidence:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_probability_chart(probabilities, iris.target_names, theme)
                report = pd.DataFrame(
                    [saved_prediction["measurements"]],
                    columns=iris.feature_names,
                )
                report["Predicted variety"] = predicted_name
                report["Confidence"] = confidence
                for class_index, class_name in enumerate(iris.target_names):
                    report[f"{str(class_name).title()} probability"] = probabilities[
                        class_index
                    ]
                st.download_button(
                    "Download prediction",
                    report.to_csv(index=False).encode("utf-8"),
                    "iris_prediction.csv",
                    "text/csv",
                )
        else:
            st.markdown(
                "Upload a CSV with one row per flower and these four measurement columns:"
            )
            st.code(", ".join(iris.feature_names), language=None)
            template = pd.DataFrame(columns=iris.feature_names).to_csv(index=False)
            st.download_button(
                "Download CSV template",
                template,
                "iris_input_template.csv",
                "text/csv",
            )
            uploaded_file = st.file_uploader(
                "Choose a measurement CSV",
                type=["csv"],
                help="CSV must include all four measurement columns and contain at most 10,000 rows.",
            )
            if uploaded_file is not None:
                try:
                    batch = pd.read_csv(uploaded_file)
                except (
                    UnicodeDecodeError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                ) as error:
                    st.error(f"Could not read this CSV: {error}")
                else:
                    missing_columns = [
                        feature for feature in iris.feature_names if feature not in batch
                    ]
                    if batch.empty:
                        st.error("The uploaded CSV has no data rows.")
                    elif len(batch) > 10_000:
                        st.error("Please upload no more than 10,000 rows at a time.")
                    elif missing_columns:
                        st.error(
                            "Missing required columns: " + ", ".join(missing_columns)
                        )
                    else:
                        numeric = batch.loc[:, iris.feature_names].apply(
                            pd.to_numeric, errors="coerce"
                        )
                        invalid = numeric.isna().any(axis=1) | ~np.isfinite(
                            numeric.to_numpy(dtype=float)
                        ).all(axis=1)
                        if invalid.any():
                            rows = (np.flatnonzero(invalid.to_numpy())[:10] + 2).tolist()
                            st.error(
                                "Measurements must be finite numbers. "
                                f"Check CSV row(s): {rows}."
                            )
                        elif (numeric < 0).any().any():
                            st.error("Measurements must be zero or greater.")
                        else:
                            predictions = model.predict(numeric.to_numpy())
                            probabilities = model.predict_proba(numeric.to_numpy())
                            results = batch.copy()
                            results["Predicted variety"] = [
                                str(iris.target_names[index]).title()
                                for index in predictions
                            ]
                            results["Confidence"] = probabilities.max(axis=1)
                            for class_index, class_name in enumerate(iris.target_names):
                                results[f"{str(class_name).title()} probability"] = (
                                    probabilities[:, class_index]
                                )
                            st.success(f"Scored {len(results):,} flower(s).")
                            st.dataframe(results, hide_index=True, width="stretch")
                            st.download_button(
                                "Download predictions",
                                results.to_csv(index=False).encode("utf-8"),
                                "iris_predictions.csv",
                                "text/csv",
                                type="primary",
                            )

    with explore_tab:
        st.markdown("## Get to know the Iris dataset")
        feature_options = list(iris.feature_names)
        chart_columns = st.columns(2)
        with chart_columns[0]:
            x_feature = st.selectbox("Horizontal axis", feature_options, index=0)
        with chart_columns[1]:
            y_feature = st.selectbox("Vertical axis", feature_options, index=2)
        iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_frame["Variety"] = [
            str(iris.target_names[index]).title() for index in iris.target
        ]
        scatter = px.scatter(
            iris_frame,
            x=x_feature,
            y=y_feature,
            color="Variety",
            color_discrete_sequence=["#6657e8", "#18a98b", "#e59c52"],
            hover_data=feature_options,
        )
        scatter.update_layout(
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Variety",
            margin=dict(l=10, r=10, t=20, b=10),
            font=dict(
                family="DM Sans, sans-serif",
                color="#dce3f2" if theme == "Midnight" else "#52617a",
            ),
        )
        scatter.update_xaxes(gridcolor="#384256" if theme == "Midnight" else "#edf0f6")
        scatter.update_yaxes(gridcolor="#384256" if theme == "Midnight" else "#edf0f6")
        st.plotly_chart(scatter, width="stretch", config={"displayModeBar": False})
        st.markdown("### Measurement summary")
        st.dataframe(
            iris_frame.groupby("Variety")[feature_options].agg(["mean", "min", "max"]).round(2),
            width="stretch",
        )

    with models_tab:
        st.markdown("## Compare the classifiers")
        st.markdown(
            '<div class="section-caption">Five models evaluated on identical, stratified data splits.</div>',
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
            height=380,
            margin=dict(l=0, r=5, t=15, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.13, x=0, title=None),
            xaxis=dict(title=None, tickangle=-12, showgrid=False),
            yaxis=dict(
                title="Score",
                range=[0, 1.08],
                gridcolor="#384256" if theme == "Midnight" else "#edf0f6",
                tickformat=".0%",
            ),
            font=dict(
                family="DM Sans, sans-serif",
                color="#dce3f2" if theme == "Midnight" else "#52617a",
            ),
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
            f"Recommended by 5-fold cross-validation F1: **{best_name}**. "
            "Predictions use the model selected in the sidebar."
        )
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
            font=dict(
                family="DM Sans, sans-serif",
                color="#dce3f2" if theme == "Midnight" else "#52617a",
            ),
        )
        st.plotly_chart(matrix_figure, width="stretch", config={"displayModeBar": False})

    st.markdown(
        '<div class="footer">Built with scikit-learn · Iris dataset · For learning and demonstration</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
