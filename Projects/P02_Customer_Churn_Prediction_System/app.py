from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "churn_pipeline.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "feature_metadata.json"
IMPORTANCE_PATH = PROJECT_DIR / "reports" / "feature_importance.csv"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; }
    .block-container { max-width: 1280px; padding-top: 2rem; }
    .hero {
      padding: 1.6rem 1.9rem; border-radius: 18px; color: white;
      background: linear-gradient(115deg, #102a43, #0b7a75);
      margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; }
    .hero p { margin: .45rem 0 0; color: #d6f1ed; }
    .small-note { color: #5c6b7a; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python train.py` to create the trained model.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError("Feature metadata is missing. Run `python train.py`.")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


try:
    bundle = load_model()
    metadata = load_metadata()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.markdown(
    """
    <div class="hero">
      <h1>Customer Churn Intelligence</h1>
      <p>Understand churn risk, prioritize retention, and explore the model behind each prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pipeline = bundle["pipeline"]
threshold = float(bundle["threshold"])
feature_info: dict[str, dict[str, Any]] = metadata["features"]
reference_values = metadata["reference_values"]

with st.sidebar:
    st.subheader("Decision settings")
    selected_threshold = st.slider(
        "Churn alert threshold",
        min_value=0.05,
        max_value=0.95,
        value=round(min(max(threshold, 0.05), 0.95), 2),
        step=0.01,
        help="Lower thresholds flag more customers, which may improve recall but also increase false positives.",
    )
    st.caption(
        f"Cross-validated default: **{threshold:.2f}** "
        f"(selected model: {bundle['model_name']})"
    )

prediction_tab, model_tab = st.tabs(["Customer assessment", "Model performance"])
with prediction_tab:
    st.subheader("Customer profile")
    st.caption(
        "Enter the customer's current service and account details. Customer IDs are not used by the model."
    )
    customer_values: dict[str, Any] = {}

    with st.form("customer_form"):
        columns = st.columns(2)
        for index, (feature, info) in enumerate(feature_info.items()):
            column = columns[index % 2]
            with column:
                if info["kind"] == "categorical":
                    options = info["options"]
                    default = info["default"]
                    customer_values[feature] = st.selectbox(
                        feature,
                        options,
                        index=options.index(default) if default in options else 0,
                    )
                elif info["options"]:
                    options = info["options"]
                    default = info["median"]
                    default_index = min(
                        range(len(options)), key=lambda idx: abs(options[idx] - default)
                    )
                    customer_values[feature] = st.selectbox(
                        feature, options, index=default_index
                    )
                else:
                    minimum, maximum = info["minimum"], info["maximum"]
                    value = min(max(info["median"], minimum), maximum)
                    customer_values[feature] = st.number_input(
                        feature,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(value),
                        step=1.0 if info["integer"] else 0.01,
                    )
        submitted = st.form_submit_button(
            "Assess churn risk", type="primary", width="stretch"
        )

    if submitted:
        customer = pd.DataFrame(
            [{column: customer_values[column] for column in bundle["feature_columns"]}]
        )
        probability = float(pipeline.predict_proba(customer)[0, 1])
        predicted_churn = probability >= selected_threshold

        st.divider()
        left, right = st.columns([1, 2])
        with left:
            st.metric("Churn probability", f"{probability:.1%}")
            st.metric("Decision threshold", f"{selected_threshold:.0%}")
        with right:
            if predicted_churn:
                st.error("Higher churn risk — consider a retention conversation.")
            else:
                st.success("Lower churn risk at the selected alert threshold.")
            st.progress(min(max(probability, 0.0), 1.0))

        deltas: list[dict[str, Any]] = []
        for feature in bundle["feature_columns"]:
            reference = reference_values[feature]
            if customer.iloc[0][feature] == reference:
                continue
            alternative = customer.copy()
            alternative.loc[0, feature] = reference
            alternative_probability = float(pipeline.predict_proba(alternative)[0, 1])
            deltas.append(
                {
                    "Feature": feature,
                    "Entered value": str(customer.iloc[0][feature]),
                    "Reference value": str(reference),
                    "Risk change vs. reference": probability - alternative_probability,
                }
            )
        st.markdown("#### Factors influencing this assessment")
        if deltas:
            local_factors = pd.DataFrame(deltas)
            local_factors["Absolute change"] = local_factors[
                "Risk change vs. reference"
            ].abs()
            local_factors = local_factors.sort_values(
                "Absolute change", ascending=False
            ).head(6)
            local_factors["Risk change vs. reference"] = local_factors[
                "Risk change vs. reference"
            ].map(lambda change: f"{change:+.1%}")
            st.dataframe(
                local_factors.drop(columns="Absolute change"),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Approximate one-feature-at-a-time sensitivity: each entered value is compared "
                "with its training-set median or most common value. These are associations, "
                "not causal effects."
            )
        else:
            st.info("All entered values match the training-set reference profile.")

with model_tab:
    st.subheader("Held-out evaluation")
    evaluation_path = PROJECT_DIR / "reports" / "evaluation.json"
    roc_path = PROJECT_DIR / "reports" / "roc_curve.png"
    if not evaluation_path.exists():
        st.info("Run `python evaluate.py` to generate the held-out evaluation report.")
    else:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        metric_columns = st.columns(5)
        for container, (label, key) in zip(
            metric_columns,
            [
                ("Accuracy", "accuracy"),
                ("Precision", "precision"),
                ("Churn recall", "recall"),
                ("F1-score", "f1"),
                ("ROC-AUC", "roc_auc"),
            ],
        ):
            container.metric(label, f"{evaluation[key]:.3f}")
        st.caption(
            f"Selected model: {evaluation['model']} · "
            f"threshold: {evaluation['threshold']:.3f} · "
            f"{evaluation['test_rows']:,} stratified holdout customers"
        )
        matrix = evaluation["confusion_matrix"]
        confusion = pd.DataFrame(
            matrix,
            index=["Actual no churn", "Actual churn"],
            columns=["Predicted no churn", "Predicted churn"],
        )
        chart_col, table_col = st.columns([1.2, 1])
        with chart_col:
            st.markdown("#### ROC curve")
            if roc_path.exists():
                st.image(str(roc_path), width="stretch")
        with table_col:
            st.markdown("#### Confusion matrix")
            st.dataframe(confusion, width="stretch")
            st.markdown("#### Classification report")
            report = pd.DataFrame(evaluation["classification_report"]).T
            st.dataframe(report.round(3), width="stretch")

    if IMPORTANCE_PATH.exists():
        st.markdown("#### Global feature importance")
        importance = pd.read_csv(IMPORTANCE_PATH).head(10).sort_values(
            "importance_mean"
        )
        st.bar_chart(
            importance,
            x="feature",
            y="importance_mean",
            horizontal=True,
            y_label="ROC-AUC decrease when shuffled",
        )
        st.caption(
            "Permutation importance is calculated on the held-out test set; larger values indicate "
            "a stronger contribution to the model's ranking performance."
        )

st.markdown(
    '<p class="small-note">Predictions support prioritization and should be reviewed alongside customer context and retention economics.</p>',
    unsafe_allow_html=True,
)
