
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG — matches 02_modeling.ipynb / 03_xai.ipynb save locations
# ---------------------------------------------------------------------------
# Resolved relative to this file, not the process's working directory —
# `../models/...` broke on Streamlit Cloud, which runs the app with the
# working directory set to the repo root rather than the app/ folder.
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "churn_model.pkl"
CONFIG_PATH = REPO_ROOT / "models" / "model_config.json"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 1. FEATURE SCHEMA — identical list/order to 02_modeling.ipynb & 03_xai.ipynb
# ---------------------------------------------------------------------------
FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges", "CLTV",
]
NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
CATEGORICAL_FEATURES = [f for f in FEATURES if f not in NUMERIC_FEATURES]

CATEGORICAL_OPTIONS = {
    "Gender": ["Male", "Female"],
    "Senior Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "Phone Service": ["Yes", "No"],
    "Multiple Lines": ["No", "Yes", "No phone service"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["Yes", "No", "No internet service"],
    "Online Backup": ["Yes", "No", "No internet service"],
    "Device Protection": ["No", "Yes", "No internet service"],
    "Tech Support": ["No", "Yes", "No internet service"],
    "Streaming TV": ["No", "Yes", "No internet service"],
    "Streaming Movies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["Yes", "No"],
    "Payment Method": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}
NUMERIC_SLIDERS = {
    "Tenure Months": (0, 72, 12),
    "Monthly Charges": (0.0, 200.0, 70.0),
    "Total Charges": (0.0, 10000.0, 1000.0),
    "CLTV": (0, 7000, 4000),
}

# ---------------------------------------------------------------------------
# Example customers — the same three (A/B/C) used throughout
# notebooks/03_xai.ipynb Sections 10-11, real held-out customers with
# verified predictions, not invented data. Lets a reviewer see a real
# high-risk / low-risk / borderline case in one click instead of manually
# filling in 16 fields.
# ---------------------------------------------------------------------------
EXAMPLE_CUSTOMERS = {
    "Example A — High risk (5178-LMXOP, 93.5%)": {
        "Gender": "Male", "Senior Citizen": "Yes", "Partner": "Yes", "Dependents": "No",
        "Tenure Months": 1, "Phone Service": "Yes", "Multiple Lines": "Yes",
        "Internet Service": "Fiber optic", "Online Security": "No", "Online Backup": "No",
        "Device Protection": "No", "Tech Support": "No", "Streaming TV": "Yes",
        "Streaming Movies": "Yes", "Contract": "Month-to-month", "Paperless Billing": "Yes",
        "Payment Method": "Electronic check", "Monthly Charges": 95.1,
        "Total Charges": 95.1, "CLTV": 5795,
    },
    "Example B — Low risk (0794-YVSGE, 0.2%)": {
        "Gender": "Male", "Senior Citizen": "No", "Partner": "Yes", "Dependents": "Yes",
        "Tenure Months": 72, "Phone Service": "Yes", "Multiple Lines": "No",
        "Internet Service": "No", "Online Security": "No internet service",
        "Online Backup": "No internet service", "Device Protection": "No internet service",
        "Tech Support": "No internet service", "Streaming TV": "No internet service",
        "Streaming Movies": "No internet service", "Contract": "Two year",
        "Paperless Billing": "No", "Payment Method": "Bank transfer (automatic)",
        "Monthly Charges": 20.3, "Total Charges": 1401.15, "CLTV": 5265,
    },
    "Example C — Borderline (9812-GHVRI, 30.0%)": {
        "Gender": "Female", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
        "Tenure Months": 40, "Phone Service": "Yes", "Multiple Lines": "Yes",
        "Internet Service": "Fiber optic", "Online Security": "No", "Online Backup": "No",
        "Device Protection": "No", "Tech Support": "No", "Streaming TV": "No",
        "Streaming Movies": "Yes", "Contract": "Month-to-month", "Paperless Billing": "No",
        "Payment Method": "Bank transfer (automatic)", "Monthly Charges": 83.85,
        "Total Charges": 3532.25, "CLTV": 4294,
    },
}


def _apply_example_customer():
    label = st.session_state["example_picker"]
    if label in EXAMPLE_CUSTOMERS:
        for feature, value in EXAMPLE_CUSTOMERS[label].items():
            st.session_state[f"field__{feature}"] = value

# ---------------------------------------------------------------------------
# 2. LOAD MODEL + CONFIG (Gracy's artifacts)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_config():
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        return None, None
    pipeline = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return pipeline, config


@st.cache_resource
def build_shap_explainer(_pipeline):
    xgb_step = _pipeline.named_steps["model"]
    return shap.TreeExplainer(xgb_step)


@st.cache_resource
def build_processed_to_original(_pipeline):
    """
    Maps each of the 47 processed SHAP columns back to (raw_feature, category),
    built from the fitted OneHotEncoder's own categories_ — same approach as
    03_xai.ipynb Section 6 — rather than parsing generated column-name strings.
    """
    preprocessor_step = _pipeline.named_steps["preprocessor"]
    numeric_features = list(preprocessor_step.transformers_[0][2])
    categorical_features = list(preprocessor_step.transformers_[1][2])
    onehot_categories = preprocessor_step.named_transformers_["cat"].categories_

    processed_to_original = [(f, None) for f in numeric_features]
    for feature_name, categories in zip(categorical_features, onehot_categories):
        for category in categories:
            processed_to_original.append((feature_name, category))
    return processed_to_original


churn_pipeline, model_config = load_model_and_config()

if churn_pipeline is None:
    st.error(
        f"Model artifacts not found.\n\n"
        f"Expected:\n- `{MODEL_PATH}`\n- `{CONFIG_PATH}`\n\n"
        f"Run `02_modeling.ipynb` (or the regeneration cell in `03_xai.ipynb`, "
        f"Section 1a) to produce `churn_model.pkl`, then place both files at "
        f"the paths above — or edit MODEL_PATH / CONFIG_PATH at the top of "
        f"app.py to match where you stored them."
    )
    st.stop()

CHURN_THRESHOLD = model_config["threshold"]
explainer = build_shap_explainer(churn_pipeline)
processed_to_original = build_processed_to_original(churn_pipeline)
preprocessor_step = churn_pipeline.named_steps["preprocessor"]

# ---------------------------------------------------------------------------
# 3. PREDICTION (03_xai.ipynb Section 3, predict_customer)
# ---------------------------------------------------------------------------
def predict_customer(customer_features: pd.DataFrame) -> dict:
    probability = churn_pipeline.predict_proba(customer_features)[:, 1][0]
    is_churn_risk = int(probability >= CHURN_THRESHOLD)
    return {
        "churn_probability": float(probability),
        "churn_prediction": is_churn_risk,
    }

# ---------------------------------------------------------------------------
# 4. LOCAL SHAP AGGREGATION (03_xai.ipynb Section 6, get_top_shap_features)
#    — adapted to run on a single freshly-submitted row instead of an index
#      into the held-out X_test set.
# ---------------------------------------------------------------------------
def get_top_shap_features_for_customer(customer_features: pd.DataFrame, top_n=None) -> pd.DataFrame:
    customer_processed = preprocessor_step.transform(customer_features)
    if hasattr(customer_processed, "toarray"):
        customer_processed = customer_processed.toarray()

    shap_row = explainer.shap_values(customer_processed)
    if isinstance(shap_row, list):
        shap_row = shap_row[0]
    shap_row = shap_row[0] if shap_row.ndim > 1 else shap_row

    raw_row = customer_features.iloc[0]

    contributions = {}
    for shap_value, (feature, _category) in zip(shap_row, processed_to_original):
        contributions[feature] = contributions.get(feature, 0.0) + shap_value

    records = [
        {
            "feature": feature,
            "customer_value": raw_row[feature],
            "shap_value": shap_value,
            "direction": "increases_risk" if shap_value > 0 else "decreases_risk",
        }
        for feature, shap_value in contributions.items()
    ]

    result = pd.DataFrame(records)
    result["abs_shap_value"] = result["shap_value"].abs()
    result = result.sort_values("abs_shap_value", ascending=False).drop(columns="abs_shap_value")
    return result.head(top_n) if top_n else result.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 5. HUMAN-READABLE EXPLANATIONS (03_xai.ipynb Section 7, verbatim)
# ---------------------------------------------------------------------------
YES_NO_FEATURES = {
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Multiple Lines", "Phone Service",
    "Paperless Billing", "Senior Citizen", "Dependents",
}
YES_NO_FEATURES_WITH_ARTICLE = {"Partner"}


def generate_business_explanation(feature: str, customer_value, shap_value: float) -> str:
    risk_word = "higher" if shap_value > 0 else "lower"

    if feature in YES_NO_FEATURES_WITH_ARTICLE and customer_value in ("Yes", "No"):
        has_clause = "having a" if customer_value == "Yes" else "not having a"
        return (f"The customer {has_clause} {feature.lower()} is contributing "
                f"to a {risk_word} predicted churn risk.")

    if feature in YES_NO_FEATURES and customer_value in ("Yes", "No"):
        has_clause = "having" if customer_value == "Yes" else "not having"
        return (f"The customer {has_clause} {feature.lower()} is contributing "
                f"to a {risk_word} predicted churn risk.")

    if feature == "Contract":
        return (f"The customer's {str(customer_value).lower()} contract is "
                f"contributing to a {risk_word} predicted churn risk.")

    if feature == "Tenure Months":
        unit = "month" if customer_value == 1 else "months"
        return (f"The customer's tenure of {customer_value} {unit} is "
                f"contributing to a {risk_word} predicted churn risk.")

    if feature == "Monthly Charges":
        return (f"The customer's monthly charge of ${float(customer_value):.2f} is "
                f"contributing to a {risk_word} predicted churn risk.")

    if feature == "Total Charges":
        return (f"The customer's total charges to date (${float(customer_value):.2f}) are "
                f"contributing to a {risk_word} predicted churn risk.")

    if feature == "CLTV":
        return (f"The customer's customer lifetime value score ({customer_value}) is "
                f"contributing to a {risk_word} predicted churn risk.")

    return (f"The customer's {feature.lower()} ('{customer_value}') is "
            f"contributing to a {risk_word} predicted churn risk.")

# ---------------------------------------------------------------------------
# 6. ACTIONABLE-DRIVER CLASSIFICATION (03_xai.ipynb Section 8, verbatim)
# ---------------------------------------------------------------------------
NON_ACTIONABLE_FEATURES = {"Gender", "Senior Citizen", "Partner", "Dependents", "CLTV"}
ACTIONABLE_FEATURES = set(FEATURES) - NON_ACTIONABLE_FEATURES

# ---------------------------------------------------------------------------
# 7. RETENTION RECOMMENDATION RULES (03_xai.ipynb Section 9, verbatim)
# ---------------------------------------------------------------------------
RECOMMENDATION_RULES = {
    "Contract": lambda v: (
        "Offer an incentive (discount or perk) to move from month-to-month "
        "to an annual or two-year contract."
        if v == "Month-to-month" else
        f"Review the customer's current contract ('{v}') ahead of renewal."
    ),
    "Online Security": lambda v: (
        "Offer an online security add-on/bundle."
        if v == "No" else f"Review the customer's online security plan (currently '{v}')."
    ),
    "Online Backup": lambda v: (
        "Offer an online backup add-on/bundle."
        if v == "No" else f"Review the customer's online backup plan (currently '{v}')."
    ),
    "Device Protection": lambda v: (
        "Offer a device protection add-on/bundle."
        if v == "No" else f"Review the customer's device protection plan (currently '{v}')."
    ),
    "Tech Support": lambda v: (
        "Offer a tech support package."
        if v == "No" else f"Review the customer's tech support plan (currently '{v}')."
    ),
    "Multiple Lines": lambda v: (
        "Offer a multiple-lines bundle discount."
        if v == "No" else f"Review the customer's lines plan (currently '{v}')."
    ),
    "Monthly Charges": lambda v: (
        f"Review the customer's plan/pricing (current monthly charge ${float(v):.2f}) "
        "or offer a personalized pricing/discount review."
    ),
    "Total Charges": lambda v: (
        "Review the customer's overall billing history for a loyalty or pricing offer."
    ),
    "Tenure Months": lambda v: (
        "Enroll the customer in an onboarding / early-engagement retention program."
        if float(v) <= 12 else
        "Schedule a loyalty check-in given the customer's tenure."
    ),
    "Internet Service": lambda v: (
        f"Review the customer's internet plan ('{v}') for a better-fit offer or service quality check."
    ),
    "Payment Method": lambda v: (
        "Encourage a switch to automatic payment (credit card / bank transfer) for reliability."
        if v == "Electronic check" else
        f"Review the customer's payment method (currently '{v}')."
    ),
    "Paperless Billing": lambda v: (
        "Confirm the customer is comfortable with paperless billing and offer support if needed."
    ),
    "Phone Service": lambda v: (
        "Review the customer's phone service plan for a better-fit bundle."
    ),
    "Streaming TV": lambda v: (
        "Review the customer's streaming TV add-on and bundle pricing."
    ),
    "Streaming Movies": lambda v: (
        "Review the customer's streaming movies add-on and bundle pricing."
    ),
}


def generate_recommendations(top_features: pd.DataFrame, top_n=5) -> list:
    actionable = top_features[top_features["feature"].isin(ACTIONABLE_FEATURES)]
    increasing = actionable[actionable["shap_value"] > 0].head(top_n)

    recommendations = []
    for row in increasing.itertuples():
        rule = RECOMMENDATION_RULES.get(row.feature)
        if rule is not None:
            action = rule(row.customer_value)
        else:
            action = (f"Flag '{row.feature}' (current value '{row.customer_value}') "
                      "for personalized retention review with the customer.")
        recommendations.append({
            "driver": row.feature,
            "customer_value": row.customer_value,
            "shap_value": row.shap_value,
            "action": action,
        })
    return recommendations

# ---------------------------------------------------------------------------
# 8. UI — INPUT FORM
# ---------------------------------------------------------------------------
st.title("📉 Customer Churn Prediction & Retention Dashboard")
st.caption("Model: Gracy (XGBoost) · Explanations & recommendations: Gurnoor (SHAP) · App: Ujjwal")

tab_single, tab_batch = st.tabs(["🧍 Single Customer", "📁 Batch Upload"])

with tab_single:
    st.selectbox(
        "Load a real example customer, or fill in the form manually below",
        ["Custom (fill in manually)"] + list(EXAMPLE_CUSTOMERS.keys()),
        key="example_picker",
        on_change=_apply_example_customer,
    )

    with st.form("customer_form"):
        st.subheader("Customer Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["Gender"], key="field__Gender")
            senior = st.selectbox("Senior Citizen", CATEGORICAL_OPTIONS["Senior Citizen"], key="field__Senior Citizen")
            partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"], key="field__Partner")
            dependents = st.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"], key="field__Dependents")
            st.session_state.setdefault("field__Tenure Months", NUMERIC_SLIDERS["Tenure Months"][2])
            tenure = st.slider(
                "Tenure (Months)", NUMERIC_SLIDERS["Tenure Months"][0], NUMERIC_SLIDERS["Tenure Months"][1],
                key="field__Tenure Months",
            )

        with col2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["Phone Service"], key="field__Phone Service")
            multiple_lines = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["Multiple Lines"], key="field__Multiple Lines")
            internet = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["Internet Service"], key="field__Internet Service")
            online_security = st.selectbox("Online Security", CATEGORICAL_OPTIONS["Online Security"], key="field__Online Security")
            online_backup = st.selectbox("Online Backup", CATEGORICAL_OPTIONS["Online Backup"], key="field__Online Backup")
            device_protection = st.selectbox("Device Protection", CATEGORICAL_OPTIONS["Device Protection"], key="field__Device Protection")
            tech_support = st.selectbox("Tech Support", CATEGORICAL_OPTIONS["Tech Support"], key="field__Tech Support")
            streaming_tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["Streaming TV"], key="field__Streaming TV")
            streaming_movies = st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["Streaming Movies"], key="field__Streaming Movies")

        with col3:
            st.markdown("**Account & Billing**")
            contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"], key="field__Contract")
            paperless = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["Paperless Billing"], key="field__Paperless Billing")
            payment_method = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["Payment Method"], key="field__Payment Method")
            st.session_state.setdefault("field__Monthly Charges", NUMERIC_SLIDERS["Monthly Charges"][2])
            monthly_charges = st.slider(
                "Monthly Charges ($)", NUMERIC_SLIDERS["Monthly Charges"][0], NUMERIC_SLIDERS["Monthly Charges"][1],
                key="field__Monthly Charges",
            )
            st.session_state.setdefault("field__Total Charges", NUMERIC_SLIDERS["Total Charges"][2])
            total_charges = st.slider(
                "Total Charges ($)", NUMERIC_SLIDERS["Total Charges"][0], NUMERIC_SLIDERS["Total Charges"][1],
                key="field__Total Charges",
            )
            st.session_state.setdefault("field__CLTV", NUMERIC_SLIDERS["CLTV"][2])
            cltv = st.slider(
                "CLTV", NUMERIC_SLIDERS["CLTV"][0], NUMERIC_SLIDERS["CLTV"][1],
                key="field__CLTV",
            )

        submitted = st.form_submit_button("Predict Churn Risk", width="stretch")

    # -----------------------------------------------------------------------
    # 9. RUN PREDICTION -> SHAP -> EXPLANATIONS -> RECOMMENDATIONS -> DISPLAY
    # -----------------------------------------------------------------------
    if submitted:
        try:
            customer = pd.DataFrame([{
                "Gender": gender, "Senior Citizen": senior, "Partner": partner,
                "Dependents": dependents, "Tenure Months": tenure, "Phone Service": phone,
                "Multiple Lines": multiple_lines, "Internet Service": internet,
                "Online Security": online_security, "Online Backup": online_backup,
                "Device Protection": device_protection, "Tech Support": tech_support,
                "Streaming TV": streaming_tv, "Streaming Movies": streaming_movies,
                "Contract": contract, "Paperless Billing": paperless,
                "Payment Method": payment_method, "Monthly Charges": monthly_charges,
                "Total Charges": total_charges, "CLTV": cltv,
            }])[FEATURES]

            prediction = predict_customer(customer)
            prob = prediction["churn_probability"]
            is_risk = prediction["churn_prediction"]
            risk_color = "#e6533c" if is_risk else "#3cb371"
            risk_label = "High churn risk" if is_risk else "Not classified as churn-risk"

            st.divider()
            st.subheader("Prediction Results")
            with st.container(border=True):
                r1, r2 = st.columns([1, 2])
                with r1:
                    st.markdown(
                        f"""
                        <div style="text-align:center;">
                          <div style="font-size:2.6rem; font-weight:700; color:{risk_color};">
                            {prob:.1%}
                          </div>
                          <div style="color:#9aa0a6; font-size:0.9rem;">predicted churn probability</div>
                          <div style="margin-top:0.6rem; display:inline-block; padding:0.25rem 0.9rem;
                                      border-radius:999px; background:{risk_color}22; color:{risk_color};
                                      font-weight:600; border:1px solid {risk_color}66;">
                            {'🔴' if is_risk else '🟢'} {risk_label}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.caption(f"Decision threshold: {CHURN_THRESHOLD:.2f} (project-selected, not the default 0.50)")
                    st.progress(min(max(prob, 0.0), 1.0))
                    st.caption(
                        "Probability is the model's raw output; the threshold is a separate, "
                        "fixed business decision on top of it — not something this app changes."
                    )

            top_features = get_top_shap_features_for_customer(customer, top_n=None)
            increasing = top_features[top_features["shap_value"] > 0].head(5)
            decreasing = top_features[top_features["shap_value"] < 0].sort_values("shap_value").head(5)

            st.subheader("Why this prediction? (SHAP explanation)")
            with st.container(border=True):
                top5 = top_features.reindex(top_features["shap_value"].abs().sort_values(ascending=False).index).head(5)

                with plt.style.context("dark_background"):
                    fig, ax = plt.subplots(figsize=(7, 3))
                    fig.patch.set_facecolor("#181c24")
                    ax.set_facecolor("#181c24")
                    colors = ["#e6533c" if v > 0 else "#3cb371" for v in top5["shap_value"][::-1]]
                    ax.barh(top5["feature"][::-1], top5["shap_value"][::-1], color=colors)
                    ax.set_xlabel("Impact on churn probability (SHAP value)", color="#e6e6e6")
                    ax.set_title("Top factors for this customer", color="#e6e6e6")
                    ax.tick_params(colors="#e6e6e6")
                    for spine in ax.spines.values():
                        spine.set_color("#3a3f4b")
                    ax.axvline(0, color="#3a3f4b", linewidth=1)
                    fig.tight_layout()
                st.pyplot(fig, width="stretch")
                st.caption("🔴 Red = pushes prediction toward churn · 🟢 Green = pushes toward staying")

                col_up, col_down = st.columns(2)
                with col_up:
                    st.markdown("**Why risk is higher**")
                    if len(increasing) == 0:
                        st.caption("No risk-increasing factors in the top drivers for this customer.")
                    for row in increasing.itertuples():
                        st.write("• " + generate_business_explanation(row.feature, row.customer_value, row.shap_value))
                with col_down:
                    st.markdown("**Why risk is lower**")
                    if len(decreasing) == 0:
                        st.caption("No risk-reducing factors in the top drivers for this customer.")
                    for row in decreasing.itertuples():
                        st.write("• " + generate_business_explanation(row.feature, row.customer_value, row.shap_value))

            st.subheader("Recommended Retention Actions")
            with st.container(border=True):
                recommendations = generate_recommendations(top_features, top_n=5)
                if recommendations:
                    st.caption(
                        "Rule-based and traceable — each action maps directly to one of this "
                        "customer's actionable, risk-increasing SHAP drivers, not something learned by the model."
                    )
                    for rec in recommendations:
                        st.markdown(f"- **{rec['driver']}** (`{rec['customer_value']}`) → {rec['action']}")
                else:
                    st.info("No actionable risk-increasing drivers identified for this customer.")

        except Exception as e:
            st.error(f"Something went wrong while generating the prediction: {e}")

# ---------------------------------------------------------------------------
# 10. BATCH UPLOAD — run the same pipeline across an entire uploaded dataset
# ---------------------------------------------------------------------------
MAX_BATCH_ROWS = 20000


def _aggregate_shap_batch(shap_values_batch: np.ndarray) -> np.ndarray:
    """Sum the 47 processed SHAP columns back into the 20 raw features, for
    every row at once. Same aggregation as get_top_shap_features_for_customer
    (one-hot columns of a raw feature summed together), vectorized across
    rows instead of looped per customer.
    """
    feature_index = {f: i for i, f in enumerate(FEATURES)}
    col_to_feature_idx = [feature_index[f] for f, _category in processed_to_original]

    n_rows = shap_values_batch.shape[0]
    aggregated = np.zeros((n_rows, len(FEATURES)))
    for col_idx, feature_idx in enumerate(col_to_feature_idx):
        aggregated[:, feature_idx] += shap_values_batch[:, col_idx]
    return aggregated


def _find_id_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.strip().lower().replace("_", "").replace(" ", "") in ("customerid", "id"):
            return col
    return None


with tab_batch:
    st.subheader("Upload a dataset")
    st.caption(
        "Upload a CSV or Excel file with one row per customer. It can be the raw "
        "Telco dataset (extra columns like CustomerID, Country, Churn Label are fine "
        "and ignored) or just the 20 modeling columns — as long as all 20 are present "
        "with these exact names: " + ", ".join(f"`{f}`" for f in FEATURES)
    )

    uploaded_file = st.file_uploader("Dataset file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read this file: {e}")
            raw_df = None

        if raw_df is not None:
            missing = [f for f in FEATURES if f not in raw_df.columns]
            if missing:
                st.error(
                    "This file is missing required column(s): " + ", ".join(f"`{c}`" for c in missing) +
                    ". Column names must match exactly (same as 02_modeling.ipynb / 03_xai.ipynb)."
                )
            elif len(raw_df) == 0:
                st.error("This file has no rows.")
            elif len(raw_df) > MAX_BATCH_ROWS:
                st.error(
                    f"This file has {len(raw_df):,} rows, which is above the {MAX_BATCH_ROWS:,}-row "
                    "limit for a single batch run on this deployment. Please split it into smaller files."
                )
            else:
                st.success(f"Loaded {len(raw_df):,} customers.")
                st.dataframe(raw_df.head(10), width="stretch")

                if st.button("Run Batch Prediction", width="stretch"):
                    with st.spinner(f"Running predictions and SHAP analysis on {len(raw_df):,} customers…"):
                        batch_df = raw_df.copy()
                        # Same coercion as 02_modeling.ipynb / 03_xai.ipynb — Total Charges
                        # arrives as text in the raw Telco file, with a few blank values.
                        batch_df["Total Charges"] = pd.to_numeric(batch_df["Total Charges"], errors="coerce")
                        n_missing_total_charges = int(batch_df["Total Charges"].isna().sum())

                        X_batch = batch_df[FEATURES]
                        probabilities = churn_pipeline.predict_proba(X_batch)[:, 1]
                        predictions = (probabilities >= CHURN_THRESHOLD).astype(int)

                        processed_batch = preprocessor_step.transform(X_batch)
                        if hasattr(processed_batch, "toarray"):
                            processed_batch = processed_batch.toarray()

                        shap_values_batch = explainer.shap_values(processed_batch)
                        if isinstance(shap_values_batch, list):
                            shap_values_batch = shap_values_batch[0]

                        aggregated = _aggregate_shap_batch(shap_values_batch)

                        id_column = _find_id_column(raw_df)

                        top_drivers_col = []
                        top_actions_col = []
                        for i in range(len(batch_df)):
                            row_shap = aggregated[i]
                            row_values = X_batch.iloc[i]
                            order = np.argsort(-np.abs(row_shap))[:3]

                            row_top_features = pd.DataFrame([
                                {
                                    "feature": FEATURES[j],
                                    "customer_value": row_values[FEATURES[j]],
                                    "shap_value": row_shap[j],
                                    "direction": "increases_risk" if row_shap[j] > 0 else "decreases_risk",
                                }
                                for j in order
                            ])

                            driver_strs = [
                                f"{r.feature} ({'↑' if r.shap_value > 0 else '↓'})"
                                for r in row_top_features.itertuples()
                            ]
                            top_drivers_col.append(", ".join(driver_strs))

                            row_recs = generate_recommendations(row_top_features, top_n=2)
                            top_actions_col.append(
                                " | ".join(r["action"] for r in row_recs) if row_recs else "—"
                            )

                        results_df = pd.DataFrame({
                            "CustomerID": raw_df[id_column].values if id_column else np.arange(1, len(batch_df) + 1),
                            "Churn Probability": np.round(probabilities, 4),
                            "Risk Level": np.where(predictions == 1, "High churn risk", "Not classified as churn-risk"),
                            "Top Drivers": top_drivers_col,
                            "Recommended Actions": top_actions_col,
                        })

                    if n_missing_total_charges:
                        st.warning(
                            f"{n_missing_total_charges} row(s) had a missing/invalid `Total Charges` value "
                            "(common for brand-new customers in the raw Telco data) — the model handles "
                            "missing values natively, so these were still predicted, not dropped."
                        )

                    st.divider()
                    st.subheader("Batch Results")

                    s1, s2, s3 = st.columns(3)
                    s1.metric("Customers analyzed", f"{len(results_df):,}")
                    s2.metric("Flagged as churn-risk", f"{int(predictions.sum()):,} ({predictions.mean():.1%})")
                    s3.metric("Average churn probability", f"{probabilities.mean():.1%}")

                    st.dataframe(
                        results_df.sort_values("Churn Probability", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )

                    st.download_button(
                        "Download results as CSV",
                        results_df.to_csv(index=False).encode("utf-8"),
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        width="stretch",
                    )
