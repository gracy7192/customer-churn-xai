
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
MODEL_PATH = Path("../models/churn_model.pkl")
CONFIG_PATH = Path("../models/model_config.json")

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

with st.form("customer_form"):
    st.subheader("Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["Gender"])
        senior = st.selectbox("Senior Citizen", CATEGORICAL_OPTIONS["Senior Citizen"])
        partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"])
        dependents = st.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"])
        tenure = st.slider("Tenure (Months)", *NUMERIC_SLIDERS["Tenure Months"])

    with col2:
        st.markdown("**Services**")
        phone = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["Phone Service"])
        multiple_lines = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["Multiple Lines"])
        internet = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["Internet Service"])
        online_security = st.selectbox("Online Security", CATEGORICAL_OPTIONS["Online Security"])
        online_backup = st.selectbox("Online Backup", CATEGORICAL_OPTIONS["Online Backup"])
        device_protection = st.selectbox("Device Protection", CATEGORICAL_OPTIONS["Device Protection"])
        tech_support = st.selectbox("Tech Support", CATEGORICAL_OPTIONS["Tech Support"])
        streaming_tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["Streaming TV"])
        streaming_movies = st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["Streaming Movies"])

    with col3:
        st.markdown("**Account & Billing**")
        contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
        paperless = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["Paperless Billing"])
        payment_method = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["Payment Method"])
        monthly_charges = st.slider("Monthly Charges ($)", *NUMERIC_SLIDERS["Monthly Charges"])
        total_charges = st.slider("Total Charges ($)", *NUMERIC_SLIDERS["Total Charges"])
        cltv = st.slider("CLTV", *NUMERIC_SLIDERS["CLTV"])

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

# ---------------------------------------------------------------------------
# 9. RUN PREDICTION -> SHAP -> EXPLANATIONS -> RECOMMENDATIONS -> DISPLAY
# ---------------------------------------------------------------------------
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

        st.divider()
        st.subheader("Prediction Results")
        m1, m2 = st.columns(2)
        m1.metric("Churn Probability", f"{prob:.1%}")
        m2.metric(
            f"Decision (threshold {CHURN_THRESHOLD:.2f})",
            "🔴 High churn risk" if is_risk else "🟢 Not classified as churn-risk",
        )

        top_features = get_top_shap_features_for_customer(customer, top_n=None)
        increasing = top_features[top_features["shap_value"] > 0].head(5)
        decreasing = top_features[top_features["shap_value"] < 0].sort_values("shap_value").head(5)

        st.subheader("Why this prediction? (SHAP explanation)")
        top5 = top_features.reindex(top_features["shap_value"].abs().sort_values(ascending=False).index).head(5)
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["#d9534f" if v > 0 else "#5cb85c" for v in top5["shap_value"][::-1]]
        ax.barh(top5["feature"][::-1], top5["shap_value"][::-1], color=colors)
        ax.set_xlabel("Impact on churn probability (SHAP value)")
        ax.set_title("Top factors for this customer")
        st.pyplot(fig)
        st.caption("🔴 Red = pushes prediction toward churn · 🟢 Green = pushes toward staying")

        col_up, col_down = st.columns(2)
        with col_up:
            st.markdown("**Why risk is higher**")
            for row in increasing.itertuples():
                st.write("• " + generate_business_explanation(row.feature, row.customer_value, row.shap_value))
        with col_down:
            st.markdown("**Why risk is lower**")
            for row in decreasing.itertuples():
                st.write("• " + generate_business_explanation(row.feature, row.customer_value, row.shap_value))

        st.subheader("Recommended Retention Actions")
        recommendations = generate_recommendations(top_features, top_n=5)
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- **{rec['driver']}** (`{rec['customer_value']}`) → {rec['action']}")
        else:
            st.info("No actionable risk-increasing drivers identified for this customer.")

    except Exception as e:
        st.error(f"Something went wrong while generating the prediction: {e}")
