"""Explainable-AI layer for the churn model: SHAP-based explanations and
rule-based retention recommendations, packaged as an importable module
for dashboard integration (Ujjwal).

This module does not train or select the churn model - it only loads the
artifact produced from 02_modeling.ipynb (see notebooks/03_xai.ipynb,
Section 1, for how models/churn_model.pkl is created and verified) and
builds explainability on top of it. The logic here mirrors
notebooks/03_xai.ipynb exactly; that notebook is the source of truth for
how each piece was derived and validated - this module exists so the same
logic can be called from outside a notebook.

Usage:
    from xai_explainer import explain_customer

    customer = {
        "Gender": "Female", "Senior Citizen": "No", "Partner": "Yes",
        "Dependents": "No", "Tenure Months": 5, "Phone Service": "Yes",
        "Multiple Lines": "No", "Internet Service": "Fiber optic",
        "Online Security": "No", "Online Backup": "No",
        "Device Protection": "No", "Tech Support": "No",
        "Streaming TV": "No", "Streaming Movies": "No",
        "Contract": "Month-to-month", "Paperless Billing": "Yes",
        "Payment Method": "Electronic check", "Monthly Charges": 85.0,
        "Total Charges": 425.0, "CLTV": 3500,
    }
    result = explain_customer(customer)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
MODEL_PATH = _PROJECT_ROOT / "models" / "churn_model.pkl"
CONFIG_PATH = _PROJECT_ROOT / "models" / "model_config.json"

FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges", "CLTV",
]

# Fixed customer attributes: not something a retention offer can change.
NON_ACTIONABLE_FEATURES = {"Gender", "Senior Citizen", "Partner", "Dependents", "CLTV"}
ACTIONABLE_FEATURES = set(FEATURES) - NON_ACTIONABLE_FEATURES

# Yes/No features phrased as "having"/"not having <feature>".
YES_NO_FEATURES = {
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Multiple Lines", "Phone Service",
    "Paperless Billing", "Senior Citizen", "Dependents",
}
# Yes/No features phrased with a singular article ("a partner").
YES_NO_FEATURES_WITH_ARTICLE = {"Partner"}

# feature -> function(customer_value) -> recommended action. Transparent,
# hand-authored, rule-based - not learned by the model.
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


class _ChurnExplainer:
    """Lazily loads the model/config/SHAP explainer once per process."""

    def __init__(self):
        self.pipeline = None
        self.threshold = None
        self.preprocessor = None
        self.xgb_model = None
        self.shap_explainer = None
        self.processed_to_original = None

    def ensure_loaded(self):
        if self.pipeline is not None:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. See notebooks/03_xai.ipynb Section 1 for "
                "how this artifact is produced (it reproduces 02_modeling.ipynb's "
                "pipeline, which is not itself persisted to disk)."
            )

        self.pipeline = joblib.load(MODEL_PATH)
        with open(CONFIG_PATH) as f:
            self.threshold = json.load(f)["threshold"]

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.xgb_model = self.pipeline.named_steps["model"]
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)

        numeric_features = list(self.preprocessor.transformers_[0][2])
        categorical_features = list(self.preprocessor.transformers_[1][2])
        onehot_categories = self.preprocessor.named_transformers_["cat"].categories_

        mapping = [(f, None) for f in numeric_features]
        for feature_name, categories in zip(categorical_features, onehot_categories):
            for category in categories:
                mapping.append((feature_name, category))
        self.processed_to_original = mapping


_state = _ChurnExplainer()


def _customer_dict_to_frame(customer: dict) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in customer]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    return pd.DataFrame([{f: customer[f] for f in FEATURES}])


def predict_customer(customer: dict) -> dict:
    """Predict churn probability/decision for one raw customer record (dict)."""
    _state.ensure_loaded()
    raw_features = _customer_dict_to_frame(customer)
    probability = _state.pipeline.predict_proba(raw_features)[:, 1][0]
    is_churn_risk = int(probability >= _state.threshold)
    return {
        "churn_probability": float(probability),
        "churn_prediction": is_churn_risk,
    }


def get_top_shap_features(customer: dict, top_n=None) -> pd.DataFrame:
    """Aggregate SHAP values by original raw feature for one customer dict.

    Returns a DataFrame with one row per original feature: the customer's
    actual value, the aggregated SHAP contribution, and its direction,
    sorted by |SHAP value| descending.
    """
    _state.ensure_loaded()
    raw_features = _customer_dict_to_frame(customer)

    processed = _state.preprocessor.transform(raw_features)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()
    processed_df = pd.DataFrame(processed, columns=_state.preprocessor.get_feature_names_out())

    shap_row = _state.shap_explainer.shap_values(processed_df)[0]

    contributions = {}
    for shap_value, (feature, _category) in zip(shap_row, _state.processed_to_original):
        contributions[feature] = contributions.get(feature, 0.0) + shap_value

    records = [
        {
            "feature": feature,
            "customer_value": customer[feature],
            "shap_value": shap_value,
            "direction": "increases_risk" if shap_value > 0 else "decreases_risk",
        }
        for feature, shap_value in contributions.items()
    ]
    result = pd.DataFrame(records)
    result["abs_shap_value"] = result["shap_value"].abs()
    result = result.sort_values("abs_shap_value", ascending=False).drop(columns="abs_shap_value")
    return result.head(top_n) if top_n else result.reset_index(drop=True)


def generate_business_explanation(feature: str, customer_value, shap_value: float) -> str:
    """Turn one (feature, customer_value, shap_value) triple into a plain-English
    sentence. Never a generic sentence unrelated to the inputs, and never
    causal language ("causes churn") - only "contributing to a higher/lower
    predicted churn risk", since SHAP explains model behavior, not causality.
    """
    risk_word = "higher" if shap_value > 0 else "lower"

    if feature in YES_NO_FEATURES_WITH_ARTICLE and customer_value in ("Yes", "No"):
        has_clause = "having a" if customer_value == "Yes" else "not having a"
        return f"The customer {has_clause} {feature.lower()} is contributing to a {risk_word} predicted churn risk."

    if feature in YES_NO_FEATURES and customer_value in ("Yes", "No"):
        has_clause = "having" if customer_value == "Yes" else "not having"
        return f"The customer {has_clause} {feature.lower()} is contributing to a {risk_word} predicted churn risk."

    if feature == "Contract":
        return f"The customer's {str(customer_value).lower()} contract is contributing to a {risk_word} predicted churn risk."

    if feature == "Tenure Months":
        unit = "month" if customer_value == 1 else "months"
        return f"The customer's tenure of {customer_value} {unit} is contributing to a {risk_word} predicted churn risk."

    if feature == "Monthly Charges":
        return f"The customer's monthly charge of ${float(customer_value):.2f} is contributing to a {risk_word} predicted churn risk."

    if feature == "Total Charges":
        return f"The customer's total charges to date (${float(customer_value):.2f}) are contributing to a {risk_word} predicted churn risk."

    if feature == "CLTV":
        return f"The customer's customer lifetime value score ({customer_value}) is contributing to a {risk_word} predicted churn risk."

    return f"The customer's {feature.lower()} ('{customer_value}') is contributing to a {risk_word} predicted churn risk."


def classify_driver(feature: str) -> str:
    """Classify a raw feature as an actionable business lever or a fixed attribute."""
    return "actionable" if feature in ACTIONABLE_FEATURES else "non_actionable"


def generate_recommendations(customer: dict, top_n=5) -> list:
    """Map a customer's top actionable, risk-increasing SHAP drivers to
    retention actions. Non-actionable drivers and risk-reducing drivers
    never produce a recommendation.
    """
    top_features = get_top_shap_features(customer, top_n=None)
    top_features = top_features[top_features["feature"].isin(ACTIONABLE_FEATURES)]
    increasing = top_features[top_features["shap_value"] > 0].head(top_n)

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
            "shap_value": float(row.shap_value),
            "action": action,
        })
    return recommendations


def explain_customer(customer: dict, top_n: int = 5) -> dict:
    """Full XAI explanation for one customer, in the API-style structure
    Ujjwal's dashboard consumes:

    {
        "churn_probability": float,
        "churn_prediction": 0 or 1,
        "risk_level": "High churn risk" | "Not classified as churn-risk",
        "top_churn_drivers": [
            {"feature", "contribution", "direction", "explanation"}, ...
        ],
        "recommendations": [{"driver", "action"}, ...],
    }

    `customer` must contain all 20 raw modeling features (see FEATURES);
    no CustomerID or target column. `risk_level` stays a two-tier label
    driven by the existing 0.30 threshold - not an invented Low/Medium/High
    scale (see project brief Section 18).
    """
    prediction = predict_customer(customer)
    top_features = get_top_shap_features(customer, top_n=top_n)
    recommendations = generate_recommendations(customer, top_n=top_n)

    top_churn_drivers = [
        {
            "feature": row.feature,
            "contribution": round(float(row.shap_value), 4),
            "direction": row.direction,
            "explanation": generate_business_explanation(row.feature, row.customer_value, row.shap_value),
        }
        for row in top_features.itertuples()
    ]

    return {
        "churn_probability": round(prediction["churn_probability"], 4),
        "churn_prediction": prediction["churn_prediction"],
        "risk_level": "High churn risk" if prediction["churn_prediction"] else "Not classified as churn-risk",
        "top_churn_drivers": top_churn_drivers,
        "recommendations": [{"driver": r["driver"], "action": r["action"]} for r in recommendations],
    }
