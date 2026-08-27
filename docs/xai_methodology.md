# Explainability and Retention Recommendation Methodology

**Author:** Gurnoor
**Scope:** This is the Explainable AI section of the shared research paper (project brief Section 21), covering the work in `notebooks/03_xai.ipynb` and `src/xai_explainer.py`. It does not cover data preparation, feature engineering, or model selection/tuning, which are Gracy's contribution and documented separately.

## 1. Why explainability is needed for churn prediction

The final model (a tuned XGBoost classifier, decision threshold 0.30) answers "is this customer likely to churn?" but a probability alone is not actionable for a retention team: it does not say *why* a customer is at risk or *what* a company should do differently for them. Two customers can have the same 70% churn probability for entirely different reasons — one from a month-to-month contract with no add-ons, another from being a brand-new customer with high monthly charges — and effective retention outreach differs accordingly. Explainability closes the gap between "a model predicted this" and "here is something a person can act on."

## 2. SHAP: global and local explanations

The model artifact is an `sklearn.Pipeline` consisting of a `ColumnTransformer` (StandardScaler on 4 numeric features, OneHotEncoder on 16 categorical features, 20 raw features -> 47 processed features) followed by the tuned XGBoost estimator. SHAP's `TreeExplainer` was applied directly to the XGBoost step, fed data already run through the preprocessor, since TreeExplainer needs the trees' native numeric input space rather than the raw pipeline's input schema.

**Global explanation:** SHAP values were computed for all 1,409 held-out test customers and aggregated as mean absolute SHAP value per feature (`results/shap_feature_importance.csv`, `results/shap_summary.png`, `results/shap_bar_importance.png`). This is a distinct metric from Gracy's existing `results/xgboost_feature_importance.csv` — that file reports XGBoost's internal gain-based importance (how much a feature reduced training loss when split on), which carries no direction and is not customer-specific. The SHAP ranking measures how much a feature actually moves individual predictions, on real held-out data, in either direction. In this analysis, the top global drivers were month-to-month contracts, short tenure, absence of dependents, high monthly charges, and absence of an online-security add-on — consistent with established churn intuition for subscription telecom services.

**Local explanation:** For an individual customer, the 47 processed (one-hot-expanded) SHAP values were aggregated back to the 20 original raw features (summing a categorical feature's one-hot columns, since exactly one is active per customer), producing one SHAP contribution per real-world feature alongside the customer's actual value. Verification confirmed that `base_value + sum(SHAP values)` exactly reconstructs the model's raw margin output for every tested customer — i.e., these SHAP values genuinely explain this specific model's specific prediction, not an approximation or a mismatched setup.

## 3. From SHAP values to a business explanation

A template-based function (`generate_business_explanation()`) converts a (feature, customer's actual value, SHAP contribution) triple into a plain-English sentence, e.g.:

> "The customer's month-to-month contract is contributing to a higher predicted churn risk."
> "The customer's tenure of 72 months is contributing to a lower predicted churn risk."

Every sentence is generated from the specific customer's specific data — there is no fixed, generic explanation independent of input. Wording deliberately avoids causal phrasing ("causes churn," "will churn") in favor of "is contributing to a higher/lower predicted churn risk," because SHAP explains *this model's* behavior on *this input*, not a real-world causal mechanism (see Limitations).

## 4. Actionability: from driver to recommendation

Not every SHAP-identified driver is something a business can change — `Gender`, `Senior Citizen`, `Partner`, `Dependents`, and `CLTV` (a derived score, not a lever) are fixed customer attributes. Each of the 20 raw features was classified as **actionable** (a real plan/product/service lever, 15 features) or **non-actionable** (5 features), and the recommendation engine only ever proposes an action for an actionable, currently risk-*increasing* driver.

`RECOMMENDATION_RULES` is a transparent, hand-authored, rule-based mapping from feature (and, where relevant, its value) to a retention action — e.g. a month-to-month contract maps to an incentive to move to an annual/two-year plan; absence of online security maps to an add-on offer; low tenure maps to an onboarding/early-engagement program. This mapping is explicitly **not** learned by the XGBoost model; the model predicts churn probability, and this separate business layer translates the churn drivers the model (via SHAP) already identified into an action. Every recommendation stays traceable to the specific driver and the customer's specific value that produced it.

## 5. Validation

Explanations and recommendations were validated on three held-out customers representing distinct cases (a high-risk customer at 93.5% probability, a low-risk customer at 0.17%, and a borderline customer at 30.0% — right at the decision threshold), checking automatically that: the probability is valid, the threshold decision is consistent, the SHAP values reconstruct the model's actual output for that customer, the reported top drivers are genuinely derived from SHAP (not invented), every recommendation traces to an identified actionable risk-increasing driver, and no causal language appears in the generated text. All checks passed for all three cases.

## 6. Limitations

- **SHAP explains model behavior, not real-world causality.** A feature contributing positively to the model's predicted churn probability means the model's learned function responds that way to that feature — it does not establish that changing the feature would change the customer's real-world likelihood of churning. Explanations here are phrased accordingly.
- **Recommendations are rule-based, not model-learned.** The mapping from driver to action was authored manually based on domain reasoning, not derived from any model that predicts which action actually retains a customer. A dedicated recommendation/uplift model (which this project does not build) would be needed to test whether a given action actually changes churn probability for a given customer.
- **Dataset limitations.** The Telco Customer Churn dataset (7,043 customers, single snapshot) does not capture customer sentiment, competitor behavior, support-interaction history, or true reasons for churn beyond the `Churn Reason` field (not used as a model feature); SHAP can only explain what the 20 modeling features encode.
- **Model performance limitations.** At the default 0.50 threshold the tuned XGBoost model reaches ~80.3% accuracy and ~85.5% ROC-AUC (`results/model_comparison.csv`); at the project's chosen 0.30 threshold, accuracy drops to ~76-77% while churn recall rises to ~78% (`results/threshold_analysis.csv`). Any explanation is only as good as the underlying prediction it explains — a false positive/negative still receives a fluent-sounding explanation that reflects the model's behavior, not necessarily the customer's true situation.
- **Threshold trade-off.** The 0.30 threshold was deliberately chosen over the default 0.50 to prioritize catching more true churners (higher recall) at the cost of more false positives (lower precision, more customers flagged as at-risk who would not have churned). All churn-risk classifications and recommendations in this notebook and module are downstream of that choice; a different threshold would flag a different, though overlapping, set of customers and would not itself require retraining the model.
