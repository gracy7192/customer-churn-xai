"""Figures/tables that need the real dataset: ROC curves, clean SHAP beeswarm,
and Table IV recomputed per ORIGINAL feature (SHAP summed over one-hot dummies).

Run from the repo root with the dataset present:
    python3 paper_review/scripts/make_data_dependent_figures.py [--data data/Telco_customer_churn.xlsx]

It also checks that the saved model reproduces the paper's confusion matrix
(787/248/83/291 at threshold 0.30); a mismatch means the data/split differs.
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_review" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.30

FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges", "CLTV",
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def load(path):
    df = pd.read_excel(path)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)
    return df[FEATURES], df["Churn Value"]


def main(path):
    X, y = load(path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = joblib.load(ROOT / "models" / "churn_model.pkl")
    prob = model.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, (prob >= THRESHOLD).astype(int))
    print("confusion matrix @0.30:\n", cm, "\nmatches paper:", cm.tolist() == [[787, 248], [83, 291]])
    print("tuned ROC-AUC:", round(roc_auc_score(y_te, prob), 4))

    pre = model.named_steps["preprocessor"]
    curves = {"Tuned XGBoost": prob}
    for name, est in [("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
                      ("Random Forest", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42))]:
        p = Pipeline([("preprocessor", clone(pre)), ("model", est)]).fit(X_tr, y_tr)
        curves[name] = p.predict_proba(X_te)[:, 1]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    styles = {"Logistic Regression": ("-.", "#7f7f7f"), "Random Forest": ("--", "#c55a11"), "Tuned XGBoost": ("-", "#1f4e79")}
    for name, pr in curves.items():
        fpr, tpr, _ = roc_curve(y_te, pr)
        ax.plot(fpr, tpr, styles[name][0], color=styles[name][1], lw=1.2, label=f"{name} (AUC = {roc_auc_score(y_te, pr):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="black", lw=0.7, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False); ax.grid(alpha=0.3, lw=0.4)
    save(fig, "fig7_roc_curves")

    Xt = pd.DataFrame(pre.transform(X_te), columns=pre.get_feature_names_out())
    booster = model.named_steps["model"]
    sv = shap.TreeExplainer(booster).shap_values(Xt)
    names = [c.replace("num__", "").replace("cat__", "").replace("_", " = ", 1) if c.startswith("cat__") else c.replace("num__", "")
             for c in Xt.columns]
    plt.figure(figsize=(3.5, 3.6))
    shap.summary_plot(sv, Xt.values, feature_names=names, max_display=12, show=False, plot_size=None)
    save(plt.gcf(), "fig3_shap_beeswarm_clean")

    # Table IV by original feature: sum SHAP over each feature's one-hot columns, then mean |.|
    owner = {}
    for col in Xt.columns:
        body = col.split("__", 1)[1]
        owner[col] = body if col.startswith("num__") else next(f for f in FEATURES if body.startswith(f + "_"))
    shap_df = pd.DataFrame(sv, columns=Xt.columns)
    agg_shap = shap_df.T.groupby(owner).sum().T.abs().mean().sort_values(ascending=False)
    gain = pd.Series(booster.feature_importances_, index=Xt.columns).groupby(owner).sum()
    table = pd.DataFrame({"mean_abs_shap": agg_shap, "gain": gain.reindex(agg_shap.index)}).round(3)
    table.to_csv(OUT / "table4_per_original_feature.csv")
    print(table.head(10))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "Telco_customer_churn.xlsx"))
    main(ap.parse_args().data)
