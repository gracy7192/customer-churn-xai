"""Build IEEE-sized paper figures from results already stored in the repo.

Run from the repo root:
    python3 paper_review/scripts/make_paper_figures.py [--ibm-csv PATH]

Fig. 1 reads data/Telco_customer_churn.xlsx if present (the 33-column file the
paper cites); otherwise pass the public 21-column IBM CSV with --ibm-csv. The
churn-rate columns are identical in both.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_review" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL1, COL2 = 3.5, 7.16  # IEEE single / double column width (inches)
BLUE, ORANGE, GRAY, RED, TEAL = "#1f4e79", "#c55a11", "#7f7f7f", "#b03a2e", "#2e86ab"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def clean(feature):
    f = feature.replace("num__", "").replace("cat__", "")
    if "_" in f and not feature.startswith("num__"):
        a, b = f.rsplit("_", 1)
        return f"{a} = {b}"
    return f


def fig1(ibm_csv):
    xlsx = ROOT / "data" / "Telco_customer_churn.xlsx"
    if xlsx.exists():
        df = pd.read_excel(xlsx)
        contract, internet, churn = df["Contract"], df["Internet Service"], df["Churn Label"].eq("Yes")
    else:
        df = pd.read_csv(ibm_csv)
        contract, internet, churn = df["Contract"], df["InternetService"], df["Churn"].eq("Yes")
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 2.1), sharey=True)
    for ax, series, order, title in [
        (axes[0], contract, ["Month-to-month", "One year", "Two year"], "(a) Contract type"),
        (axes[1], internet, ["DSL", "Fiber optic", "No"], "(b) Internet service"),
    ]:
        rate = churn.groupby(series).mean().reindex(order) * 100
        bars = ax.bar(range(3), rate.values, color=[BLUE, TEAL, GRAY], edgecolor="black", linewidth=0.5, width=0.65)
        for b, v in zip(bars, rate.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["M2M" if o == "Month-to-month" else o if o != "No" else "None" for o in order])
        ax.set_title(title)
        ax.set_ylim(0, 52)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Churn rate (%)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig1_churn_by_contract_internet")


def fig2():
    t = pd.read_csv(ROOT / "results" / "threshold_analysis.csv")
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    ax.plot(t.Threshold, t.Recall, "-o", color=BLUE, ms=3, label="Recall")
    ax.plot(t.Threshold, t.Precision, "--s", color=ORANGE, ms=3, label="Precision")
    ax.plot(t.Threshold, t.F1, "-.^", color=GRAY, ms=3.5, label="F1-score")
    ax.axvline(0.30, color="black", ls=":", lw=0.9)
    ax.annotate("selected\nthreshold = 0.30", xy=(0.30, 0.95), xytext=(0.36, 0.93), fontsize=7,
                va="top", arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.2, 1.0)
    ax.set_xlim(0.18, 0.72)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="lower center", ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "fig2_threshold_sweep")


def fig3():
    shap_df = pd.read_csv(ROOT / "results" / "shap_feature_importance.csv").head(10)
    gain = pd.read_csv(ROOT / "results" / "xgboost_feature_importance.csv").set_index("Feature")["Importance"]
    labels = [clean(f) for f in shap_df.feature]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5), sharey=True)
    y = np.arange(len(labels))[::-1]
    axes[0].barh(y, shap_df.mean_abs_shap_value, color=BLUE, edgecolor="black", linewidth=0.4, height=0.65)
    axes[0].set_xlabel("Mean |SHAP value| (log-odds)")
    axes[0].set_title("(a) SHAP")
    axes[1].barh(y, gain.reindex(shap_df.feature).values, color=ORANGE, edgecolor="black", linewidth=0.4, height=0.65)
    axes[1].set_xlabel("XGBoost gain importance")
    axes[1].set_title("(b) Gain")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig3_shap_vs_gain")


def fig4():
    cm = np.array([[787, 248], [83, 291]])  # verified: notebooks/02_modeling.ipynb cell 83
    fig, ax = plt.subplots(figsize=(COL1 * 0.8, 2.3))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    names = [["TN", "FP"], ["FN", "TP"]]
    row_tot = cm.sum(axis=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}\n({cm[i, j] / row_tot[i] * 100:.1f}% of row)", ha="center",
                    va="center", fontsize=7.5, color="white" if cm[i, j] > 500 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Stayed", "Churned"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Stayed", "Churned"], rotation=90, va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    save(fig, "fig4_confusion_matrix")


def fig5():
    # Exact values printed in notebooks/02_modeling.ipynb (not the rounded results/model_comparison.csv).
    rows = {
        "Logistic\nRegression": (0.8020, 0.6426, 0.5722, 0.6054, 0.8489),
        "Random\nForest": (0.7942, 0.6373, 0.5214, 0.5735, 0.8396),
        "XGBoost\n(baseline)": (0.7566, 0.5284, 0.7701, 0.6268, 0.8492),
        "Tuned XGB\n@ 0.50": (0.8034, 0.6580, 0.5401, 0.5932, 0.8555),
        "Tuned XGB\n@ 0.30": (0.7651, 0.5399, 0.7781, 0.6375, 0.8555),
    }
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    cols = [GRAY, TEAL, BLUE, ORANGE, RED]
    hatches = ["", "//", "..", "xx", "\\\\"]
    fig, ax = plt.subplots(figsize=(COL2, 2.5))
    w = 0.16
    x = np.arange(len(rows))
    for k, m in enumerate(metrics):
        vals = [v[k] for v in rows.values()]
        bars = ax.bar(x + (k - 2) * w, vals, w, color=cols[k], edgecolor="black", linewidth=0.4, hatch=hatches[k], label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(list(rows))
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("Score")
    ax.axhline(0.7346, color="black", ls=":", lw=0.7)
    ax.text(1.005, 0.7346, "majority-\nclass\naccuracy\n(73.5%)", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=6)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig5_model_comparison")


def fig6():
    d = json.load(open(ROOT / "results" / "example_customer_explanations.json"))
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2))
    for ax, (name, c), tag in zip(axes, d.items(), "abc"):
        drivers = c["top_churn_drivers"][:5][::-1]
        vals = [x["contribution"] for x in drivers]
        ax.barh(range(5), vals, color=[RED if v > 0 else BLUE for v in vals], edgecolor="black", linewidth=0.4, height=0.65)
        ax.set_yticks(range(5)); ax.set_yticklabels([x["feature"] for x in drivers], fontsize=6.5)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"({tag}) {name.split(' (')[0]}: P(churn) = {c['churn_probability'] * 100:.1f}%", fontsize=7.5)
        ax.set_xlabel("SHAP contribution (log-odds)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig6_case_study_drivers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm-csv", default=None)
    a = ap.parse_args()
    fig1(a.ibm_csv)
    fig2(); fig3(); fig4(); fig5(); fig6()
