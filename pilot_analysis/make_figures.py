#!/usr/bin/env python3
"""
Generate pilot study figures for VoiceOver/pilot_analysis/.

Figure 1 (fig1_roc.pdf): ROC curves — AU12 alone, Duchenne (AU06+AU12),
  best pairwise (AU09+AU12), and logistic over all AUs. Two panels:
  unweighted (equal task weight) and weighted (by number of annotators).

Figure 2 (fig2_auc_ranking.pdf): AUC ranking bar chart for all univariate
  AU features (mean_r), both weightings.

Figure 3 (fig3_logistic_coef.pdf): Standardized logistic regression
  coefficients, unweighted and weighted, showing which AUs predict smile.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_DIR / "analysis"
OUT_DIR = Path(__file__).resolve().parent

DATASET = ANALYSIS_DIR / "au_features_dataset.csv"
ROC_RESULTS = ANALYSIS_DIR / "au_roc_sweep" / "au_roc_results.csv"
COEF_UW = ANALYSIS_DIR / "au_roc_sweep" / "logistic_coef_unweighted.json"
COEF_W  = ANALYSIS_DIR / "au_roc_sweep" / "logistic_coef_weighted.json"

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,  # editable text in Illustrator
    "ps.fonttype": 42,
})

# ── data ─────────────────────────────────────────────────────────────────────
def load_data():
    df_all = pd.read_csv(DATASET)
    df_task = (
        df_all.drop_duplicates("task_number")
        .copy()
        .pipe(lambda d: d[d["manifest_AU12_mean_r"] >= 1.5])
        .reset_index(drop=True)
    )
    y = df_task["consensus_binary"].to_numpy(dtype=np.int32)
    w = df_task["n_annotators"].to_numpy(dtype=np.float64)

    au_r_mean_cols = sorted(
        c for c in df_task.columns if c.startswith("AU") and c.endswith("_r_mean")
    )
    X = df_task[au_r_mean_cols].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit two logistic models
    clf_uw = LogisticRegressionCV(cv=5, max_iter=2000, random_state=42, n_jobs=-1)
    clf_uw.fit(X_scaled, y)
    score_uw = clf_uw.predict_proba(X_scaled)[:, 1]

    clf_w = LogisticRegressionCV(cv=5, max_iter=2000, random_state=42, n_jobs=-1)
    clf_w.fit(X_scaled, y, sample_weight=w)
    score_w = clf_w.predict_proba(X_scaled)[:, 1]

    return df_task, y, w, au_r_mean_cols, score_uw, score_w


def roc(y, score, weights=None):
    fpr, tpr, thr = roc_curve(y, score, sample_weight=weights)
    auc = roc_auc_score(y, score, sample_weight=weights)
    return fpr, tpr, auc


# ── figure 1: ROC comparison ──────────────────────────────────────────────────
def fig1_roc(df_task, y, w, score_uw, score_w):
    au12 = df_task["AU12_r_mean"].to_numpy()
    au06_au12 = (df_task["AU06_r_mean"] + df_task["AU12_r_mean"]).to_numpy()
    au09_au12 = (df_task["AU09_r_mean"] + df_task["AU12_r_mean"]).to_numpy()

    models = [
        ("AU12 alone",        au12,       au12),
        ("AU06+AU12 (Duchenne)", au06_au12, au06_au12),
        ("AU09+AU12",          au09_au12, au09_au12),
        ("Logistic (all AUs)", score_uw,  score_w),
    ]
    colors = ["#6b7280", "#2563eb", "#16a34a", "#dc2626"]
    linestyles = [":", "--", "-.", "-"]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)

    for ax, (w_label, get_score) in zip(
        axes,
        [("Unweighted", lambda s_uw, s_w: s_uw),
         ("Weighted by N annotators", lambda s_uw, s_w: s_w)],
    ):
        ax_weights = None if "Unweighted" in w_label else w
        for (label, s_uw, s_w), color, ls in zip(models, colors, linestyles):
            score = get_score(s_uw, s_w)
            fpr, tpr, auc = roc(y, score, ax_weights)
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.4,
                    label=f"{label}  (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#d1d5db", linewidth=0.8, zorder=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        if ax is axes[0]:
            ax.set_ylabel("True Positive Rate")
        ax.set_title(w_label, fontsize=8, pad=4)
        ax.legend(loc="lower right", fontsize=6.5)
        ax.set_aspect("equal")

    fig.tight_layout(pad=0.8, w_pad=1.2)
    path = OUT_DIR / "fig1_roc.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1 → {path}")


# ── figure 2: AUC ranking ─────────────────────────────────────────────────────
def fig2_auc_ranking(df_roc):
    uni = (
        df_roc[
            (df_roc["type"] == "univariate") &
            (df_roc["feature"].str.endswith("_r_mean"))
        ]
        .copy()
    )
    uni["au"] = uni["feature"].str.replace("_r_mean", "", regex=False)

    # Add logistic as reference rows
    log_rows = df_roc[df_roc["type"] == "logistic_regression"].copy()
    log_rows["au"] = "Logistic\n(all AUs)"

    combined = pd.concat([
        uni[["au", "weighting", "auc"]],
        log_rows[["au", "weighting", "auc"]],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2), sharey=True)

    HIGHLIGHT = {"AU12": "#dc2626", "AU06": "#2563eb", "AU09": "#16a34a",
                 "AU25": "#9333ea", "Logistic\n(all AUs)": "#f97316"}
    DEFAULT_COLOR = "#94a3b8"

    for ax, w_mode in zip(axes, ["unweighted", "weighted"]):
        sub = (
            combined[combined["weighting"] == w_mode]
            .sort_values("auc", ascending=True)
            .reset_index(drop=True)
        )
        colors = [HIGHLIGHT.get(au, DEFAULT_COLOR) for au in sub["au"]]
        bars = ax.barh(sub["au"], sub["auc"], color=colors, height=0.65)
        ax.axvline(0.5, color="#9ca3af", linewidth=0.8, linestyle="--", zorder=0)

        for bar, val in zip(bars, sub["auc"]):
            ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=6)

        ax.set_xlim(0.35, 0.87)
        ax.set_xlabel("AUC")
        title = "Unweighted" if w_mode == "unweighted" else "Weighted by N annotators"
        ax.set_title(title, fontsize=8, pad=4)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    axes[0].set_ylabel("Feature")
    fig.tight_layout(pad=0.8, w_pad=0.6)
    path = OUT_DIR / "fig2_auc_ranking.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2 → {path}")


# ── figure 3: logistic coefficients ──────────────────────────────────────────
def fig3_coef():
    with open(COEF_UW) as f: coef_uw = json.load(f)
    with open(COEF_W)  as f: coef_w  = json.load(f)

    # Relabel AUs
    def clean_label(k): return k.replace("_r_mean", "")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0), sharey=False)

    for ax, coef_dict, title in [
        (axes[0], coef_uw, "Unweighted"),
        (axes[1], coef_w,  "Weighted by N annotators"),
    ]:
        labels = [clean_label(k) for k in coef_dict]
        vals = list(coef_dict.values())
        sorted_pairs = sorted(zip(labels, vals), key=lambda x: x[1])
        slabels, svals = zip(*sorted_pairs)

        HIGHLIGHT = {"AU12": "#dc2626", "AU06": "#2563eb", "AU09": "#16a34a",
                     "AU25": "#9333ea"}
        bar_colors = []
        for lbl, v in zip(slabels, svals):
            if lbl in HIGHLIGHT:
                bar_colors.append(HIGHLIGHT[lbl])
            elif v > 0:
                bar_colors.append("#93c5fd")
            else:
                bar_colors.append("#fca5a5")

        ax.barh(slabels, svals, color=bar_colors, height=0.65)
        ax.axvline(0, color="#6b7280", linewidth=0.8)
        ax.set_xlabel("Coefficient (standardized)")
        ax.set_title(title, fontsize=8, pad=4)

    fig.tight_layout(pad=0.8, w_pad=1.2)
    path = OUT_DIR / "fig3_logistic_coef.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3 → {path}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data and fitting models...")
    df_task, y, w, au_r_mean_cols, score_uw, score_w = load_data()
    df_roc = pd.read_csv(ROC_RESULTS)

    print("Generating figures...")
    fig1_roc(df_task, y, w, score_uw, score_w)
    fig2_auc_ranking(df_roc)
    fig3_coef()
    print("Done.")
