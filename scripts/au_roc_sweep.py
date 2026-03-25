#!/usr/bin/env python3
"""
AU Feature ROC Sweep.

Uses human annotation labels (smile vs not_a_smile) as ground truth to find
which OpenFace AU features — individually and in combination — best discriminate
smile segments from non-smile segments.

Scope: only tasks with AU12_r_mean >= 1.5 (guaranteed by the upstream manifest).
New combinations are thus evaluated on the same candidate pool as the original
AU12-only pipeline, making comparisons scientifically valid.

Two weighting modes run in parallel:
  unweighted -- each task contributes weight 1 regardless of annotator count
  weighted   -- each task weighted by its number of annotators (more raters = higher confidence)

Analysis
--------
1. Univariate ROC for every AU_r_mean, AU_r_peak, AU_r_mass
2. All pairwise AU_r_mean sum scores (136 pairs for 17 AUs)
3. Duchenne composite: AU06_r_mean + AU12_r_mean (additive) and
   AU06_r_mean * AU12_r_mean (multiplicative, Duchenne-specific)
4. Logistic regression on all AU_r_mean features (5-fold CV, L2)
5. For each analysis, also the binary classifier mean_c variants

Outputs (analysis/au_roc_sweep/)
---------------------------------
au_roc_results.csv           all AUC / F1 / threshold results
logistic_coef_unweighted.json
logistic_coef_weighted.json
figures/roc_univariate.png
figures/auc_ranking.png
figures/roc_duchenne_vs_au12.png
figures/roc_top5_combos.png
"""

import json
import warnings
from itertools import combinations
from pathlib import Path

# Suppress sklearn deprecation noise unrelated to analysis
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = PROJECT_DIR / "analysis"
DATASET_PATH = ANALYSIS_DIR / "au_features_dataset.csv"
OUT_DIR = ANALYSIS_DIR / "au_roc_sweep"
FIG_DIR = OUT_DIR / "figures"

AU12_MIN = 1.5  # minimum AU12_r_mean; tasks below this are excluded (should be none)
DUCHENNE_AUS = ["AU06", "AU12"]


# ── helpers ──────────────────────────────────────────────────────────────────

def consensus_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per task using pre-computed majority-vote consensus.
    Weight column = n_annotators (used for weighted analysis).
    """
    return (
        df.drop_duplicates("task_number")
        .copy()
        .assign(weight=lambda d: d["n_annotators"].astype(float))
        .reset_index(drop=True)
    )


def roc_stats(
    y_true: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """ROC curve, AUC, and operating point at Youden's J maximum."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            auc = float(roc_auc_score(y_true, score, sample_weight=weights))
        except ValueError:
            return {}
        fpr, tpr, thresholds = roc_curve(y_true, score, sample_weight=weights)

    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_thresh = float(thresholds[best_idx])

    pred = (score >= best_thresh).astype(int)
    w = weights if weights is not None else np.ones(len(y_true))
    tp = float(np.sum(((pred == 1) & (y_true == 1)) * w))
    fp = float(np.sum(((pred == 1) & (y_true == 0)) * w))
    fn = float(np.sum(((pred == 0) & (y_true == 1)) * w))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "auc": auc,
        "best_threshold": best_thresh,
        "precision_at_best": precision,
        "recall_at_best": recall,
        "f1_at_best": f1,
        "_fpr": fpr,
        "_tpr": tpr,
    }


def record(feature: str, kind: str, weighting: str, n_tasks: int, stats: dict) -> dict | None:
    if not stats:
        return None
    return {
        "feature": feature,
        "type": kind,
        "weighting": weighting,
        "n_tasks": n_tasks,
        "auc": stats["auc"],
        "best_threshold": stats["best_threshold"],
        "precision_at_best": stats["precision_at_best"],
        "recall_at_best": stats["recall_at_best"],
        "f1_at_best": stats["f1_at_best"],
        "_fpr": stats["_fpr"],
        "_tpr": stats["_tpr"],
    }


def run_pair(y, score, n, label, kind, w_col):
    """Run weighted + unweighted and return two records."""
    out = []
    for w_mode, weights in [("unweighted", None), ("weighted", w_col)]:
        r = record(label, kind, w_mode, n, roc_stats(y, score, weights))
        if r:
            out.append(r)
    return out


# ── analysis steps ────────────────────────────────────────────────────────────

def univariate_sweep(df: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> list[dict]:
    """ROC for every AU_r_mean, AU_r_peak, AU_r_mass, AU_c_mean."""
    results = []
    feat_cols = [c for c in df.columns if
                 c.startswith("AU") and
                 (c.endswith("_r_mean") or c.endswith("_r_peak") or
                  c.endswith("_r_mass") or c.endswith("_c_mean"))]
    n = len(y)
    for col in feat_cols:
        score = df[col].to_numpy(dtype=np.float64)
        results.extend(run_pair(y, score, n, col, "univariate", weights))
    return results


def pairwise_sweep(df: pd.DataFrame, y: np.ndarray, weights: np.ndarray,
                   au_names: list[str]) -> list[dict]:
    """All pairwise sums of AU_r_mean (136 pairs for 17 AUs)."""
    results = []
    n = len(y)
    available = [au for au in au_names if f"{au}_r_mean" in df.columns]
    for a, b in combinations(available, 2):
        score = (df[f"{a}_r_mean"] + df[f"{b}_r_mean"]).to_numpy(dtype=np.float64)
        label = f"{a}+{b}"
        results.extend(run_pair(y, score, n, label, "combo_2", weights))
    return results


def duchenne_sweep(df: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> list[dict]:
    """Additive and multiplicative Duchenne (AU06 × AU12) scores."""
    results = []
    n = len(y)
    au06 = df["AU06_r_mean"].to_numpy(dtype=np.float64)
    au12 = df["AU12_r_mean"].to_numpy(dtype=np.float64)

    # Additive already covered by pairwise, but add multiplicative explicitly
    score_product = au06 * au12
    results.extend(run_pair(y, score_product, n, "AU06*AU12_product", "duchenne", weights))

    # Normalized Duchenne ratio: AU06 / (AU12 + 1e-6)  — relative cheek activation
    score_ratio = au06 / (au12 + 1e-6)
    results.extend(run_pair(y, score_ratio, n, "AU06/AU12_ratio", "duchenne", weights))

    return results


def logistic_sweep(df: pd.DataFrame, y: np.ndarray, weights: np.ndarray,
                   au_r_mean_cols: list[str]) -> tuple[list[dict], dict, dict]:
    """LogisticRegressionCV on all AU_r_mean features (5-fold)."""
    X = df[au_r_mean_cols].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(y)

    results = []
    coefs: dict[str, dict] = {}

    for w_mode, w in [("unweighted", None), ("weighted", weights)]:
        try:
            clf = LogisticRegressionCV(cv=5, max_iter=2000, random_state=42, n_jobs=-1)
            clf.fit(X_scaled, y, sample_weight=w)
            score = clf.predict_proba(X_scaled)[:, 1]
            stats = roc_stats(y, score, w)
            r = record("logistic_all_AUs", "logistic_regression", w_mode, n, stats)
            if r:
                results.append(r)
            coefs[w_mode] = dict(zip(au_r_mean_cols, clf.coef_[0].tolist()))
        except Exception as e:
            print(f"  Logistic regression ({w_mode}) failed: {e}")

    return results, coefs.get("unweighted", {}), coefs.get("weighted", {})


# ── figures ───────────────────────────────────────────────────────────────────

def _au_color(au_name: str) -> str:
    if au_name == "AU12":
        return "#dc2626"
    if au_name == "AU06":
        return "#2563eb"
    return "#94a3b8"


def fig_auc_ranking(results: list[dict], out_dir: Path) -> None:
    """Horizontal bar chart of univariate AUC, sorted, for both weightings."""
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results])
    uni = df[(df["type"] == "univariate") & (df["feature"].str.endswith("_r_mean"))].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, w_mode in zip(axes, ["unweighted", "weighted"]):
        sub = uni[uni["weighting"] == w_mode].sort_values("auc", ascending=True)
        labels = sub["feature"].str.replace("_r_mean", "", regex=False)
        colors = [_au_color(lbl.split("_")[0]) for lbl in labels]
        bars = ax.barh(labels, sub["auc"], color=colors, height=0.7)
        ax.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1, alpha=0.6,
                   label="Random (AUC=0.5)")
        for bar, val in zip(bars, sub["auc"]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        ax.set_xlabel("AUC (ROC)", fontsize=10)
        ax.set_title(f"Univariate AU AUC — {w_mode}", fontsize=11)
        ax.set_xlim(0.3, 1.0)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("AU Feature Discriminability (smile vs not_a_smile)", fontsize=12)
    fig.savefig(out_dir / "auc_ranking.png", dpi=150)
    plt.close(fig)


def fig_roc_univariate(results: list[dict], out_dir: Path) -> None:
    """All univariate AU_r_mean ROC curves, unweighted, sorted by AUC."""
    uni = [r for r in results
           if r["type"] == "univariate"
           and r["weighting"] == "unweighted"
           and r["feature"].endswith("_r_mean")]
    uni_sorted = sorted(uni, key=lambda r: r["auc"], reverse=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab20")
    for idx, r in enumerate(uni_sorted):
        au = r["feature"].replace("_r_mean", "")
        is_key = au in DUCHENNE_AUS
        lw = 2.5 if is_key else 1.2
        ls = "-" if is_key else "--"
        color = _au_color(au) if is_key else cmap(idx / len(uni_sorted))
        ax.plot(r["_fpr"], r["_tpr"], linewidth=lw, linestyle=ls, color=color,
                label=f"{au} ({r['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Univariate ROC — all AU_r_mean (unweighted)", fontsize=12)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_univariate.png", dpi=150)
    plt.close(fig)


def fig_duchenne_vs_au12(results: list[dict], out_dir: Path) -> None:
    """Head-to-head: AU12 vs AU06+AU12 additive vs Duchenne product vs logistic."""
    keys = {
        "AU12_r_mean": "AU12 only",
        "AU06+AU12": "AU06+AU12 (additive)",
        "AU06*AU12_product": "AU06×AU12 (Duchenne product)",
        "logistic_all_AUs": "Logistic all AUs",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, w_mode in zip(axes, ["unweighted", "weighted"]):
        for r in results:
            if r["weighting"] != w_mode or r["feature"] not in keys:
                continue
            name = keys[r["feature"]]
            lw = 2.5 if "Logistic" in name else 2.0
            ax.plot(r["_fpr"], r["_tpr"], linewidth=lw,
                    label=f"{name} (AUC={r['auc']:.3f})")
        ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"Duchenne vs AU12 vs Logistic — {w_mode}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Key Feature Comparisons (smile vs not_a_smile)", fontsize=12)
    fig.savefig(out_dir / "roc_duchenne_vs_au12.png", dpi=150)
    plt.close(fig)


def fig_top5_combos(results: list[dict], out_dir: Path) -> None:
    """Top-5 pairwise combos vs AU12 baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, w_mode in zip(axes, ["unweighted", "weighted"]):
        combos = [r for r in results
                  if r["type"] == "combo_2" and r["weighting"] == w_mode]
        top5 = sorted(combos, key=lambda r: r["auc"], reverse=True)[:5]
        cmap = plt.get_cmap("tab10")
        for idx, r in enumerate(top5):
            ax.plot(r["_fpr"], r["_tpr"], linewidth=1.8, color=cmap(idx),
                    label=f"{r['feature']} ({r['auc']:.3f})")
        # AU12 baseline
        for r in results:
            if r["feature"] == "AU12_r_mean" and r["weighting"] == w_mode:
                ax.plot(r["_fpr"], r["_tpr"], "k--", linewidth=2,
                        label=f"AU12 baseline ({r['auc']:.3f})")
                break
        ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"Top-5 pairwise combos — {w_mode}", fontsize=11)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_top5_combos.png", dpi=150)
    plt.close(fig)


def fig_logistic_coef(coef_unweighted: dict, coef_weighted: dict, out_dir: Path) -> None:
    """Bar chart of logistic regression coefficients."""
    if not coef_unweighted and not coef_weighted:
        return
    ref = coef_unweighted or coef_weighted
    labels = [k.replace("_r_mean", "") for k in ref.keys()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, (title, coef_dict) in zip(
        axes,
        [("Unweighted", coef_unweighted), ("Weighted by N annotators", coef_weighted)],
    ):
        if not coef_dict:
            ax.set_visible(False)
            continue
        vals = list(coef_dict.values())
        sorted_pairs = sorted(zip(labels, vals), key=lambda x: x[1])
        slabels, svals = zip(*sorted_pairs)
        colors = ["#dc2626" if v > 0 else "#3b82f6" for v in svals]
        ax.barh(slabels, svals, color=colors, height=0.7)
        ax.axvline(0, color="#6b7280", linewidth=1)
        ax.set_xlabel("Logistic coefficient (standardized)", fontsize=10)
        ax.set_title(f"Logistic Coefficients — {title}", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("AU Feature Weights (Logistic Regression, all AU_r_mean)", fontsize=12)
    fig.savefig(out_dir / "logistic_coefficients.png", dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--min-annotators", type=int, default=1,
                        help="Minimum number of annotators per task to include")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset}...")
    df_all = pd.read_csv(args.dataset)
    print(f"  {len(df_all):,} rows, {df_all['task_number'].nunique()} unique tasks")

    # Build consensus task-level view
    df_task = consensus_rows(df_all)
    df_task = df_task[df_task["n_annotators"] >= args.min_annotators].copy()

    # Confirm AU12 >= 1.5 using the manifest's pre-merge mean_r (the authoritative value).
    # The raw CSV-computed AU12_r_mean can be lower because merged windows include
    # gap frames; the manifest already enforced AU12 >= 1.5 at extraction time.
    before = len(df_task)
    df_task = df_task[df_task["manifest_AU12_mean_r"] >= AU12_MIN].reset_index(drop=True)
    dropped = before - len(df_task)
    if dropped:
        print(f"  WARNING: {dropped} tasks dropped by manifest_AU12_mean_r < {AU12_MIN} (unexpected)")

    n_tasks = len(df_task)
    y = df_task["consensus_binary"].to_numpy(dtype=np.int32)
    weights = df_task["weight"].to_numpy(dtype=np.float64)

    label_counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  {n_tasks} tasks: {label_counts}  (1=smile, 0=not_a_smile)")

    au_r_mean_cols = sorted(c for c in df_task.columns
                            if c.startswith("AU") and c.endswith("_r_mean"))
    au_names = [c.replace("_r_mean", "") for c in au_r_mean_cols]
    print(f"  {len(au_names)} AUs: {au_names}\n")

    # ── 1. Univariate ─────────────────────────────────────────────────────
    print("1. Univariate sweep (mean_r, peak_r, mass_r, c_mean)...")
    all_results = univariate_sweep(df_task, y, weights)
    print(f"   {len(all_results)} result rows")

    # ── 2. Pairwise ──────────────────────────────────────────────────────
    print("2. Pairwise AU_r_mean sum scores...")
    pair_results = pairwise_sweep(df_task, y, weights, au_names)
    all_results.extend(pair_results)
    print(f"   {len(pair_results)} pairwise rows ({len(pair_results)//2} combos × 2 weightings)")

    # ── 3. Duchenne product / ratio ───────────────────────────────────────
    print("3. Duchenne product and ratio scores...")
    duch_results = duchenne_sweep(df_task, y, weights)
    all_results.extend(duch_results)

    # Report Duchenne additive (already in pairwise) vs product
    for r in all_results:
        if r["feature"] in ("AU06+AU12", "AU06*AU12_product") and r["weighting"] == "unweighted":
            print(f"   {r['feature']:30s} AUC={r['auc']:.4f}  F1={r['f1_at_best']:.4f}")

    # ── 4. Logistic regression ────────────────────────────────────────────
    print("4. Logistic regression (all AU_r_mean, 5-fold CV)...")
    logistic_results, coef_uw, coef_w = logistic_sweep(df_task, y, weights, au_r_mean_cols)
    all_results.extend(logistic_results)
    for r in logistic_results:
        print(f"   {r['weighting']:12s}: AUC={r['auc']:.4f}  F1={r['f1_at_best']:.4f}")

    top3_uw = sorted(coef_uw.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f"   Top-3 coefficients (unweighted): {[(k.replace('_r_mean',''), round(v,3)) for k,v in top3_uw]}")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_auc_ranking(all_results, FIG_DIR)
    fig_roc_univariate(all_results, FIG_DIR)
    fig_duchenne_vs_au12(all_results, FIG_DIR)
    fig_top5_combos(all_results, FIG_DIR)
    fig_logistic_coef(coef_uw, coef_w, FIG_DIR)
    print(f"  Figures → {FIG_DIR}")

    # ── Save logistic coefficients ────────────────────────────────────────
    for name, coef in [("unweighted", coef_uw), ("weighted", coef_w)]:
        if coef:
            coef_sorted = dict(sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True))
            path = OUT_DIR / f"logistic_coef_{name}.json"
            with open(path, "w") as f:
                json.dump(coef_sorted, f, indent=2)
            print(f"  Logistic coef ({name}) → {path}")

    # ── Write results CSV ─────────────────────────────────────────────────
    results_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                                for r in all_results])
    results_df = results_df.sort_values(
        ["weighting", "type", "auc"], ascending=[True, True, False]
    ).reset_index(drop=True)
    results_path = OUT_DIR / "au_roc_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults → {results_path} ({len(results_df):,} rows)")

    # ── Summary tables ────────────────────────────────────────────────────
    for w_mode in ["unweighted", "weighted"]:
        sub = results_df[results_df["weighting"] == w_mode]
        top10 = sub.sort_values("auc", ascending=False).head(10)
        print(f"\n── Top-10 by AUC ({w_mode}) ──")
        print(top10[["feature", "type", "auc", "f1_at_best", "best_threshold"]].to_string(index=False))


if __name__ == "__main__":
    main()
