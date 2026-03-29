#!/usr/bin/env python3
"""Phase 1: Univariate AUC sanity check for all extracted features.

Reads data/labeled_segment_features.csv and computes ROC-AUC for each numeric
feature column independently against the binary smile label.  Produces:
  - Console table of all AUCs sorted descending
  - data/phase1_auc_results.csv   — machine-readable results
  - data/phase1_auc_chart.png     — bar chart of top features
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_PATH = DATA_DIR / "labeled_segment_features.csv"

SKIP_COLS = {
    "video_id", "segment_start", "segment_end", "label", "label_binary",
    "source", "n_annotators", "manifest_logistic_score", "manifest_mean_r",
    "manifest_peak_r", "manifest_bin", "kw_raw",
    # frame counts are metadata, not predictive features
    "au_before10s_n_frames", "au_during_n_frames", "au_after10s_n_frames",
    "gaze_before10s_n", "gaze_during_n", "gaze_after10s_n",
    "audio_before10s_n_segs", "audio_during_n_segs", "audio_after10s_n_segs",
}

# Human-readable feature group labels for the chart
def feature_group(col: str) -> str:
    if col.startswith("au_zscore"):
        return "AU z-score"
    if col.startswith("au_delta_before"):
        return "AU Δ (vs before)"
    if col.startswith("au_delta_after"):
        return "AU Δ (vs after)"
    if col.startswith("au_during"):
        return "AU during"
    if col.startswith("au_before"):
        return "AU before 10s"
    if col.startswith("au_after"):
        return "AU after 10s"
    if col.startswith("gaze_delta"):
        return "Gaze Δ"
    if col.startswith("gaze_"):
        return "Gaze VAD"
    if col.startswith("audio_delta"):
        return "Audio Δ"
    if col.startswith("audio_") and "coverage" in col:
        return "Audio coverage"
    if col.startswith("audio_"):
        return "Audio VAD"
    if col.startswith("kw_"):
        return "Keywords"
    return "Other"


GROUP_COLORS = {
    "AU during": "#3b82f6",
    "AU z-score": "#818cf8",
    "AU Δ (vs before)": "#a78bfa",
    "AU Δ (vs after)": "#c4b5fd",
    "AU before 10s": "#94a3b8",
    "AU after 10s": "#64748b",
    "Gaze VAD": "#f472b6",
    "Gaze Δ": "#f9a8d4",
    "Audio VAD": "#34d399",
    "Audio Δ": "#6ee7b7",
    "Audio coverage": "#a7f3d0",
    "Keywords": "#fbbf24",
    "Other": "#475569",
}


def short_name(col: str) -> str:
    """Shorten column name for chart readability."""
    return (col
            .replace("au_zscore_", "z·")
            .replace("au_delta_before_", "Δb·")
            .replace("au_delta_after_", "Δa·")
            .replace("au_during_", "dur·")
            .replace("au_before10s_", "b10·")
            .replace("au_after10s_", "a10·")
            .replace("gaze_delta_before_", "gΔb·")
            .replace("gaze_delta_after_", "gΔa·")
            .replace("gaze_before10s_", "gb·")
            .replace("gaze_during_", "gd·")
            .replace("gaze_after10s_", "ga·")
            .replace("audio_delta_before_", "aΔb·")
            .replace("audio_delta_after_", "aΔa·")
            .replace("audio_before10s_", "ab·")
            .replace("audio_during_", "ad·")
            .replace("audio_after10s_", "aa·")
            .replace("kw_", "kw·")
            .replace("_r", ""))


def main():
    df = pd.read_csv(FEATURES_PATH)
    y = df["label_binary"].values
    print(f"Loaded {len(df)} segments: {y.sum()} smiles, {len(y) - y.sum()} not-smiles")
    print(f"Total feature columns: {len(df.columns) - len(SKIP_COLS)}\n")

    results = []
    for col in df.columns:
        if col in SKIP_COLS:
            continue
        vals = df[col].values
        valid = ~np.isnan(vals.astype(float))
        n_valid = int(valid.sum())
        if n_valid < 50:
            continue

        x = vals[valid].astype(float)
        y_sub = y[valid]

        if len(np.unique(x)) < 2:
            continue

        try:
            auc = roc_auc_score(y_sub, x)
        except ValueError:
            continue

        # AUC < 0.5 means the feature is negatively correlated; flip to show absolute discriminative power
        auc_abs = max(auc, 1 - auc)
        direction = "+" if auc >= 0.5 else "−"

        results.append({
            "feature": col,
            "group": feature_group(col),
            "auc_raw": round(auc, 4),
            "auc_abs": round(auc_abs, 4),
            "direction": direction,
            "n_valid": n_valid,
            "n_total": len(df),
            "coverage_pct": round(n_valid / len(df) * 100, 1),
        })

    results.sort(key=lambda r: -r["auc_abs"])

    # Console output
    print(f"{'Rank':>4}  {'AUC':>6}  {'Dir':>3}  {'N':>5}  {'Cov%':>5}  {'Group':<20}  Feature")
    print("-" * 100)
    for i, r in enumerate(results, 1):
        print(f"{i:4d}  {r['auc_abs']:.4f}  {r['direction']:>3}  {r['n_valid']:5d}  {r['coverage_pct']:5.1f}  {r['group']:<20}  {r['feature']}")

    # Save CSV
    results_df = pd.DataFrame(results)
    results_path = DATA_DIR / "phase1_auc_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results → {results_path}")

    # ── Chart: top 50 features ──
    top = results[:50]
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    y_pos = np.arange(len(top))
    bars = ax.barh(
        y_pos,
        [r["auc_abs"] for r in top],
        color=[GROUP_COLORS.get(r["group"], "#475569") for r in top],
        edgecolor="#1e293b",
        height=0.7,
    )

    # AUC value labels
    for bar, r in zip(bars, top):
        w = bar.get_width()
        ax.text(
            w + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{r["auc_abs"]:.3f} {r["direction"]}',
            va="center", fontsize=7.5, color="#e2e8f0",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([short_name(r["feature"]) for r in top], fontsize=8, color="#cbd5e1")
    ax.invert_yaxis()
    ax.set_xlim(0.45, max(r["auc_abs"] for r in top) + 0.04)
    ax.axvline(x=0.5, color="#475569", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Univariate AUC (|AUC|, chance = 0.5)", fontsize=10, color="#94a3b8")
    ax.set_title("Phase 1: Univariate Feature AUC for Smile Detection", fontsize=13,
                  color="#f8fafc", fontweight="bold", pad=12)
    ax.tick_params(axis="x", colors="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")

    # Legend for groups
    seen_groups = []
    for r in top:
        if r["group"] not in seen_groups:
            seen_groups.append(r["group"])
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=GROUP_COLORS.get(g, "#475569"))
        for g in seen_groups
    ]
    legend = ax.legend(legend_handles, seen_groups, loc="lower right", fontsize=7.5,
                       framealpha=0.8, edgecolor="#334155", facecolor="#1e293b",
                       labelcolor="#cbd5e1")
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    chart_path = DATA_DIR / "phase1_auc_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved chart → {chart_path}")

    # ── Summary by group ──
    print("\n=== Best AUC per feature group ===")
    group_best: dict[str, dict] = {}
    for r in results:
        g = r["group"]
        if g not in group_best or r["auc_abs"] > group_best[g]["auc_abs"]:
            group_best[g] = r
    for g, r in sorted(group_best.items(), key=lambda x: -x[1]["auc_abs"]):
        print(f"  {g:<22}  AUC={r['auc_abs']:.4f}  {r['feature']}")

    # ── Quick check: does z-score beat raw? ──
    print("\n=== AU12 comparison ===")
    for r in results:
        if "AU12" in r["feature"]:
            print(f"  {r['feature']:<40}  AUC={r['auc_abs']:.4f}  dir={r['direction']}")


if __name__ == "__main__":
    main()
