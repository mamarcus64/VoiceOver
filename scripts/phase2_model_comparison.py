#!/usr/bin/env python3
"""Phase 2: Cross-validated model comparison for smile detection.

Models (feature sets):
  A) AU12 threshold only (1 feature)
  B) 17-AU logistic (current model, 17 features)
  C) Top AU z-scores + temporal deltas (9 features)
  D) C + keywords (kw_positive, kw_negative, kw_valence_ratio)  — 12 features
  E) Full: C + keywords + all non-trivial features               — ~20 features

Each model is tested with 4 audio/gaze combinations:
  (none, +audio, +gaze, +audio+gaze)

20-fold stratified CV, L2-regularized logistic regression with C swept over
a grid. Reports: mean CV AUC, F1, precision, recall; plus aggregate PR curve.

Outputs:
  data/phase2_results.csv           — per-config summary
  data/phase2_pr_curves.json        — PR curve data for all configs
  data/phase2_comparison_chart.png  — grouped bar chart
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_PATH = DATA_DIR / "labeled_segment_features.csv"

N_FOLDS = 20
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# ── Feature groups ─────────────────────────────────────────────────────────────

AU_NAMES = [f"AU{n:02d}_r" for n in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]]

AUDIO_FEATURES = [
    "audio_during_valence", "audio_during_arousal", "audio_during_dominance",
    "audio_before10s_valence", "audio_before10s_arousal", "audio_before10s_dominance",
    "audio_after10s_valence", "audio_after10s_arousal", "audio_after10s_dominance",
    "audio_delta_before_arousal", "audio_delta_after_arousal",
    "audio_during_coverage",
]

GAZE_FEATURES = [
    "gaze_during_valence", "gaze_during_arousal", "gaze_during_dominance",
    "gaze_before10s_valence", "gaze_before10s_arousal", "gaze_before10s_dominance",
    "gaze_after10s_valence", "gaze_after10s_arousal", "gaze_after10s_dominance",
    "gaze_delta_before_valence", "gaze_delta_after_valence",
    "gaze_delta_before_dominance", "gaze_delta_after_dominance",
]

KEYWORD_FEATURES = ["kw_positive", "kw_negative", "kw_valence_ratio"]

# Base model feature sets (before adding audio/gaze)
MODELS: dict[str, list[str]] = {
    "A: AU12 only": [
        "au_during_AU12_r",
    ],
    "B: 17-AU (current)": [
        f"au_during_{au}" for au in AU_NAMES
    ],
    "C: z-scores + deltas": [
        "au_zscore_AU06_r", "au_zscore_AU12_r", "au_zscore_AU10_r",
        "au_zscore_AU07_r", "au_zscore_AU25_r",
        "au_delta_before_AU06_r", "au_delta_before_AU12_r",
        "au_delta_after_AU06_r", "au_delta_after_AU12_r",
    ],
    "D: C + keywords": [
        "au_zscore_AU06_r", "au_zscore_AU12_r", "au_zscore_AU10_r",
        "au_zscore_AU07_r", "au_zscore_AU25_r",
        "au_delta_before_AU06_r", "au_delta_before_AU12_r",
        "au_delta_after_AU06_r", "au_delta_after_AU12_r",
    ] + KEYWORD_FEATURES,
    "E: full feature set": [
        f"au_zscore_{au}" for au in AU_NAMES
    ] + [
        f"au_delta_before_{au}" for au in ["AU06_r", "AU12_r", "AU10_r", "AU07_r", "AU25_r"]
    ] + [
        f"au_delta_after_{au}" for au in ["AU06_r", "AU12_r", "AU10_r", "AU07_r", "AU25_r"]
    ] + KEYWORD_FEATURES,
}

AUDIO_GAZE_COMBOS = [
    ("none", []),
    ("+audio", AUDIO_FEATURES),
    ("+gaze", GAZE_FEATURES),
    ("+audio+gaze", AUDIO_FEATURES + GAZE_FEATURES),
]


def evaluate_config(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    config_name: str,
) -> dict:
    """Run N_FOLDS stratified CV with inner C selection, return metrics."""
    # Drop rows with NaN in any feature column
    valid_mask = df[feature_cols].notna().all(axis=1).values
    X = df.loc[valid_mask, feature_cols].values.astype(np.float64)
    y_valid = y[valid_mask]
    n_valid = int(valid_mask.sum())

    if n_valid < N_FOLDS * 2:
        return {
            "config": config_name,
            "n_features": len(feature_cols),
            "n_valid": n_valid,
            "auc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "best_C": float("nan"),
        }

    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_aucs = []
    fold_f1s = []
    fold_precs = []
    fold_recs = []
    all_y_true = []
    all_y_prob = []
    best_Cs = []

    for train_idx, test_idx in outer_cv.split(X, y_valid):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_valid[train_idx], y_valid[test_idx]

        # Inner CV for C selection (5-fold)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        best_C = 1.0
        best_inner_auc = -1.0

        for C in C_GRID:
            inner_aucs = []
            for tr_i, val_i in inner_cv.split(X_train, y_train):
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
                ])
                pipe.fit(X_train[tr_i], y_train[tr_i])
                prob = pipe.predict_proba(X_train[val_i])[:, 1]
                try:
                    inner_aucs.append(roc_auc_score(y_train[val_i], prob))
                except ValueError:
                    pass
            if inner_aucs:
                mean_auc = np.mean(inner_aucs)
                if mean_auc > best_inner_auc:
                    best_inner_auc = mean_auc
                    best_C = C

        best_Cs.append(best_C)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")),
        ])
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        try:
            fold_aucs.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            pass
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))
        fold_precs.append(precision_score(y_test, y_pred, zero_division=0))
        fold_recs.append(recall_score(y_test, y_pred, zero_division=0))

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

    # Aggregate PR curve from pooled OOF predictions
    pr_prec, pr_rec, pr_thr = precision_recall_curve(
        np.array(all_y_true), np.array(all_y_prob)
    )

    return {
        "config": config_name,
        "n_features": len(feature_cols),
        "n_valid": n_valid,
        "auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else float("nan"),
        "auc_std": round(float(np.std(fold_aucs)), 4) if fold_aucs else float("nan"),
        "f1": round(float(np.mean(fold_f1s)), 4),
        "f1_std": round(float(np.std(fold_f1s)), 4),
        "precision": round(float(np.mean(fold_precs)), 4),
        "precision_std": round(float(np.std(fold_precs)), 4),
        "recall": round(float(np.mean(fold_recs)), 4),
        "recall_std": round(float(np.std(fold_recs)), 4),
        "best_C": round(float(np.median(best_Cs)), 4),
        "pr_curve_precision": [round(float(v), 4) for v in pr_prec.tolist()],
        "pr_curve_recall": [round(float(v), 4) for v in pr_rec.tolist()],
        "pr_curve_thresholds": [round(float(v), 4) for v in pr_thr.tolist()],
    }


def main():
    df = pd.read_csv(FEATURES_PATH)
    y = df["label_binary"].values
    print(f"Loaded {len(df)} segments: {y.sum()} smiles, {len(y) - y.sum()} not-smiles")
    print(f"{N_FOLDS}-fold stratified CV with nested C selection over {len(C_GRID)} values\n")

    results = []
    pr_curves = {}

    for model_name, base_features in MODELS.items():
        for combo_name, combo_features in AUDIO_GAZE_COMBOS:
            config_name = f"{model_name} {combo_name}"
            all_features = base_features + combo_features

            print(f"  {config_name:<45} ({len(all_features)} features) ... ", end="", flush=True)
            result = evaluate_config(df, y, all_features, config_name)
            print(f"AUC={result['auc']:.4f}  F1={result['f1']:.4f}  P={result['precision']:.4f}  R={result['recall']:.4f}  (n={result['n_valid']}, C={result['best_C']})")
            results.append(result)

            if "pr_curve_precision" in result:
                pr_curves[config_name] = {
                    "precision": result.pop("pr_curve_precision"),
                    "recall": result.pop("pr_curve_recall"),
                    "thresholds": result.pop("pr_curve_thresholds"),
                }

    # ── Save results ──
    results_df = pd.DataFrame(results)
    results_path = DATA_DIR / "phase2_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved → {results_path}")

    pr_path = DATA_DIR / "phase2_pr_curves.json"
    with open(pr_path, "w") as f:
        json.dump(pr_curves, f)
    print(f"Saved → {pr_path}")

    # ── Console summary table ──
    print("\n" + "=" * 110)
    print(f"{'Config':<45} {'Feat':>4} {'N':>5} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'C':>6}")
    print("-" * 110)
    for r in sorted(results, key=lambda x: -x.get("auc", 0)):
        print(
            f"{r['config']:<45} {r['n_features']:4d} {r['n_valid']:5d} "
            f"{r['auc']:7.4f} {r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} "
            f"{r['best_C']:6.3f}"
        )

    # ── Chart: grouped bar chart by model, colored by audio/gaze combo ──
    model_names = list(MODELS.keys())
    combo_names = [c[0] for c in AUDIO_GAZE_COMBOS]
    combo_colors = ["#64748b", "#34d399", "#f472b6", "#a78bfa"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0f172a")

    for ax_idx, (metric, metric_label) in enumerate([("auc", "ROC AUC"), ("f1", "F1 Score")]):
        ax = axes[ax_idx]
        ax.set_facecolor("#0f172a")

        x = np.arange(len(model_names))
        width = 0.18
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

        for ci, (combo_name, color) in enumerate(zip(combo_names, combo_colors)):
            vals = []
            errs = []
            for model_name in model_names:
                config = f"{model_name} {combo_name}"
                r = next((r for r in results if r["config"] == config), None)
                vals.append(r[metric] if r and not np.isnan(r.get(metric, float("nan"))) else 0)
                errs.append(r.get(f"{metric}_std", 0) if r else 0)

            bars = ax.bar(
                x + offsets[ci], vals, width,
                yerr=errs, capsize=2,
                label=combo_name, color=color, edgecolor="#1e293b",
                alpha=0.85, error_kw={"ecolor": "#475569", "linewidth": 0.8},
            )
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6, color="#94a3b8", rotation=90,
                    )

        short_labels = [n.split(":")[0] for n in model_names]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9, color="#cbd5e1")
        ax.set_ylabel(metric_label, fontsize=10, color="#94a3b8")
        ax.set_title(f"{metric_label} by Model × Audio/Gaze", fontsize=12,
                      color="#f8fafc", fontweight="bold")
        ax.tick_params(axis="y", colors="#94a3b8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#334155")
        ax.spines["left"].set_color("#334155")
        ax.set_ylim(0.4, 1.0)
        ax.axhline(y=0.5, color="#334155", linewidth=0.5, linestyle="--")

        if ax_idx == 1:
            legend = ax.legend(fontsize=8, loc="lower right", framealpha=0.8,
                               edgecolor="#334155", facecolor="#1e293b", labelcolor="#cbd5e1")
            legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    chart_path = DATA_DIR / "phase2_comparison_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved chart → {chart_path}")

    # ── PR curve chart for best configs ──
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fig2.patch.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    # Plot PR curves for "none" variant of each model + best overall
    pr_colors = ["#64748b", "#3b82f6", "#818cf8", "#fbbf24", "#f472b6"]
    best_config = max(results, key=lambda r: r.get("f1", 0))["config"]

    plotted = set()
    for i, model_name in enumerate(model_names):
        config = f"{model_name} none"
        if config in pr_curves:
            prc = pr_curves[config]
            ax2.plot(prc["recall"], prc["precision"],
                     color=pr_colors[i % len(pr_colors)], linewidth=1.5,
                     label=f"{model_name.split(':')[0]}: none", alpha=0.8)
            plotted.add(config)

    if best_config not in plotted and best_config in pr_curves:
        prc = pr_curves[best_config]
        ax2.plot(prc["recall"], prc["precision"],
                 color="#22c55e", linewidth=2.5, linestyle="--",
                 label=f"BEST: {best_config}")

    ax2.set_xlabel("Recall", fontsize=11, color="#94a3b8")
    ax2.set_ylabel("Precision", fontsize=11, color="#94a3b8")
    ax2.set_title("Precision–Recall Curves (pooled out-of-fold predictions)",
                   fontsize=13, color="#f8fafc", fontweight="bold")
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(colors="#94a3b8")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_color("#334155")
    ax2.spines["left"].set_color("#334155")
    legend2 = ax2.legend(fontsize=8, loc="lower left", framealpha=0.8,
                          edgecolor="#334155", facecolor="#1e293b", labelcolor="#cbd5e1")
    legend2.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    pr_chart_path = DATA_DIR / "phase2_pr_chart.png"
    plt.savefig(pr_chart_path, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    print(f"Saved PR chart → {pr_chart_path}")


if __name__ == "__main__":
    main()
