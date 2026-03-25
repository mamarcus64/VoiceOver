"""
All-in-one experiment sweep for smile prediction.

Sweeps across:
  - Model:    mlp, gru, cnn
  - Modality: audio, eyegaze, both
  - Window:   various before/after configurations
  - Epochs:   configurable (default 100)

Produces a results table + plots saved to an output directory.

Usage:
    python -m analysis.smile_prediction.sweep [OPTIONS]

    --epochs     int   (default: 100)
    --folds      int   (default: 10, -1=LOO)
    --out-dir    path  (default: analysis/smile_prediction/sweep_results)
    --quick      run a minimal subset for smoke testing
"""

import argparse
import json
import time
import itertools
from pathlib import Path

import numpy as np

from .train import main as train_main

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Smile prediction experiment sweep")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--folds", type=int, default=10,
                   help="CV folds; -1 = LOO")
    p.add_argument("--out-dir", type=str,
                   default=str(SCRIPT_DIR / "sweep_results"))
    p.add_argument("--text-features", type=str, default=None,
                   help="Path to LLM features JSON. When provided, sweeps "
                        "with/without text features as an additional axis.")
    p.add_argument("--quick", action="store_true",
                   help="Minimal sweep for smoke testing (1 epoch, 3 folds)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

MODELS = ["mlp", "gru", "cnn"]
MODALITIES = ["audio", "eyegaze", "both"]

WINDOW_CONFIGS = [
    {"name": "before+during+after", "window_before": 30, "window_after": 30},
    {"name": "during_only",         "window_before": 0,  "window_after": 0},
    {"name": "before+during",       "window_before": 30, "window_after": 0},
    {"name": "during+after",        "window_before": 0,  "window_after": 30},
    {"name": "narrow_context",      "window_before": 10, "window_after": 10},
    {"name": "wide_context",        "window_before": 60, "window_after": 60},
]


def build_experiments(args) -> list[dict]:
    """Build the full grid of experiments to run."""
    if args.quick:
        models = ["mlp", "gru"]
        modalities = ["audio", "both"]
        windows = WINDOW_CONFIGS[:2]
    else:
        models = MODELS
        modalities = MODALITIES
        windows = WINDOW_CONFIGS

    # Text feature axis: if a features file is provided, sweep with/without
    text_opts = [None]
    if args.text_features:
        text_opts = [None, args.text_features]

    experiments = []
    for model, modality, win, text in itertools.product(
        models, modalities, windows, text_opts
    ):
        experiments.append({
            "model": model,
            "modality": modality,
            "window_name": win["name"],
            "window_before": win["window_before"],
            "window_after": win["window_after"],
            "text_features": text,
        })
    return experiments


def run_experiment(exp: dict, args) -> dict:
    """Run a single experiment and return its result row."""
    text_tag = "+text" if exp.get("text_features") else ""
    exp_name = f"{exp['model']}_{exp['modality']}_{exp['window_name']}{text_tag}"
    out_path = Path(args.out_dir) / f"{exp_name}.json"

    argv = [
        "--model", exp["model"],
        "--modality", exp["modality"],
        "--folds", str(args.folds),
        "--epochs", str(args.epochs),
        "--window-before", str(exp["window_before"]),
        "--window-after", str(exp["window_after"]),
        "--out", str(out_path),
        "--quiet",
    ]

    if exp.get("text_features"):
        argv.extend(["--text-features", exp["text_features"]])

    metrics = train_main(argv)

    return {
        "experiment": exp_name,
        "has_text": bool(exp.get("text_features")),
        **{k: v for k, v in exp.items() if k != "text_features"},
        "brier": metrics["brier_score"],
        "brier_skill": metrics["brier_skill"],
        "prior_brier": metrics["prior_brier"],
        "weighted_ce": metrics["weighted_ce"],
        "pred_entropy": metrics["pred_entropy"],
        "accuracy": metrics["accuracy"],
        "weighted_accuracy": metrics["weighted_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "classes_predicted": metrics["classes_predicted"],
        **{f"{c}_f1": metrics["per_class"][c]["f1"] for c in ["genuine", "polite", "masking"]},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(rows: list[dict], out_dir: Path):
    """Generate comparison plots from sweep results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # --- 1. Brier skill by modality ---
    _plot_grouped_bar(
        rows, group_key="modality", metric="brier_skill",
        title="Brier Skill Score by Modality (>0 = better than prior)",
        filename=fig_dir / "brier_skill_by_modality.png",
        filter_key="window_name", filter_val="before+during+after",
        center_zero=True,
    )

    # --- 2. Accuracy by modality ---
    _plot_grouped_bar(
        rows, group_key="modality", metric="accuracy",
        title="Accuracy by Modality",
        filename=fig_dir / "accuracy_by_modality.png",
        filter_key="window_name", filter_val="before+during+after",
    )

    # --- 3. Brier skill by window ---
    _plot_grouped_bar(
        rows, group_key="window_name", metric="brier_skill",
        title="Brier Skill Score by Time Window",
        filename=fig_dir / "brier_skill_by_window.png",
        filter_key="modality", filter_val="both",
        center_zero=True,
    )

    # --- 4. Accuracy by window ---
    _plot_grouped_bar(
        rows, group_key="window_name", metric="accuracy",
        title="Accuracy by Time Window",
        filename=fig_dir / "accuracy_by_window.png",
        filter_key="modality", filter_val="both",
    )

    # --- 5. Heatmaps ---
    _plot_heatmap(rows, row_key="model", col_key="modality", metric="brier_skill",
                  title="Brier Skill: Model × Modality (best window)",
                  filename=fig_dir / "heatmap_model_modality_skill.png")
    _plot_heatmap(rows, row_key="modality", col_key="window_name", metric="brier_skill",
                  title="Brier Skill: Modality × Window (best model)",
                  filename=fig_dir / "heatmap_modality_window_skill.png")

    # --- 6. Per-class F1 ---
    _plot_per_class_f1(rows, fig_dir)

    print(f"  Plots saved to {fig_dir}/")


def _plot_grouped_bar(rows, group_key, metric, title, filename,
                      filter_key=None, filter_val=None, center_zero=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filtered = rows
    if filter_key and filter_val:
        filtered = [r for r in rows if r.get(filter_key) == filter_val]
    if not filtered:
        filtered = rows

    models_present = sorted(set(r["model"] for r in filtered))
    groups = sorted(set(r[group_key] for r in filtered),
                    key=lambda g: g)

    x = np.arange(len(groups))
    width = 0.8 / max(len(models_present), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), 5))
    for mi, model in enumerate(models_present):
        vals = []
        for g in groups:
            matches = [r[metric] for r in filtered
                       if r["model"] == model and r[group_key] == g]
            vals.append(max(matches) if matches else 0)
        bars = ax.bar(x + mi * width, vals, width, label=model)
        for bar, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            offset = 0.005 if v >= 0 else -0.005
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{v:.2f}", ha="center", va=va, fontsize=7)

    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models_present) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    ax.legend()
    if center_zero:
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
        ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]), 0.05)
        ax.set_ylim(-ylim * 1.15, ylim * 1.15)
    else:
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _plot_heatmap(rows, row_key, col_key, metric, title, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_vals = sorted(set(r[row_key] for r in rows))
    col_vals = sorted(set(r[col_key] for r in rows))

    matrix = np.zeros((len(row_vals), len(col_vals)))
    for ri, rv in enumerate(row_vals):
        for ci, cv in enumerate(col_vals):
            matches = [r[metric] for r in rows if r[row_key] == rv and r[col_key] == cv]
            matrix[ri, ci] = max(matches) if matches else 0

    fig, ax = plt.subplots(figsize=(max(6, len(col_vals) * 1.2), max(4, len(row_vals) * 0.8)))
    has_neg = matrix.min() < -0.001
    if has_neg:
        vlim = max(abs(matrix.min()), abs(matrix.max()), 0.01)
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vlim, vmax=vlim)
    else:
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0,
                       vmax=max(0.01, matrix.max() * 1.1))
    ax.set_xticks(range(len(col_vals)))
    ax.set_xticklabels(col_vals, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_vals)))
    ax.set_yticklabels(row_vals, fontsize=9)
    for ri in range(len(row_vals)):
        for ci in range(len(col_vals)):
            ax.text(ci, ri, f"{matrix[ri, ci]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _plot_per_class_f1(rows, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    default_win = [r for r in rows if r["window_name"] == "before+during+after"]
    if not default_win:
        default_win = rows

    modalities = sorted(set(r["modality"] for r in default_win))
    models_present = sorted(set(r["model"] for r in default_win))
    classes = ["genuine", "polite", "masking"]

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 5), sharey=True)
    if len(classes) == 1:
        axes = [axes]

    for ci, cls in enumerate(classes):
        ax = axes[ci]
        x = np.arange(len(modalities))
        width = 0.8 / max(len(models_present), 1)

        for mi, model in enumerate(models_present):
            vals = []
            for mod in modalities:
                matches = [r[f"{cls}_f1"] for r in default_win
                           if r["model"] == model and r["modality"] == mod]
                vals.append(max(matches) if matches else 0)
            ax.bar(x + mi * width, vals, width, label=model)

        ax.set_title(f"{cls} F1")
        ax.set_xticks(x + width * (len(models_present) - 1) / 2)
        ax.set_xticklabels(modalities, fontsize=9)
        ax.set_ylim(0, 1.05)
        if ci == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Per-class F1 by Modality (default window)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_class_f1.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict]):
    """Print a formatted results table."""
    header = (f"{'Experiment':<42s} {'Brier':>6s} {'Skill':>6s} "
              f"{'Acc':>5s} {'MF1':>5s} {'Ent':>5s} {'Cls':>3s}  "
              f"{'Gen':>4s} {'Pol':>4s} {'Msk':>4s}")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: x["brier"]):
        print(f"{r['experiment']:<42s} "
              f"{r['brier']:6.4f} {r['brier_skill']:+6.3f} "
              f"{r['accuracy']:5.3f} {r['macro_f1']:5.3f} "
              f"{r['pred_entropy']:5.3f} {r['classes_predicted']:3d}  "
              f"{r['genuine_f1']:4.2f} {r['polite_f1']:4.2f} {r['masking_f1']:4.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.quick:
        args.epochs = 1
        args.folds = 3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(args)
    fold_desc = "LOO" if args.folds == -1 else f"{args.folds}-fold"
    print(f"=== Smile Prediction Sweep ===")
    print(f"Experiments: {len(experiments)}, Epochs: {args.epochs}, CV: {fold_desc}")
    print(f"Output: {out_dir}")
    if args.quick:
        print("(quick mode — minimal epochs and folds)")
    print()

    all_rows = []
    t0 = time.time()
    for i, exp in enumerate(experiments):
        exp_name = f"{exp['model']}_{exp['modality']}_{exp['window_name']}"
        print(f"[{i + 1}/{len(experiments)}] {exp_name} ...", end=" ", flush=True)
        et = time.time()
        row = run_experiment(exp, args)
        all_rows.append(row)
        print(f"brier={row['brier']:.4f} skill={row['brier_skill']:+.3f} "
              f"acc={row['accuracy']:.3f} cls={row['classes_predicted']}/3 "
              f"({time.time() - et:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nAll experiments done in {elapsed:.0f}s")
    print()

    print_summary_table(all_rows)

    # Save combined results
    summary_path = out_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Plot
    print("\nGenerating plots...")
    plot_results(all_rows, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
