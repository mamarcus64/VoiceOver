"""
Cross-validated training and evaluation for smile prediction.

Usage:
    python -m analysis.smile_prediction.train [OPTIONS]

Key options:
    --model          mlp | gru | cnn                   (default: mlp)
    --modality       both | audio | eyegaze            (default: both)
    --folds          int  k-fold CV; -1 = LOO          (default: 10)
    --epochs         int                                (default: 100)
    --entropy-reg    float  anti-collapse strength      (default: 0.1)
    --label-smoothing float  for single-annotator       (default: 0.1)
    --quiet          suppress per-fold progress output
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from .dataset import (
    SmileTask, build_tasks, extract_features, extract_aggregated_features,
    SMILE_CLASSES, Modality, agg_feature_dim, seq_feature_dim,
    load_llm_features, attach_llm_features, N_LLM_FEATURES,
)
from .models import (
    AggregatedMLP, GRUClassifier, CNNClassifier,
    soft_label_loss, compute_class_weights,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Smile prediction cross-validation")
    p.add_argument("--model", choices=["mlp", "gru", "cnn"], default="mlp")
    p.add_argument("--modality", choices=["both", "audio", "eyegaze"], default="both")
    p.add_argument("--folds", type=int, default=10,
                   help="Number of CV folds. -1 = leave-one-out.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--window-before", type=float, default=30.0)
    p.add_argument("--window-after", type=float, default=30.0)
    p.add_argument("--grid-step", type=float, default=3.0)
    p.add_argument("--min-annotators", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--entropy-reg", type=float, default=0.1,
                   help="Entropy regularization to prevent class collapse")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label smoothing for single-annotator hard labels")
    p.add_argument("--text-features", type=str, default=None,
                   help="Path to LLM features JSON (from llm_features.py). "
                        "Appends 10-dim text features to VAD features.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results.json")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_all_data(tasks: list[SmileTask], args) -> list[dict]:
    use_agg = args.model == "mlp"
    modality: Modality = args.modality
    data = []
    skipped = 0
    for task in tasks:
        if use_agg:
            item = extract_aggregated_features(
                task, args.window_before, args.window_after, args.grid_step, modality
            )
        else:
            item = extract_features(
                task, args.window_before, args.window_after, args.grid_step, modality
            )
        if item is None:
            skipped += 1
            continue
        data.append(item)
    if skipped and not getattr(args, 'quiet', False):
        print(f"  Skipped {skipped} tasks (no features)")
    return data


def collate_sequences(items: list[dict]) -> tuple:
    sequences = [item["sequence"] for item in items]
    labels = np.stack([item["soft_label"] for item in items])
    weights = np.array([item["weight"] for item in items], dtype=np.float32)

    lengths = np.array([len(s) for s in sequences])
    max_len = max(lengths)
    feat_dim = sequences[0].shape[1]

    padded = np.zeros((len(sequences), max_len, feat_dim), dtype=np.float32)
    for i, s in enumerate(sequences):
        padded[i, :len(s)] = s

    return (
        torch.from_numpy(padded),
        torch.from_numpy(lengths.astype(np.int64)),
        torch.from_numpy(labels),
        torch.from_numpy(weights),
    )


def collate_aggregated(items: list[dict]) -> tuple:
    features = np.stack([item["features"] for item in items])
    labels = np.stack([item["soft_label"] for item in items])
    weights = np.array([item["weight"] for item in items], dtype=np.float32)
    return (
        torch.from_numpy(features),
        None,
        torch.from_numpy(labels),
        torch.from_numpy(weights),
    )


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_one_fold(
    train_data: list[dict],
    test_data: list[dict],
    args,
) -> list[dict]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_seq = args.model in ("gru", "cnn")

    if is_seq:
        X_train, L_train, Y_train, W_train = collate_sequences(train_data)
        X_test, L_test, Y_test, W_test = collate_sequences(test_data)
    else:
        X_train, _, Y_train, W_train = collate_aggregated(train_data)
        X_test, _, Y_test, W_test = collate_aggregated(test_data)

    # Class balancing from training label distribution
    class_weights = compute_class_weights(Y_train)

    if args.model == "mlp":
        model = AggregatedMLP(
            input_dim=X_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout
        )
    elif args.model == "gru":
        model = GRUClassifier(
            input_dim=X_train.shape[2], hidden_dim=args.hidden_dim, dropout=args.dropout
        )
    elif args.model == "cnn":
        model = CNNClassifier(
            input_dim=X_train.shape[2], channels=args.hidden_dim, dropout=args.dropout
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for _epoch in range(args.epochs):
        optimizer.zero_grad()
        if is_seq:
            logits = model(X_train, L_train)
        else:
            logits = model(X_train)
        loss = soft_label_loss(
            logits, Y_train, W_train,
            class_weights=class_weights,
            entropy_reg=args.entropy_reg,
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        if is_seq:
            logits = model(X_test, L_test)
        else:
            logits = model(X_test)
        probs = torch.softmax(logits, dim=-1).numpy()

    results = []
    for i, item in enumerate(test_data):
        p = probs[i]
        results.append({
            "task_number": item["task_number"],
            "predicted_probs": p.tolist(),
            "predicted_class": SMILE_CLASSES[int(p.argmax())],
            "true_soft_label": item["soft_label"].tolist(),
            "true_top_class": SMILE_CLASSES[int(item["soft_label"].argmax())],
            "weight": float(item["weight"]),
        })
    return results


# ---------------------------------------------------------------------------
# Metrics — distributional + hard
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict], prior_dist: np.ndarray | None = None) -> dict:
    """
    Compute metrics from CV fold results.

    Distributional metrics (primary — evaluate the predicted distribution):
        - brier_score: mean squared error between predicted and true distributions
        - weighted_ce: weighted cross-entropy (soft labels)

    Hard metrics (secondary — evaluate the argmax prediction):
        - accuracy, weighted_accuracy, macro_f1, per-class P/R/F1

    Baselines:
        - prior_brier / prior_ce: what a "predict the class prior" model achieves
    """
    n = len(results)
    if n == 0:
        return {}

    all_pred = np.array([r["predicted_probs"] for r in results])
    all_true = np.array([r["true_soft_label"] for r in results])
    all_weights = np.array([r["weight"] for r in results])

    # --- Distributional metrics ---
    # Brier score: mean over samples of sum_c (pred_c - true_c)^2
    brier_per = ((all_pred - all_true) ** 2).sum(axis=1)
    brier_score = float(np.average(brier_per, weights=all_weights))

    # Weighted CE
    pred_clipped = np.clip(all_pred, 1e-8, 1.0)
    ce_per = -(all_true * np.log(pred_clipped)).sum(axis=1)
    weighted_ce = float(np.average(ce_per, weights=all_weights))

    # --- Baselines (predict class prior) ---
    if prior_dist is None:
        prior_dist = all_true.mean(axis=0)
    prior_tiled = np.tile(prior_dist, (n, 1))
    prior_brier = float(np.average(
        ((prior_tiled - all_true) ** 2).sum(axis=1), weights=all_weights
    ))
    prior_clipped = np.clip(prior_tiled, 1e-8, 1.0)
    prior_ce = float(np.average(
        -(all_true * np.log(prior_clipped)).sum(axis=1), weights=all_weights
    ))

    # --- Hard metrics ---
    correct = 0
    weighted_correct = 0.0
    total_weight = 0.0
    per_class_tp = {c: 0 for c in SMILE_CLASSES}
    per_class_fp = {c: 0 for c in SMILE_CLASSES}
    per_class_fn = {c: 0 for c in SMILE_CLASSES}

    classes_predicted = set()

    for r in results:
        pred_cls = r["predicted_class"]
        true_cls = r["true_top_class"]
        w = r["weight"]
        classes_predicted.add(pred_cls)

        if pred_cls == true_cls:
            correct += 1
            weighted_correct += w
        total_weight += w

        per_class_tp[pred_cls] += int(pred_cls == true_cls)
        if pred_cls != true_cls:
            per_class_fp[pred_cls] += 1
            per_class_fn[true_cls] += 1

    per_class_metrics = {}
    for c in SMILE_CLASSES:
        tp = per_class_tp[c]
        fp = per_class_fp[c]
        fn = per_class_fn[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_metrics[c] = {"precision": prec, "recall": rec, "f1": f1,
                                "support": tp + fn}

    macro_f1 = float(np.mean([m["f1"] for m in per_class_metrics.values()]))

    # Prediction entropy (how spread out are model predictions?)
    pred_entropy = float(-(all_pred * np.log(np.clip(all_pred, 1e-8, 1.0))).sum(axis=1).mean())

    return {
        "n_samples": n,
        # Distributional (primary)
        "brier_score": brier_score,
        "weighted_ce": weighted_ce,
        "pred_entropy": pred_entropy,
        # Baselines
        "prior_brier": prior_brier,
        "prior_ce": prior_ce,
        "brier_skill": 1 - brier_score / prior_brier if prior_brier > 0 else 0,
        # Hard (secondary)
        "accuracy": correct / n if n > 0 else 0,
        "weighted_accuracy": weighted_correct / total_weight if total_weight > 0 else 0,
        "macro_f1": macro_f1,
        "classes_predicted": len(classes_predicted),
        "per_class": per_class_metrics,
    }


# ---------------------------------------------------------------------------
# Cross-validation driver
# ---------------------------------------------------------------------------

def run_cv(all_data: list[dict], args) -> list[dict]:
    n = len(all_data)
    indices = np.arange(n)

    if args.folds == -1:
        folds = [(np.delete(indices, i), np.array([i])) for i in range(n)]
        fold_label = "LOO"
    else:
        k = min(args.folds, n)
        kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
        folds = list(kf.split(indices))
        fold_label = f"{k}-fold"

    if not args.quiet:
        print(f"Running {fold_label} CV ({len(folds)} folds)...")

    all_results = []
    t0 = time.time()
    for fi, (train_idx, test_idx) in enumerate(folds):
        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]

        fold_results = train_one_fold(train_data, test_data, args)
        all_results.extend(fold_results)

        if not args.quiet and ((fi + 1) % max(1, len(folds) // 10) == 0 or (fi + 1) == len(folds)):
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed if elapsed > 0 else float('inf')
            eta = (len(folds) - fi - 1) / rate if rate > 0 else 0
            print(f"  Fold {fi + 1}/{len(folds)}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    args = parse_args(argv)

    fold_desc = "LOO" if args.folds == -1 else f"{args.folds}-fold"
    if not args.quiet:
        print(f"=== Smile Prediction {fold_desc} CV ===")
        print(f"Model: {args.model}, Modality: {args.modality}, "
              f"Epochs: {args.epochs}, LR: {args.lr}")
        print(f"Window: [{args.window_before}s before, {args.window_after}s after], "
              f"Grid: {args.grid_step}s")
        print(f"Entropy reg: {args.entropy_reg}, Label smoothing: {args.label_smoothing}")
        print(f"Min annotators: {args.min_annotators}")
        print()

    tasks = build_tasks(
        min_annotators=args.min_annotators,
        label_smoothing=args.label_smoothing,
    )
    if not args.quiet:
        print(f"Loaded {len(tasks)} tasks with annotations")
        print("Extracting features...")

    t0 = time.time()
    all_data = prepare_all_data(tasks, args)
    if not args.quiet:
        print(f"  {len(all_data)} samples ready ({time.time() - t0:.1f}s)")

    # Attach LLM text features if provided
    if args.text_features:
        from pathlib import Path as _Path
        llm_feats = load_llm_features(_Path(args.text_features))
        n_before = len(all_data)
        all_data = attach_llm_features(all_data, llm_feats)
        if not args.quiet:
            print(f"  + LLM text features: {len(all_data)}/{n_before} matched "
                  f"(+{N_LLM_FEATURES} dims)")

    if not args.quiet:
        print()

    if len(all_data) < 2:
        print("ERROR: Need at least 2 samples for CV")
        sys.exit(1)

    results = run_cv(all_data, args)
    metrics = compute_metrics(results)

    if not args.quiet:
        print()
        print("=== Results ===")
        print(f"  Brier score:       {metrics['brier_score']:.4f}  "
              f"(prior baseline: {metrics['prior_brier']:.4f}, "
              f"skill: {metrics['brier_skill']:+.3f})")
        print(f"  Weighted CE:       {metrics['weighted_ce']:.4f}  "
              f"(prior baseline: {metrics['prior_ce']:.4f})")
        print(f"  Pred entropy:      {metrics['pred_entropy']:.4f}  "
              f"(classes used: {metrics['classes_predicted']}/3)")
        print()
        print(f"  Accuracy:          {metrics['accuracy']:.3f}")
        print(f"  Weighted accuracy: {metrics['weighted_accuracy']:.3f}")
        print(f"  Macro F1:          {metrics['macro_f1']:.3f}")
        print()
        print("  Per-class:")
        for c in SMILE_CLASSES:
            m = metrics["per_class"][c]
            print(f"    {c:>10s}: P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} (n={m['support']})")

    out_path = Path(args.out)
    output = {
        "args": vars(args),
        "metrics": metrics,
        "predictions": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating)
                  else int(x) if isinstance(x, np.integer) else x)
    if not args.quiet:
        print(f"\nResults saved to {out_path}")

    return metrics


if __name__ == "__main__":
    main()
