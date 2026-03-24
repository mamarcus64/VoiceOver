"""
LOO cross-validation training and evaluation for smile prediction.

Usage:
    python -m analysis.smile_prediction.train [OPTIONS]

Options:
    --model         mlp | gru | cnn          (default: mlp)
    --epochs        int                       (default: 100)
    --lr            float                     (default: 1e-3)
    --weight-decay  float                     (default: 1e-2)
    --window-before float  seconds            (default: 30)
    --window-after  float  seconds            (default: 30)
    --grid-step     float  seconds            (default: 3)
    --min-annotators int                      (default: 1)
    --hidden-dim    int                       (default: 24)
    --dropout       float                     (default: 0.3)
    --max-folds     int    for quick testing  (default: 0 = all)
    --seed          int                       (default: 42)
    --out           path   results JSON       (default: results.json)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dataset import (
    SmileTask, build_tasks, extract_features, extract_aggregated_features,
    SMILE_CLASSES,
)
from .models import AggregatedMLP, GRUClassifier, CNNClassifier, soft_label_loss


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Smile prediction LOO cross-validation")
    p.add_argument("--model", choices=["mlp", "gru", "cnn"], default="mlp")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--window-before", type=float, default=30.0)
    p.add_argument("--window-after", type=float, default=30.0)
    p.add_argument("--grid-step", type=float, default=3.0)
    p.add_argument("--min-annotators", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-folds", type=int, default=0, help="0 = all folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results.json")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def prepare_all_data(tasks: list[SmileTask], args) -> list[dict]:
    """Extract features for all tasks, returning list of dicts."""
    use_agg = args.model == "mlp"
    data = []
    skipped = 0
    for task in tasks:
        if use_agg:
            item = extract_aggregated_features(
                task, args.window_before, args.window_after, args.grid_step
            )
        else:
            item = extract_features(
                task, args.window_before, args.window_after, args.grid_step
            )
        if item is None:
            skipped += 1
            continue
        data.append(item)
    if skipped:
        print(f"  Skipped {skipped} tasks (no features)")
    return data


def collate_sequences(items: list[dict]) -> tuple:
    """Pad variable-length sequences for batch processing."""
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
    """Stack aggregated feature vectors."""
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
# Training loop for a single LOO fold
# ---------------------------------------------------------------------------

def train_one_fold(
    train_data: list[dict],
    test_item: dict,
    args,
) -> dict:
    """Train model on train_data, predict on test_item. Return prediction dict."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_seq = args.model in ("gru", "cnn")

    # Collate training data
    if is_seq:
        X_train, L_train, Y_train, W_train = collate_sequences(train_data)
        X_test, L_test, Y_test, W_test = collate_sequences([test_item])
    else:
        X_train, _, Y_train, W_train = collate_aggregated(train_data)
        X_test, _, Y_test, W_test = collate_aggregated([test_item])

    # Build model
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

    # Training
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        if is_seq:
            logits = model(X_train, L_train)
        else:
            logits = model(X_train)
        loss = soft_label_loss(logits, Y_train, W_train)
        loss.backward()
        optimizer.step()

    # Prediction
    model.eval()
    with torch.no_grad():
        if is_seq:
            logits = model(X_test, L_test)
        else:
            logits = model(X_test)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    return {
        "task_number": test_item["task_number"],
        "predicted_probs": probs.tolist(),
        "predicted_class": SMILE_CLASSES[int(probs.argmax())],
        "true_soft_label": test_item["soft_label"].tolist(),
        "true_top_class": SMILE_CLASSES[int(test_item["soft_label"].argmax())],
        "weight": float(test_item["weight"]),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from LOO fold results."""
    n = len(results)
    if n == 0:
        return {}

    correct = 0
    weighted_correct = 0.0
    total_weight = 0.0
    weighted_kl = 0.0

    per_class_tp = {c: 0 for c in SMILE_CLASSES}
    per_class_fp = {c: 0 for c in SMILE_CLASSES}
    per_class_fn = {c: 0 for c in SMILE_CLASSES}

    for r in results:
        pred_cls = r["predicted_class"]
        true_cls = r["true_top_class"]
        w = r["weight"]

        if pred_cls == true_cls:
            correct += 1
            weighted_correct += w
        total_weight += w

        per_class_tp[pred_cls] += int(pred_cls == true_cls)
        if pred_cls != true_cls:
            per_class_fp[pred_cls] += 1
            per_class_fn[true_cls] += 1

        # Weighted cross-entropy (approximation of KL without constant)
        pred_probs = np.array(r["predicted_probs"])
        true_dist = np.array(r["true_soft_label"])
        pred_probs = np.clip(pred_probs, 1e-8, 1.0)
        kl = -(true_dist * np.log(pred_probs)).sum()
        weighted_kl += w * kl

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

    macro_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])

    return {
        "n_samples": n,
        "accuracy": correct / n,
        "weighted_accuracy": weighted_correct / total_weight if total_weight > 0 else 0,
        "weighted_cross_entropy": weighted_kl / total_weight if total_weight > 0 else 0,
        "macro_f1": float(macro_f1),
        "per_class": per_class_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    print(f"=== Smile Prediction LOO CV ===")
    print(f"Model: {args.model}, Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Window: [{args.window_before}s before, {args.window_after}s after], "
          f"Grid: {args.grid_step}s")
    print(f"Min annotators: {args.min_annotators}")
    print()

    # Load tasks
    tasks = build_tasks(min_annotators=args.min_annotators)
    print(f"Loaded {len(tasks)} tasks with annotations")

    # Extract features
    print("Extracting features...")
    t0 = time.time()
    all_data = prepare_all_data(tasks, args)
    print(f"  {len(all_data)} samples ready ({time.time() - t0:.1f}s)")
    print()

    if len(all_data) < 2:
        print("ERROR: Need at least 2 samples for LOO CV")
        sys.exit(1)

    # LOO cross-validation
    n_folds = len(all_data) if args.max_folds == 0 else min(args.max_folds, len(all_data))
    print(f"Running {n_folds} LOO folds...")

    results = []
    t0 = time.time()
    for i in range(n_folds):
        test_item = all_data[i]
        train_data = all_data[:i] + all_data[i + 1:]

        result = train_one_fold(train_data, test_item, args)
        results.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == n_folds:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_folds - i - 1) / rate if rate > 0 else 0
            print(f"  Fold {i + 1}/{n_folds}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Compute metrics
    metrics = compute_metrics(results)
    print()
    print("=== Results ===")
    print(f"Accuracy:          {metrics['accuracy']:.3f}")
    print(f"Weighted accuracy: {metrics['weighted_accuracy']:.3f}")
    print(f"Weighted CE loss:  {metrics['weighted_cross_entropy']:.3f}")
    print(f"Macro F1:          {metrics['macro_f1']:.3f}")
    print()
    print("Per-class:")
    for c in SMILE_CLASSES:
        m = metrics["per_class"][c]
        print(f"  {c:>10s}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (n={m['support']})")

    # Save results
    out_path = Path(args.out)
    output = {
        "args": vars(args),
        "metrics": metrics,
        "predictions": results,
    }
    # Convert numpy types for JSON serialization
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
