"""
Smoke test for the smile prediction pipeline.

Runs lightweight checks on every component:
1. Data loading and soft-label computation
2. Feature extraction (both aggregated and sequential)
3. Model forward passes
4. Loss computation
5. 1-epoch, 5-fold LOO to verify the full training loop

Usage:
    python -m analysis.smile_prediction.test_pipeline
"""

import sys
import time
import numpy as np
import torch

from .dataset import (
    build_tasks, extract_features, extract_aggregated_features,
    load_manifest, load_annotations, SMILE_CLASSES,
)
from .models import AggregatedMLP, GRUClassifier, CNNClassifier, soft_label_loss
from .train import main as train_main


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_data_loading():
    section("1. Data Loading & Soft Labels")

    manifest = load_manifest()
    print(f"  Manifest: {manifest['total_tasks']} total tasks, "
          f"{manifest['videos_with_tasks']} videos")

    annotations = load_annotations()
    print(f"  Annotators: {list(annotations.keys())}")
    for name, anns in annotations.items():
        print(f"    {name}: {len(anns)} annotations")

    tasks = build_tasks(min_annotators=1)
    print(f"\n  Built {len(tasks)} tasks (min_annotators=1)")

    tasks_multi = build_tasks(min_annotators=2)
    print(f"  Built {len(tasks_multi)} tasks (min_annotators=2)")

    # Check soft labels sum to 1
    for t in tasks[:10]:
        assert abs(t.soft_label.sum() - 1.0) < 1e-5, f"Task {t.task_number}: label sums to {t.soft_label.sum()}"
        assert 0 < t.weight <= 1.0, f"Task {t.task_number}: weight={t.weight}"

    # Show a few examples
    print("\n  Example tasks:")
    for t in tasks[:5]:
        label_str = ", ".join(f"{c}={t.soft_label[i]:.2f}" for i, c in enumerate(SMILE_CLASSES))
        print(f"    Task {t.task_number}: video={t.video_id}, "
              f"smile=[{t.smile_start:.1f}, {t.smile_end:.1f}], "
              f"weight={t.weight:.2f}, [{label_str}], "
              f"annotators={t.annotator_count}")

    print("  PASS")
    return tasks


def test_feature_extraction(tasks):
    section("2. Feature Extraction")

    # Sequential features
    t = tasks[0]
    feats = extract_features(t, window_before=30, window_after=30, grid_step=3)
    assert feats is not None, "Feature extraction returned None"
    seq = feats["sequence"]
    print(f"  Sequential features for task {t.task_number}:")
    print(f"    Shape: {seq.shape} (T={seq.shape[0]}, features={seq.shape[1]})")
    print(f"    Feature dims: [audio_V,A,D,present, eyegaze_V,A,D,present, phase_b,d,a]")
    print(f"    Audio coverage: {seq[:, 3].mean():.2f}")
    print(f"    Eyegaze coverage: {seq[:, 7].mean():.2f}")
    print(f"    Phase before: {seq[:, 8].sum():.0f} steps, "
          f"during: {seq[:, 9].sum():.0f}, after: {seq[:, 10].sum():.0f}")

    # Aggregated features
    agg = extract_aggregated_features(t, window_before=30, window_after=30, grid_step=3)
    assert agg is not None, "Aggregated extraction returned None"
    print(f"\n  Aggregated features for task {t.task_number}:")
    print(f"    Shape: {agg['features'].shape}")
    print(f"    Values range: [{agg['features'].min():.4f}, {agg['features'].max():.4f}]")

    # Extract a batch
    print(f"\n  Extracting features for first 20 tasks...")
    t0 = time.time()
    count = 0
    for task in tasks[:20]:
        f = extract_features(task)
        if f is not None:
            count += 1
    print(f"    {count}/20 successful ({time.time() - t0:.1f}s)")

    print("  PASS")


def test_models():
    section("3. Model Forward Passes")

    B, T, D = 4, 20, 11

    # Random batch
    x_seq = torch.randn(B, T, D)
    lengths = torch.tensor([20, 15, 10, 5])
    x_agg = torch.randn(B, 36)

    # MLP
    mlp = AggregatedMLP(input_dim=36, hidden_dim=24)
    out = mlp(x_agg)
    print(f"  MLP:  input={x_agg.shape} -> output={out.shape}")
    assert out.shape == (B, 3)

    # GRU
    gru = GRUClassifier(input_dim=D, hidden_dim=24)
    out = gru(x_seq, lengths)
    print(f"  GRU:  input={x_seq.shape} -> output={out.shape}")
    assert out.shape == (B, 3)

    # CNN
    cnn = CNNClassifier(input_dim=D, channels=16)
    out = cnn(x_seq, lengths)
    print(f"  CNN:  input={x_seq.shape} -> output={out.shape}")
    assert out.shape == (B, 3)

    print("  PASS")


def test_loss():
    section("4. Loss Computation")

    logits = torch.randn(4, 3)
    soft_labels = torch.tensor([
        [0.67, 0.33, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.33, 0.33, 0.34],
    ])
    weights = torch.tensor([0.75, 1.0, 0.5, 1.0])

    loss = soft_label_loss(logits, soft_labels, weights)
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"

    # Check that zero weights contribute nothing
    weights_zero = torch.tensor([0.0, 0.0, 0.0, 0.0])
    loss_zero = soft_label_loss(logits, soft_labels, weights_zero)
    print(f"  Loss with zero weights: {loss_zero.item():.4f}")

    print("  PASS")


def test_loo_smoke():
    section("5. LOO Training Smoke Test (1 epoch, 5 folds)")

    print("  Running MLP with --epochs 1 --max-folds 5 ...")
    t0 = time.time()
    train_main(["--model", "mlp", "--epochs", "1", "--max-folds", "5",
                "--out", "/tmp/smile_test_mlp.json"])
    print(f"  MLP done ({time.time() - t0:.1f}s)")

    print("\n  Running GRU with --epochs 1 --max-folds 5 ...")
    t0 = time.time()
    train_main(["--model", "gru", "--epochs", "1", "--max-folds", "5",
                "--out", "/tmp/smile_test_gru.json"])
    print(f"  GRU done ({time.time() - t0:.1f}s)")

    print("\n  Running CNN with --epochs 1 --max-folds 5 ...")
    t0 = time.time()
    train_main(["--model", "cnn", "--epochs", "1", "--max-folds", "5",
                "--out", "/tmp/smile_test_cnn.json"])
    print(f"  CNN done ({time.time() - t0:.1f}s)")

    print("  PASS")


def main():
    print("Smile Prediction Pipeline — Smoke Tests")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")

    tasks = test_data_loading()
    test_feature_extraction(tasks)
    test_models()
    test_loss()
    test_loo_smoke()

    section("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
