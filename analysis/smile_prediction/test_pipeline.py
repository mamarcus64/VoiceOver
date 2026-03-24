"""
Smoke test for the smile prediction pipeline.

Runs lightweight checks on every component:
1. Data loading and soft-label computation
2. Feature extraction with modality selection
3. Model forward passes
4. Loss computation
5. k-fold CV training loop (1 epoch, 3 folds)
6. Quick sweep (1 epoch, 3 folds, minimal grid)

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
    seq_feature_dim, agg_feature_dim,
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

    for t in tasks[:10]:
        assert abs(t.soft_label.sum() - 1.0) < 1e-5
        assert 0 < t.weight <= 1.0

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
    section("2. Feature Extraction (modality variants)")

    t = tasks[0]
    for modality in ["both", "audio", "eyegaze"]:
        feats = extract_features(t, window_before=30, window_after=30, grid_step=3,
                                 modality=modality)
        assert feats is not None
        seq = feats["sequence"]
        expected_dim = seq_feature_dim(modality)
        assert seq.shape[1] == expected_dim, (
            f"{modality}: got {seq.shape[1]}, expected {expected_dim}")
        print(f"  Sequential [{modality:>7s}]: shape={seq.shape}")

        agg = extract_aggregated_features(t, window_before=30, window_after=30, grid_step=3,
                                          modality=modality)
        assert agg is not None
        expected_agg = agg_feature_dim(modality)
        assert agg["features"].shape[0] == expected_agg, (
            f"{modality}: agg got {agg['features'].shape[0]}, expected {expected_agg}")
        print(f"  Aggregated [{modality:>7s}]: shape={agg['features'].shape}")

    print("  PASS")


def test_models():
    section("3. Model Forward Passes")

    B, T = 4, 20

    for modality in ["both", "audio", "eyegaze"]:
        d_seq = seq_feature_dim(modality)
        d_agg = agg_feature_dim(modality)
        x_seq = torch.randn(B, T, d_seq)
        lengths = torch.tensor([20, 15, 10, 5])
        x_agg = torch.randn(B, d_agg)

        mlp = AggregatedMLP(input_dim=d_agg, hidden_dim=24)
        assert mlp(x_agg).shape == (B, 3)

        gru = GRUClassifier(input_dim=d_seq, hidden_dim=24)
        assert gru(x_seq, lengths).shape == (B, 3)

        cnn = CNNClassifier(input_dim=d_seq, channels=16)
        assert cnn(x_seq, lengths).shape == (B, 3)

        print(f"  [{modality:>7s}] MLP({d_agg}), GRU({d_seq}), CNN({d_seq}) — OK")

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
    assert loss.item() > 0
    assert not torch.isnan(loss)

    weights_zero = torch.tensor([0.0, 0.0, 0.0, 0.0])
    loss_zero = soft_label_loss(logits, soft_labels, weights_zero)
    print(f"  Loss with zero weights: {loss_zero.item():.4f}")

    print("  PASS")


def test_kfold_cv():
    section("5. k-fold CV Training Smoke Test")

    configs = [
        ("mlp", "both",    "10-fold"),
        ("gru", "audio",   "10-fold"),
        ("cnn", "eyegaze", "LOO"),
    ]

    for model, modality, fold_desc in configs:
        folds_arg = "-1" if fold_desc == "LOO" else "3"
        print(f"  {model}/{modality} ({fold_desc}) ...", end=" ", flush=True)
        t0 = time.time()
        # Use --folds 3 for quick testing (or -1 for LOO test with small data)
        actual_folds = folds_arg
        if fold_desc == "LOO":
            actual_folds = "-1"
            # LOO on full data is slow; use min-annotators=5 to shrink dataset
            extra = ["--min-annotators", "5"]
        else:
            extra = []
        train_main([
            "--model", model, "--modality", modality,
            "--epochs", "1", "--folds", actual_folds,
            "--out", f"/tmp/smile_test_{model}_{modality}.json",
            "--quiet",
        ] + extra)
        print(f"done ({time.time() - t0:.1f}s)")

    print("  PASS")


def test_sweep_quick():
    section("6. Quick Sweep (smoke test)")

    from .sweep import main as sweep_main
    t0 = time.time()
    # sweep --quick uses 1 epoch, 3 folds, minimal grid
    sys.argv = ["sweep", "--quick", "--out-dir", "/tmp/smile_sweep_test"]
    sweep_main()
    print(f"\n  Sweep done ({time.time() - t0:.1f}s)")
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
    test_kfold_cv()
    test_sweep_quick()

    section("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
