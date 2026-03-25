"""
Smoke test for the smile prediction pipeline.

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
    attach_llm_features, N_LLM_FEATURES,
)
from .models import AggregatedMLP, GRUClassifier, CNNClassifier, soft_label_loss, compute_class_weights
from .train import main as train_main


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_data_loading():
    section("1. Data Loading & Soft Labels")

    manifest = load_manifest()
    print(f"  Manifest: {manifest['total_tasks']} total tasks")

    annotations = load_annotations()
    for name, anns in annotations.items():
        print(f"    {name}: {len(anns)} annotations")

    tasks_raw = build_tasks(min_annotators=1, label_smoothing=0.0)
    tasks_smooth = build_tasks(min_annotators=1, label_smoothing=0.1)
    print(f"\n  Tasks (no smoothing): {len(tasks_raw)}")
    print(f"  Tasks (smoothing=0.1): {len(tasks_smooth)}")

    # Verify smoothing works: single-annotator hard labels should differ
    for raw, smooth in zip(tasks_raw[:5], tasks_smooth[:5]):
        if raw.annotator_count == 1:
            assert not np.allclose(raw.soft_label, smooth.soft_label), \
                "Smoothing should change single-annotator labels"
            assert smooth.soft_label.min() > 0, "Smoothed label should have no zeros"

    for t in tasks_smooth[:10]:
        assert abs(t.soft_label.sum() - 1.0) < 1e-5

    print("  PASS")
    return tasks_smooth


def test_feature_extraction(tasks):
    section("2. Feature Extraction")

    for modality in ["both", "audio", "eyegaze"]:
        feats = extract_features(tasks[0], modality=modality)
        agg = extract_aggregated_features(tasks[0], modality=modality)
        assert feats is not None and agg is not None
        assert feats["sequence"].shape[1] == seq_feature_dim(modality)
        assert agg["features"].shape[0] == agg_feature_dim(modality)
        print(f"  [{modality:>7s}] seq={feats['sequence'].shape}, agg={agg['features'].shape}")

    print("  PASS")


def test_models_and_loss():
    section("3. Models + Class-Balanced Loss")

    B, T = 4, 20
    soft_labels = torch.tensor([
        [0.9, 0.067, 0.033],
        [0.033, 0.9, 0.067],
        [0.067, 0.033, 0.9],
        [0.5, 0.3, 0.2],
    ])
    weights = torch.tensor([1.0, 1.0, 1.0, 0.75])
    class_weights = compute_class_weights(soft_labels)
    print(f"  Class weights: {class_weights.tolist()}")

    x_agg = torch.randn(B, 36)
    mlp = AggregatedMLP(input_dim=36)
    logits = mlp(x_agg)
    loss = soft_label_loss(logits, soft_labels, weights,
                           class_weights=class_weights, entropy_reg=0.1)
    print(f"  MLP loss (with class balance + entropy reg): {loss.item():.4f}")
    assert not torch.isnan(loss)

    x_seq = torch.randn(B, T, 11)
    lengths = torch.tensor([20, 15, 10, 5])
    gru = GRUClassifier(input_dim=11)
    logits = gru(x_seq, lengths)
    probs = torch.softmax(logits, dim=-1)
    print(f"  GRU output probs: {probs.detach().numpy().round(3)}")
    # With entropy reg, no class should be exactly 0
    loss = soft_label_loss(logits, soft_labels, weights,
                           class_weights=class_weights, entropy_reg=0.1)
    print(f"  GRU loss: {loss.item():.4f}")

    print("  PASS")


def test_training_with_metrics():
    section("4. Training with New Metrics (3-fold, 1 epoch)")

    print("  Running mlp/both ...", end=" ", flush=True)
    metrics = train_main([
        "--model", "mlp", "--modality", "both",
        "--epochs", "1", "--folds", "3",
        "--out", "/tmp/smoke_mlp.json", "--quiet",
    ])
    print(f"brier={metrics['brier_score']:.4f} skill={metrics['brier_skill']:+.3f} "
          f"cls={metrics['classes_predicted']}/3")

    print("  Running gru/both ...", end=" ", flush=True)
    metrics = train_main([
        "--model", "gru", "--modality", "both",
        "--epochs", "1", "--folds", "3",
        "--out", "/tmp/smoke_gru.json", "--quiet",
    ])
    print(f"brier={metrics['brier_score']:.4f} skill={metrics['brier_skill']:+.3f} "
          f"cls={metrics['classes_predicted']}/3")

    assert "brier_score" in metrics
    assert "prior_brier" in metrics
    assert "brier_skill" in metrics
    assert "pred_entropy" in metrics
    assert "classes_predicted" in metrics

    print("  PASS")


def test_text_feature_attachment():
    section("5. Text Feature Attachment (synthetic)")

    tasks = build_tasks(min_annotators=1, label_smoothing=0.1)

    # Get a few aggregated and sequence items
    agg_items = [extract_aggregated_features(t) for t in tasks[:5]]
    agg_items = [i for i in agg_items if i is not None]
    seq_items = [extract_features(t) for t in tasks[:5]]
    seq_items = [i for i in seq_items if i is not None]

    # Synthetic LLM features
    fake_llm_feats = {item["task_number"]: np.random.rand(N_LLM_FEATURES).astype(np.float32)
                      for item in agg_items}

    # Test aggregated
    agg_with_text = attach_llm_features(agg_items, fake_llm_feats)
    assert len(agg_with_text) == len(agg_items)
    orig_dim = agg_items[0]["features"].shape[0]
    new_dim = agg_with_text[0]["features"].shape[0]
    assert new_dim == orig_dim + N_LLM_FEATURES, f"{new_dim} != {orig_dim}+{N_LLM_FEATURES}"
    print(f"  Aggregated: {orig_dim} -> {new_dim} (+{N_LLM_FEATURES} text)")

    # Test sequence
    fake_llm_feats_seq = {item["task_number"]: np.random.rand(N_LLM_FEATURES).astype(np.float32)
                          for item in seq_items}
    seq_with_text = attach_llm_features(seq_items, fake_llm_feats_seq)
    assert len(seq_with_text) == len(seq_items)
    orig_d = seq_items[0]["sequence"].shape[1]
    new_d = seq_with_text[0]["sequence"].shape[1]
    assert new_d == orig_d + N_LLM_FEATURES
    print(f"  Sequence: (T, {orig_d}) -> (T, {new_d}) (+{N_LLM_FEATURES} text)")

    # Missing LLM features should drop items
    partial = {agg_items[0]["task_number"]: np.zeros(N_LLM_FEATURES, dtype=np.float32)}
    partial_result = attach_llm_features(agg_items, partial)
    assert len(partial_result) == 1
    print(f"  Partial match: {len(agg_items)} items -> {len(partial_result)} matched")

    print("  PASS")


def test_sweep_quick():
    section("6. Quick Sweep")

    from .sweep import main as sweep_main
    t0 = time.time()
    sys.argv = ["sweep", "--quick", "--out-dir", "/tmp/smile_sweep_test"]
    sweep_main()
    print(f"\n  Sweep done ({time.time() - t0:.1f}s)")
    print("  PASS")


def main():
    print("Smile Prediction Pipeline — Smoke Tests")
    print(f"PyTorch: {torch.__version__}, NumPy: {np.__version__}")

    tasks = test_data_loading()
    test_feature_extraction(tasks)
    test_models_and_loss()
    test_training_with_metrics()
    test_text_feature_attachment()
    test_sweep_quick()

    section("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
