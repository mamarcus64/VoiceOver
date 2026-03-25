"""
Model definitions for smile prediction.

Tier 1: Aggregated feature baseline (logistic regression / small MLP)
Tier 2: Sequence models (GRU, 1D-CNN)

Loss: class-balanced soft CE with entropy regularization to prevent collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AggregatedMLP(nn.Module):
    """
    Tier 1 baseline: small MLP over hand-crafted summary features.
    Input: (batch, D) aggregated stats.
    Output: (batch, 3) logits over [genuine, polite, masking].
    """

    def __init__(self, input_dim: int = 36, hidden_dim: int = 24, num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUClassifier(nn.Module):
    """
    Tier 2: small GRU over the time-aligned feature sequence.
    Input: (batch, T, input_dim).
    Output: (batch, 3) logits over [genuine, polite, masking].
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 24, num_layers: int = 1,
                 num_classes: int = 3, dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        dir_mult = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * dir_mult, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h = self.gru(packed)
        else:
            _, h = self.gru(x)

        if self.gru.bidirectional:
            h = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h = h[-1]

        h = self.dropout(h)
        return self.head(h)


class CNNClassifier(nn.Module):
    """
    Tier 2 alternative: small 1D-CNN over the time-aligned feature sequence.
    Input: (batch, T, input_dim).
    Output: (batch, 3) logits over [genuine, polite, masking].
    """

    def __init__(self, input_dim: int = 11, channels: int = 16, num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if lengths is not None:
            mask = torch.arange(x.size(2), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * mask.unsqueeze(1).float()
            x = x.sum(dim=2) / lengths.unsqueeze(1).float()
        else:
            x = x.mean(dim=2)
        x = self.dropout(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_class_weights(soft_labels: torch.Tensor) -> torch.Tensor:
    """
    Inverse-frequency class weights from the training soft labels.
    Upweights rare classes (masking) so the loss doesn't ignore them.
    """
    # Effective count per class = sum of soft label mass
    class_mass = soft_labels.sum(dim=0).clamp(min=1e-6)
    # Inverse frequency, normalized to mean=1
    inv_freq = 1.0 / class_mass
    weights = inv_freq / inv_freq.mean()
    return weights


def soft_label_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    weights: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    entropy_reg: float = 0.1,
) -> torch.Tensor:
    """
    Weighted soft cross-entropy with class balancing and entropy regularization.

    Args:
        logits: (B, C) raw model output
        soft_labels: (B, C) target distribution, sums to 1
        weights: (B,) per-sample weight (not-a-smile discount)
        class_weights: (C,) inverse-frequency weights per class. If None, uniform.
        entropy_reg: coefficient for output entropy bonus. Higher values push
            the model away from collapsed (single-class) predictions.

    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(logits, dim=-1)

    # Class-weighted soft CE: sum_c class_weight_c * target_c * (-log pred_c)
    if class_weights is not None:
        per_sample = -(soft_labels * log_probs * class_weights.unsqueeze(0)).sum(dim=-1)
    else:
        per_sample = -(soft_labels * log_probs).sum(dim=-1)

    # Entropy bonus: encourage the model to spread probability mass
    # H(pred) = -sum_c pred_c * log pred_c — higher is more spread out
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    per_sample = per_sample - entropy_reg * entropy

    weighted = per_sample * weights
    if weights.sum() > 0:
        return weighted.sum() / weights.sum()
    return weighted.mean()
