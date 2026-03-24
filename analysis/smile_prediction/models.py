"""
Model definitions for smile prediction.

Tier 1: Aggregated feature baseline (logistic regression / small MLP)
Tier 2: Sequence models (GRU, 1D-CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregatedMLP(nn.Module):
    """
    Tier 1 baseline: small MLP over hand-crafted summary features.
    Input: (batch, 36) aggregated stats.
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
    Input: (batch, T, input_dim) where input_dim=11.
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
    Input: (batch, T, input_dim) where input_dim=11.
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
        # x: (B, T, C) -> (B, C, T) for conv1d
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Global average pool over time
        if lengths is not None:
            mask = torch.arange(x.size(2), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * mask.unsqueeze(1).float()
            x = x.sum(dim=2) / lengths.unsqueeze(1).float()
        else:
            x = x.mean(dim=2)
        x = self.dropout(x)
        return self.head(x)


def soft_label_loss(logits: torch.Tensor, soft_labels: torch.Tensor,
                    weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted KL-divergence loss against soft label targets.

    Args:
        logits: (B, C) raw model output
        soft_labels: (B, C) target distribution, sums to 1
        weights: (B,) per-sample weight (not-a-smile discount)

    Returns:
        Scalar loss (mean over batch, weighted).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # KL(target || pred) = sum_c target_c * (log target_c - log pred_c)
    # We skip the target entropy term (constant wrt model params)
    per_sample = -(soft_labels * log_probs).sum(dim=-1)
    weighted = per_sample * weights
    if weights.sum() > 0:
        return weighted.sum() / weights.sum()
    return weighted.mean()
