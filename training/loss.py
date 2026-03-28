#!/usr/bin/env python3
"""Loss placeholders for speech enhancement training."""

from __future__ import annotations

import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple baseline loss for initial model bring-up."""
    return torch.mean((pred - target) ** 2)
