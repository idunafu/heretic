# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Philipp Emanuel Weidmann + contributors

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import RowNormalization


def additive_factors(
    weight: Tensor,
    directions: Tensor,
    strengths: Tensor,
    normalization: RowNormalization,
    rank: int,
    seed: int | None,
) -> tuple[Tensor, Tensor]:
    """Return B, A for a weighted sum of directional updates.

    NONE/PRE are exact (up to rounding), even for nonorthogonal directions.
    FULL approximates the renormalized difference using a truncated SVD.
    Zero padding keeps adapter shapes fixed across trials and source layers.
    """
    if directions.ndim != 2 or directions.shape[1] != weight.shape[0]:
        raise ValueError("Directions must have shape (count, weight output dimension)")
    if strengths.shape != (directions.shape[0],):
        raise ValueError("One strength is required per direction")
    if rank < directions.shape[0]:
        raise ValueError("Adapter rank must be at least the direction count")
    W = weight.float()
    V = directions.to(device=W.device, dtype=torch.float32)
    strengths = strengths.to(device=W.device, dtype=torch.float32)
    if not torch.isfinite(V).all() or not torch.isfinite(strengths).all():
        raise ValueError("Directions and strengths must be finite")

    row_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)
    normalized = W if normalization == RowNormalization.NONE else F.normalize(W, dim=1)
    A = V @ normalized
    B = -V.T * strengths
    if normalization == RowNormalization.PRE:
        B = row_norms * B
    elif normalization == RowNormalization.FULL:
        delta = F.normalize(normalized + B @ A, dim=1) * row_norms - W
        # Bound q for small modules and avoid perturbing the caller's RNG state.
        devices = [W.device.index] if W.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
                if W.is_cuda:
                    with torch.cuda.device(W.device):
                        torch.cuda.manual_seed(seed)
            U, S, V = torch.svd_lowrank(
                delta, q=min(2 * rank + 4, *delta.shape), niter=6
            )
        used_rank = min(rank, S.numel())
        sqrt_s = S[:used_rank].sqrt()
        B = U[:, :used_rank] * sqrt_s
        A = sqrt_s[:, None] * V[:, :used_rank].T

    padding = rank - A.shape[0]
    return F.pad(B, (0, padding)), F.pad(A, (0, 0, 0, padding))
