from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import NamedTuple


def sinkhorn_log_space(scores, num_iters=1, temperature=1.0):
    """Log-space Sinkhorn for maximum numerical stability."""
    log_matrix = scores / temperature

    for _ in range(num_iters):
        log_matrix = log_matrix - log_matrix.logsumexp(dim=-1, keepdim=True)
        log_matrix = log_matrix - log_matrix.logsumexp(dim=-2, keepdim=True)

    return torch.exp(log_matrix)


def make_bistochastic_log(*args, **kwargs):
    """Deprecated alias for sinkhorn_log_space."""
    import warnings
    warnings.warn(
        "make_bistochastic_log is deprecated, use sinkhorn_log_space",
        DeprecationWarning,
        stacklevel=2,
    )
    return sinkhorn_log_space(*args, **kwargs)


def bistochastic_loss(matrix, row_weight=1.0, col_weight=1.0):
    """Version with separate weights for row and column constraints."""
    row_sums = matrix.sum(dim=-1)
    col_sums = matrix.sum(dim=-2)

    row_loss = torch.mean((row_sums - 1.0) ** 2)
    col_loss = torch.mean((col_sums - 1.0) ** 2)

    return row_weight * row_loss + col_weight * col_loss


def graph_matching_loss_batched(M, A, B):
    """
    Batched graph matching loss.

    Args:
        M: Bistochastic matrices, shape (batch, n, n)
        A: Adjacency matrices of graph 1, shape (batch, n, n)
        B: Adjacency matrices of graph 2, shape (batch, n, n)

    Returns:
        Scalar loss (mean over batch)
    """
    # M @ B @ M^T
    MBMT = torch.matmul(torch.matmul(M, B), M.transpose(-2, -1))
    nA = torch.sum(A**2, dim=(-2, -1), keepdim=True)
    nB = torch.sum(B**2, dim=(-2, -1), keepdim=True)
    # trace(A^T @ M @ B @ M^T) per batch
    loss = torch.sum(A * MBMT, dim=(-2, -1)) / torch.sqrt(nA * nB)

    # Negate and average
    return -loss.mean()


def combined_loss_with_sinkhorn(
    scores,
    A,
    B,
    lambda_bisto=1.0,
    sinkhorn_iters=1,
    row_weight=1.0,
    col_weight=1.0,
    return_M=False,
):
    """
    Combined graph matching + bistochastic loss with Sinkhorn normalization.

    The Sinkhorn algorithm is applied INSIDE the loss function to convert
    raw scores into bistochastic matrices.

    Args:
        scores: Raw matching scores, shape (batch, n, n)
        A: Adjacency matrices of graph 1, shape (batch, n, n)
        B: Adjacency matrices of graph 2, shape (batch, n, n)
        lambda_bisto: Weight for bistochastic constraint
        sinkhorn_iters: Number of Sinkhorn iterations
        row_weight: Weight for row sum constraint
        col_weight: Weight for column sum constraint
        return_M: If True, also return the bistochastic matrix M

    Returns:
        total_loss, matching_loss, bisto_loss (and optionally M)
    """
    # Apply Sinkhorn to make bistochastic
    M = sinkhorn_log_space(scores, num_iters=sinkhorn_iters)

    # Graph matching loss
    matching_loss = graph_matching_loss_batched(M, A, B)

    # Bistochastic constraint loss
    bisto_loss = bistochastic_loss(M, row_weight, col_weight)

    # Combined loss
    total_loss = matching_loss + lambda_bisto * bisto_loss

    if return_M:
        return total_loss, matching_loss, bisto_loss, M
    return total_loss, matching_loss, bisto_loss


@dataclass(frozen=True)
class MatchingLossConfig:
    lambda_bisto: float = 1.0
    sinkhorn_iters: int = 1
    row_weight: float = 1.0
    col_weight: float = 1.0
    temperature: float = 1.0
    epsilon: float = 1e-8


class MatchingLossOutput(NamedTuple):
    total: torch.Tensor
    matching: torch.Tensor
    bistochastic: torch.Tensor
    matrix: torch.Tensor


def compute_matching_loss(
    scores: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: MatchingLossConfig = MatchingLossConfig(),
) -> MatchingLossOutput:
    """
    Compute the combined graph matching + bistochastic loss.

    Applies Sinkhorn normalization (in log-space) to convert raw scores into a
    bistochastic matrix M, then evaluates:
        matching  = -trace(A^T M B M^T) / sqrt(||A||_F * ||B||_F)
        bisto     = row_weight * MSE(M.sum(-1), 1) + col_weight * MSE(M.sum(-2), 1)
        total     = matching + lambda_bisto * bisto

    Args:
        scores: Raw matching scores, shape (batch, n, n).
        A: Adjacency matrices of graph 1, shape (batch, n, n).
        B: Adjacency matrices of graph 2, shape (batch, n, n).
        config: Hyperparameter bundle (MatchingLossConfig).

    Returns:
        MatchingLossOutput with fields total, matching, bistochastic, matrix.
    """
    M = sinkhorn_log_space(scores / config.temperature, config.sinkhorn_iters)
    nA = A.sum(dim=(-2, -1), keepdim=True)
    nB = B.sum(dim=(-2, -1), keepdim=True)
    matching = -(A * (M @ B @ M.transpose(-2, -1))).sum() / (
        torch.sqrt(nA * nB + config.epsilon)
    )
    bistochastic = (
        config.row_weight * F.mse_loss(M.sum(-1), torch.ones_like(M.sum(-1))) +
        config.col_weight * F.mse_loss(M.sum(-2), torch.ones_like(M.sum(-2)))
    )
    total = matching + config.lambda_bisto * bistochastic
    return MatchingLossOutput(
        total=total,
        matching=matching,
        bistochastic=bistochastic,
        matrix=M,
    )
