"""
Data Simulation for Multichoice Logit Models

Generates synthetic datasets following the Random Utility Maximization (RUM) principle
for multichoice logit models where agents can select single alternatives or pairs.
Fully vectorized implementation for fast simulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def simulate_data(
    N: int,
    J: int,
    K: int,
    true_beta: npt.NDArray[np.float64] | None = None,
    mix_ratio: float = 0.5,
    seed: int | None = 42,
    rng: np.random.Generator | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int8],
    npt.NDArray[np.int8],
    npt.NDArray[np.float64],
]:
    """
    Generates synthetic dataset following the Random Utility Maximization (RUM) principle.

    The simulation process:
    1. Generates random covariates X ~ N(0, 1)
    2. Computes deterministic utilities V = X @ beta.T
    3. Adds Gumbel noise to create realized utilities U = V + epsilon
    4. Agents choose either the single best alternative (with probability mix_ratio)
       or the top two alternatives as an unordered pair (with probability 1-mix_ratio)

    Args:
        N (int): Number of observations.
        J (int): Number of alternatives.
        K (int): Number of covariates.
        true_beta (np.ndarray, optional): True parameters (J-1, K).
                                         If None, generated uniformly in [-1, 1].
        mix_ratio (float): Fraction of population making single choices (0.0 to 1.0).
                          Default is 0.5 (equal mix of single and dual choices).
        seed (int | None): Random seed for reproducibility. Ignored if rng is provided.
        rng (np.random.Generator, optional): Use an existing RNG instead of seed.
        dtype: dtype for generated covariates and parameters (default float64).

    Returns:
        tuple: (X, y_single, y_dual, true_beta_free)
            - X (np.ndarray): Covariate matrix of shape (N, K)
            - y_single (np.ndarray): Binary matrix (N, J) indicating single choices
            - y_dual (np.ndarray): Binary tensor (N, J, J) indicating dual choices
            - true_beta_free (np.ndarray): True parameter matrix (J-1, K) used in simulation

    Raises:
        ValueError: If N, J, or K are invalid, or if mix_ratio is not in [0, 1].
        ValueError: If true_beta has an incorrect shape.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if J < 2:
        raise ValueError(f"J must be >= 2, got {J}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if not 0 <= mix_ratio <= 1:
        raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")
    if true_beta is not None and true_beta.shape != (J - 1, K):
        raise ValueError(
            f"true_beta must have shape ({J - 1}, {K}), got {true_beta.shape}"
        )

    rng = rng or np.random.default_rng(seed)

    # Generate covariates
    X = rng.normal(size=(N, K)).astype(dtype, copy=False)

    # Generate or Use Parameters
    if true_beta is None:
        true_beta_free = rng.uniform(-1, 1, (J - 1, K)).astype(dtype, copy=False)
    else:
        true_beta_free = true_beta.astype(dtype, copy=False)

    # Add fixed class 0 (identification constraint)
    beta_full = np.vstack([np.zeros((1, K), dtype=dtype), true_beta_free])

    # Calculate Deterministic Utility
    V = X @ beta_full.T

    # Add Gumbel Noise
    epsilon = rng.gumbel(loc=0, scale=1, size=(N, J))
    U = V + epsilon

    # Selection Logic
    y_single = np.zeros((N, J), dtype=np.int8)
    y_dual = np.zeros((N, J, J), dtype=np.int8)

    # Efficiently find Top 2 using argpartition
    top_k_indices = np.argpartition(U, -2, axis=1)[:, -2:]
    top_k_values = np.take_along_axis(U, top_k_indices, axis=1)
    sorted_within_top = np.argsort(top_k_values, axis=1)

    best_idx = np.take_along_axis(
        top_k_indices, sorted_within_top[:, -1:], axis=1
    ).flatten()
    second_best_idx = np.take_along_axis(
        top_k_indices, sorted_within_top[:, -2:-1], axis=1
    ).flatten()

    # Randomly assign choice mode (single vs dual) based on mix_ratio
    mode_choice = rng.binomial(1, mix_ratio, N).astype(bool)

    # Single choice assignment
    # For all agents with mode_choice == 1, set y_single[i, best_idx[i]] = 1
    single_mask = mode_choice
    if np.any(single_mask):
        row_indices_single = np.arange(N)[single_mask]
        col_indices_single = best_idx[single_mask]
        y_single[row_indices_single, col_indices_single] = 1

    # Dual choice assignment
    # For all agents with mode_choice == 0, set y_dual[i, s, t] = 1 where s < t
    dual_mask = ~mode_choice
    if np.any(dual_mask):
        row_indices_dual = np.arange(N)[dual_mask]
        s_indices = best_idx[dual_mask]
        t_indices = second_best_idx[dual_mask]

        # Ensure s < t (swap if needed)
        swap_mask = s_indices > t_indices
        s_indices[swap_mask], t_indices[swap_mask] = (
            t_indices[swap_mask],
            s_indices[swap_mask],
        )

        y_dual[row_indices_dual, s_indices, t_indices] = 1

    return X, y_single, y_dual, true_beta_free
