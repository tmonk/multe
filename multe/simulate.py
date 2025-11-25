"""
Data Simulation for Multichoice Logit Models

Generates synthetic datasets following the Random Utility Maximization (RUM) principle
for multichoice logit models where agents can select single alternatives or pairs.
Fully vectorized implementation for fast simulation.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def parse_choices(
    choices: Sequence[int | tuple[int, int]] | np.ndarray,
    J: int,
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """
    Convert a list of choices to matrix/tensor format for model fitting.

    Args:
        choices: Length-N sequence where each element is either:
            - int j: single choice of alternative j
            - tuple (s, t): dual choice of pair {s, t}
        J: Number of alternatives.

    Returns:
        y_single: Binary matrix (N, J)
        y_dual: Binary tensor (N, J, J) with upper triangle entries.
    """
    if J < 2:
        raise ValueError(f"J must be >= 2, got {J}")

    # Handle pandas objects gracefully
    if hasattr(choices, "to_numpy"):
        choices_seq = list(typing.cast(np.ndarray, choices.to_numpy()).tolist())
    elif isinstance(choices, np.ndarray):
        choices_seq = choices.tolist()
    else:
        choices_seq = list(choices)

    N = len(choices_seq)
    y_single = np.zeros((N, J), dtype=np.int8)
    y_dual = np.zeros((N, J, J), dtype=np.int8)

    for i, raw_choice in enumerate(choices_seq):
        choice = raw_choice
        if (
            isinstance(choice, (list, tuple, np.ndarray))
            and len(choice) == 2
            and not isinstance(choice, (np.integer, int))
        ):
            # Normalize list/array pairs to tuple
            choice = (choice[0], choice[1])

        if isinstance(choice, (int, np.integer)):
            j = int(choice)
            if not 0 <= j < J:
                raise ValueError(f"Choice {choice} at index {i} out of range [0, {J})")
            y_single[i, j] = 1
            continue

        if isinstance(choice, tuple) and len(choice) == 2:
            s = int(choice[0])
            t = int(choice[1])

            if s == t:
                raise ValueError(
                    f"Dual choice at index {i} has identical alternatives: {(s, t)}"
                )
            if not (0 <= s < J and 0 <= t < J):
                raise ValueError(f"Choice {(s, t)} at index {i} out of range [0, {J})")
            if s > t:
                s, t = t, s

            y_dual[i, s, t] = 1
            continue

        raise ValueError(
            f"Choice at index {i} must be int or tuple[int, int], got {raw_choice!r}"
        )

    return y_single, y_dual


def simulate_choices(
    N: int,
    J: int,
    K: int,
    true_beta: npt.NDArray[np.float64] | None = None,
    mix_ratio: float = 0.5,
    seed: int | None = 42,
    rng: np.random.Generator | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> tuple[
    npt.NDArray[np.float64], list[int | tuple[int, int]], npt.NDArray[np.float64]
]:
    """
    Simulate data and return a choices list alongside X and true parameters.

    Returns (X, choices, true_beta_free).
    """
    X, y_single, y_dual, true_beta_free = simulate_data(
        N=N,
        J=J,
        K=K,
        true_beta=true_beta,
        mix_ratio=mix_ratio,
        seed=seed,
        rng=rng,
        dtype=dtype,
    )

    choices: list[int | tuple[int, int]] = []
    for i in range(N):
        single_cols = np.flatnonzero(y_single[i])
        if single_cols.size == 1:
            choices.append(int(single_cols[0]))
            continue

        dual_indices = np.argwhere(y_dual[i])
        if dual_indices.shape[0] != 1:
            raise RuntimeError("Simulated data has invalid choice structure")
        s, t = dual_indices[0]
        choices.append((int(s), int(t)))

    return X, choices, true_beta_free


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
