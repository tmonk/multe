"""
Multichoice Logit Model

Vectorized implementation for fast and accurate MLE estimation.
Supports single and dual (pairwise) discrete choices.
"""

from __future__ import annotations

import typing
import warnings
from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp

# Numerical constants for stability and accuracy
CLIP_THRESHOLD = 1e-10  # Minimum probability value (avoid log(0))
HESSIAN_EPSILON = 1e-5  # Step size for Hessian finite differences

# Type alias for flexible dual-choice input formats
DualInput = (
    npt.NDArray[np.int8]
    | npt.NDArray[np.int64]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
    | sp.spmatrix
)


class MultichoiceLogit:
    """
    Multichoice Logit discrete choice model with vectorized operations.

    Supports both single choices (standard MNL) and dual/pairwise choices
    using an inclusion-exclusion probability formulation.

    Attributes:
        J (int): Total number of alternatives available.
        K (int): Number of covariates (features) for each alternative.
        coef_ (np.ndarray): Fitted coefficients of shape (J-1, K). Available after fit().
        optimization_result_ (OptimizeResult): Full optimization result. Available after fit().

    Example:
        >>> model = MultichoiceLogit(num_alternatives=3, num_covariates=2)
        >>> model.fit(X, y_single, y_dual)
        >>> print(model.coef_)
    """

    def __init__(self, num_alternatives: int, num_covariates: int) -> None:
        """
        Initialize the Multichoice Logit model dimensions.

        Args:
            num_alternatives: Total number of choices (J). Must be >= 2.
            num_covariates: Number of independent variables/features (K). Must be >= 1.

        Raises:
            ValueError: If num_alternatives < 2 or num_covariates < 1.
        """
        if num_alternatives < 2:
            raise ValueError(f"num_alternatives must be >= 2, got {num_alternatives}")
        if num_covariates < 1:
            raise ValueError(f"num_covariates must be >= 1, got {num_covariates}")

        self.J = num_alternatives
        self.K = num_covariates

        # Fitted attributes (set by fit method)
        self.coef_: npt.NDArray[np.float64] | None = None
        self.optimization_result_: OptimizeResult | None = None

    def _transform_params(
        self, flat_beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Reshape a flat parameter vector into a (J, K) matrix with identification constraint.

        The first alternative (index 0) is the reference category with coefficients
        fixed to zero for identification.

        Args:
            flat_beta: 1D array of learnable parameters, size (J-1) * K.

        Returns:
            Matrix of shape (J, K) where row 0 is zeros and rows 1..J-1
            contain the learned parameters.

        Raises:
            ValueError: If flat_beta has incorrect size.
        """
        expected_size = (self.J - 1) * self.K
        if flat_beta.size != expected_size:
            raise ValueError(
                f"flat_beta must have size {expected_size}, got {flat_beta.size}"
            )

        beta_free = flat_beta.reshape((self.J - 1, self.K))
        beta_fixed = np.zeros((1, self.K))
        return np.vstack([beta_fixed, beta_free])

    # Public alias for compatibility
    def transform_params(
        self, flat_beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Public wrapper for parameter reshaping (kept for compatibility)."""
        return self._transform_params(flat_beta)

    def _calculate_utilities(
        self, X: npt.NDArray[np.float64], beta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute deterministic utility V and exponentiated utility a.

        Uses the max-subtraction trick for numerical stability in exp().

        Args:
            X: Covariate matrix of shape (N, K).
            beta: Parameter matrix of shape (J, K).

        Returns:
            V: Deterministic utilities of shape (N, J).
            a: Exponentiated utilities exp(V_stable) of shape (N, J).
        """
        V = X @ beta.T
        # Numerical stability: subtract max V per row to avoid overflow
        V_stable = V - np.max(V, axis=1, keepdims=True)
        a = np.exp(V_stable)
        return V, a

    # Public alias for compatibility
    def calculate_utilities(
        self, X: npt.NDArray[np.float64], beta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Public wrapper for utility calculation (kept for compatibility)."""
        return self._calculate_utilities(X, beta)

    def _normalize_dual_indices(
        self, y_dual: DualInput, *, N: int, J: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert dual choice inputs into index triplets (rows, s, t).

        Accepts:
            - Dense tensors of shape (N, J, J)
            - Scipy sparse matrices of shape (N, J*J) with row-major flattening
            - Explicit index tuples (rows, s, t)

        Sparse flattening uses row-major order: column = s * J + t.

        Args:
            y_dual: Dual choice data in any supported format.
            N: Number of observations.
            J: Number of alternatives.

        Returns:
            Tuple of (row_indices, s_indices, t_indices) arrays.

        Raises:
            ValueError: If tuple format is invalid.
            TypeError: If y_dual type is unsupported.
        """
        if isinstance(y_dual, tuple):
            if len(y_dual) != 3:
                raise ValueError("y_dual tuple must have length 3 (rows, s, t)")
            rows, s_idx, t_idx = y_dual
            if not (len(rows) == len(s_idx) == len(t_idx)):
                raise ValueError("y_dual index arrays must have the same length")
            return (
                np.asarray(rows, dtype=np.int64),
                np.asarray(s_idx, dtype=np.int64),
                np.asarray(t_idx, dtype=np.int64),
            )

        if sp.issparse(y_dual):
            coo = y_dual.tocoo()
            rows = coo.row
            cols = coo.col
            s_idx = cols // J
            t_idx = cols % J
            return rows.astype(np.int64), s_idx.astype(np.int64), t_idx.astype(np.int64)

        if isinstance(y_dual, np.ndarray):
            dual_rows, dual_s, dual_t = np.nonzero(y_dual)
            return dual_rows, dual_s, dual_t

        raise TypeError(f"Unsupported y_dual type: {type(y_dual)}")

    def _validate_data(
        self,
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Validate input data and return prepared indices.

        Args:
            X: Covariate matrix (N, K).
            y_single: Single choice indicators (N, J).
            y_dual: Dual choice indicators in any supported format.

        Returns:
            single_indices: (row_indices, col_indices) for single choices.
            dual_indices: (row_indices, s_indices, t_indices) for dual choices.

        Raises:
            ValueError: If data dimensions or constraints are violated.
        """
        N = X.shape[0]

        if X.shape[1] != self.K:
            raise ValueError(f"X must have {self.K} columns, got {X.shape[1]}")

        if y_single.shape != (N, self.J):
            raise ValueError(
                f"y_single must have shape ({N}, {self.J}), got {y_single.shape}"
            )

        # Validate binary values
        if not np.isin(y_single, (0, 1)).all():
            raise ValueError("y_single must be binary (contain only 0 or 1).")

        # Normalize dual indices
        dual_rows, dual_s, dual_t = self._normalize_dual_indices(y_dual, N=N, J=self.J)

        # Bounds checks for dual indices
        if len(dual_rows) > 0:
            if dual_rows.max() >= N or dual_rows.min() < 0:
                raise ValueError("y_dual row indices out of bounds.")
            if dual_s.max() >= self.J or dual_t.max() >= self.J:
                raise ValueError("y_dual alternative indices out of bounds.")
            if dual_s.min() < 0 or dual_t.min() < 0:
                raise ValueError("y_dual alternative indices must be non-negative.")

            # Enforce upper triangle (s < t) and no diagonal
            if np.any(dual_s == dual_t):
                raise ValueError("y_dual diagonal must be zero (s != t).")
            if np.any(dual_s > dual_t):
                raise ValueError(
                    "y_dual must only have entries in upper triangle (s < t)."
                )

        # Binary checks for dense/sparse formats
        if isinstance(y_dual, np.ndarray):
            if y_dual.shape != (N, self.J, self.J):
                raise ValueError(
                    f"y_dual must have shape ({N}, {self.J}, {self.J}), got {y_dual.shape}"
                )
            if not np.isin(y_dual, (0, 1)).all():
                raise ValueError("y_dual must be binary (contain only 0 or 1).")
        elif sp.issparse(y_dual):
            if y_dual.shape != (N, self.J * self.J):
                raise ValueError(
                    f"sparse y_dual must have shape ({N}, {self.J * self.J})"
                )
            if y_dual.data.size and not np.isin(y_dual.data, (0, 1)).all():
                raise ValueError("Sparse y_dual must be binary.")

        # Check that each agent has exactly one choice
        single_rows, single_cols = np.nonzero(y_single)
        counts = np.bincount(single_rows, minlength=N)
        if len(dual_rows) > 0:
            counts += np.bincount(dual_rows, minlength=N)

        if not np.all(counts == 1):
            invalid = np.where(counts != 1)[0]
            raise ValueError(
                f"Each agent must have exactly one choice. "
                f"Agents with invalid choices: {invalid[:10]}"
                + ("..." if len(invalid) > 10 else "")
            )

        return (single_rows, single_cols), (dual_rows, dual_s, dual_t)

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
        init_beta: Optional[npt.NDArray[np.float64]] = None,
        method: str = "L-BFGS-B",
        options: Optional[dict[str, Any]] = None,
        bounds: Optional[Sequence[tuple[float | None, float | None]]] = None,
        constraints: Optional[Sequence[Any]] = None,
        num_restarts: int = 0,
        restart_scale: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> MultichoiceLogit:
        """
        Fit the multichoice logit model using maximum likelihood estimation.

        Args:
            X: Covariate matrix of shape (N, K).
            y_single: Binary matrix (N, J). y_single[i, j] = 1 if i chose j alone.
            y_dual: Dual choice indicators. Accepts:
                - Dense tensor (N, J, J) with y_dual[i, s, t] = 1 for pair {s, t}
                - Sparse matrix (N, J*J) with row-major flattening
                - Index triplet (rows, s, t)
            init_beta: Initial parameter values, flat array of size (J-1)*K.
                Defaults to zeros.
            method: Optimization method for scipy.optimize.minimize.
                Default 'L-BFGS-B'. Other options: 'BFGS', 'Newton-CG'.
            options: Additional options for the optimizer.
                Default: {'gtol': 1e-5, 'maxiter': 1000}.
            bounds: Parameter bounds for scipy.optimize.minimize.
            constraints: Constraints for scipy.optimize.minimize.
            num_restarts: Number of random restarts beyond init_beta.
            restart_scale: Standard deviation of normal noise for restart initialization.
            rng: Random generator for restarts. Uses default_rng() if None.

        Returns:
            self: The fitted model instance (for method chaining).

        Raises:
            ValueError: If data dimensions are incompatible or constraints violated.
            RuntimeError: If optimization fails to converge.
        """
        if options is None:
            options = {"gtol": 1e-5, "maxiter": 1000}

        # Validate data and prepare indices once
        single_indices, dual_indices = self._validate_data(X, y_single, y_dual)

        # Initialize parameters
        expected_size = (self.J - 1) * self.K
        if init_beta is None:
            init_beta = np.zeros(expected_size)
        elif init_beta.size != expected_size:
            raise ValueError(
                f"init_beta must have size {expected_size}, got {init_beta.size}"
            )

        rng = rng or np.random.default_rng()

        def run_optimization(start_beta: npt.NDArray[np.float64]) -> OptimizeResult:
            """Run optimization for a given starting point."""
            return minimize(
                fun=self._neg_log_likelihood,
                jac=self._gradient,
                x0=start_beta,
                args=(X, single_indices, dual_indices),
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )

        # Run optimizations and keep best result
        best_result: OptimizeResult | None = None
        start_points = [init_beta]
        if num_restarts > 0:
            noise = rng.normal(scale=restart_scale, size=(num_restarts, expected_size))
            start_points.extend(init_beta + noise_row for noise_row in noise)

        for start in start_points:
            # Run optimization
            result = run_optimization(start)
            if best_result is None or result.fun < best_result.fun:
                best_result = result

        assert best_result is not None

        if not best_result.success:
            raise RuntimeError(
                f"Optimization failed to converge: {best_result.message}\n"
                f"Try a different optimization method or adjust tolerance."
            )

        # Store results
        self.coef_ = best_result.x.reshape(self.J - 1, self.K)
        self.optimization_result_ = best_result

        return self

    def _neg_log_likelihood(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        single_indices: tuple[np.ndarray, np.ndarray],
        dual_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        """Compute negative log-likelihood using pre-computed indices."""
        beta = self._transform_params(flat_beta)
        V, a = self._calculate_utilities(X, beta)

        log_lik = 0.0

        # 1. Handle Single Choices (MNL)
        # single_indices = (row_idx, col_idx)
        if len(single_indices[0]) > 0:
            row_idx, col_idx = single_indices

            # Extract V for chosen alternatives
            V_chosen = V[row_idx, col_idx]

            # Extract full V for relevant rows
            V_sub = V[row_idx]

            # Use logsumexp for numerical stability (avoids overflow in exp)
            log_sum = logsumexp(V_sub, axis=1)

            log_lik += np.sum(V_chosen - log_sum)

        # 2. Handle Dual Choices
        if len(dual_indices[0]) > 0:
            # Extract all dual choice indices at once
            i_idx, s_idx, t_idx = dual_indices

            # Get utilities for all dual choices at once
            a_s = a[i_idx, s_idx]  # Shape: (n_dual,)
            a_t = a[i_idx, t_idx]  # Shape: (n_dual,)

            # Compute R = sum(a) - a_s - a_t for all dual choices
            a_sum = np.sum(a[i_idx], axis=1)  # Shape: (n_dual,)
            R = a_sum - a_s - a_t  # Shape: (n_dual,)

            # Vectorized probability computation using inclusion-exclusion:
            # P = a_s/(a_s+R) + a_t/(a_t+R) - (a_s+a_t)/(a_s+a_t+R)
            D1 = a_s + R
            D2 = a_t + R
            D3 = a_s + a_t + R

            probs = (a_s / D1) + (a_t / D2) - ((a_s + a_t) / D3)

            # Safety clipping for numerical stability
            probs = np.maximum(probs, CLIP_THRESHOLD)

            # Sum log probabilities
            log_lik += np.sum(np.log(probs))

        return -log_lik

    def neg_log_likelihood(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
    ) -> float:
        """
        Public wrapper: compute negative log-likelihood with on-the-fly indices.
        """
        single_indices, dual_indices = self._validate_data(X, y_single, y_dual)
        return self._neg_log_likelihood(flat_beta, X, single_indices, dual_indices)

    def _gradient(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        single_indices: tuple[np.ndarray, np.ndarray],
        dual_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> npt.NDArray[np.float64]:
        """Compute gradient using pre-computed indices and matrix multiplication."""
        beta = self._transform_params(flat_beta)
        V, a = self._calculate_utilities(X, beta)

        # Gradient buffer for all parameters (J, K); flattened later to ((J-1)*K,)
        grad = np.zeros((self.J, self.K))

        # 1. Single Choice Gradient (Standard MNL)
        if len(single_indices[0]) > 0:
            row_idx, col_idx = single_indices

            X_sub = X[row_idx]
            a_sub = a[row_idx]

            # Probabilities P(y=j) = exp(V_j) / sum(exp(V_k))
            probs = a_sub / np.sum(a_sub, axis=1, keepdims=True)

            # Gradient = X * (y - p)
            # y is one-hot, so for chosen col y=1, else 0
            # Add positive term for chosen alternatives (y=1)
            np.add.at(grad, col_idx, X_sub)

            # Subtract prob term for all alternatives
            grad -= probs.T @ X_sub

        # 2. Dual Choice Gradient
        if len(dual_indices[0]) > 0:
            i_idx, s_idx, t_idx = dual_indices
            n_dual = len(i_idx)

            # Covariates and utilities for dual choices
            X_i = X[i_idx]  # Shape: (n_dual, K)
            a_s = a[i_idx, s_idx]  # Shape: (n_dual,)
            a_t = a[i_idx, t_idx]  # Shape: (n_dual,)

            # Compute R and denominators
            a_sum = np.sum(a[i_idx], axis=1)  # Shape: (n_dual,)
            R = a_sum - a_s - a_t

            D1 = a_s + R
            D2 = a_t + R
            D3 = a_s + a_t + R

            # Probability (unclipped for gradient computation)
            P_raw = (a_s / D1) + (a_t / D2) - ((a_s + a_t) / D3)

            # Clip mask for safe division
            clipped_mask = P_raw >= CLIP_THRESHOLD
            inv_P = np.zeros_like(P_raw)
            inv_P[clipped_mask] = 1.0 / P_raw[clipped_mask]

            # Derivatives of P with respect to utilities
            dP_dVs = a_s * R * (1 / (D1**2) - 1 / (D3**2))  # (n_dual,)
            dP_dVt = a_t * R * (1 / (D2**2) - 1 / (D3**2))  # (n_dual,)
            common_r = (
                (a_s + a_t) / (D3**2) - a_s / (D1**2) - a_t / (D2**2)
            )  # (n_dual,)

            # Weights for s and t
            w_s = inv_P * dP_dVs  # (n_dual,)
            w_t = inv_P * dP_dVt  # (n_dual,)

            # Weights for r (all other alternatives)
            w_r = inv_P * common_r  # (n_dual,)
            # Complexity: O(n_dual) weight computations; still cheaper than looping over J.

            # For 'r' alternatives (neither s nor t): grad += sum_i (w_r_i * a_ij) * X_i
            # 1. Compute M = w_r[:, None] * a[i_idx]   (Shape: n_dual, J)
            a_i = a[i_idx]  # (n_dual, J)
            M = w_r[:, np.newaxis] * a_i

            # 2. Zero out columns s and t for each row
            rows = np.arange(n_dual)
            M[rows, s_idx] = 0.0
            M[rows, t_idx] = 0.0

            # 3. Compute gradient contribution from r-terms using matrix multiplication
            # This is an O(n_dual * J * K) dense step; faster than looping for typical J.
            grad += M.T @ X_i

            # 4. Add contributions from s and t (O(n_dual))
            grad_contrib_s = w_s[:, np.newaxis] * X_i  # (n_dual, K)
            grad_contrib_t = w_t[:, np.newaxis] * X_i  # (n_dual, K)

            np.add.at(grad, s_idx, grad_contrib_s)
            np.add.at(grad, t_idx, grad_contrib_t)

        # Return negative gradient for minimization, remove fixed class 0
        return -grad[1:].flatten()

    def compute_standard_errors(
        self,
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
        flat_beta: Optional[npt.NDArray[np.float64]] = None,
        epsilon: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute standard errors via numerical Hessian approximation.

        Uses central finite differences of the analytical gradient to
        approximate the Hessian, then inverts to get the covariance matrix.

        Args:
            X: Covariate matrix (N, K).
            y_single: Single-choice indicators (N, J).
            y_dual: Dual-choice indicators (dense, sparse, or index triplet).
            flat_beta: Parameter vector of size (J-1)*K. Uses fitted coef_ if None.
            epsilon: Finite-difference step size. Defaults to HESSIAN_EPSILON.

        Returns:
            1D array of standard errors of size (J-1)*K.

        Raises:
            ValueError: If model is unfitted and no parameters provided.

        Warns:
            RuntimeWarning: If Hessian inverse has negative diagonal elements.
        """
        if flat_beta is None:
            if self.coef_ is None:
                raise ValueError(
                    "Model is not fitted. Provide flat_beta or call fit() first."
                )
            flat_beta = self.coef_.flatten()

        single_indices, dual_indices = self._validate_data(X, y_single, y_dual)

        n_params = len(flat_beta)
        hessian = np.zeros((n_params, n_params))
        step = epsilon if epsilon is not None else HESSIAN_EPSILON

        # Central finite differences for Hessian
        for j in range(n_params):
            beta_plus = flat_beta.copy()
            beta_plus[j] += step

            beta_minus = flat_beta.copy()
            beta_minus[j] -= step

            grad_plus = self._gradient(beta_plus, X, single_indices, dual_indices)
            grad_minus = self._gradient(beta_minus, X, single_indices, dual_indices)

            # Central difference: O(ε²) accuracy
            hessian[:, j] = (grad_plus - grad_minus) / (2 * step)

        # Variance-Covariance Matrix is inverse of Hessian
        # Use pinv for stability with near-singular Hessians
        cov_matrix = np.linalg.pinv(hessian)

        # Check if diagonal elements are positive (valid variance)
        diag_cov = np.diag(cov_matrix)

        # If any variance is negative (numerical noise with singular hessian), warn
        if np.any(diag_cov < 0):
            warnings.warn(
                "Hessian inverse has negative diagonal elements. "
                "Standard errors may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Compute standard errors, setting invalid (negative variance) to NaN
        std_errs = np.sqrt(np.where(diag_cov >= 0, diag_cov, np.nan))
        return typing.cast(npt.NDArray[np.float64], std_errs)

    def predict_proba(
        self,
        X: npt.NDArray[np.float64],
        flat_beta: Optional[npt.NDArray[np.float64]] = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute predicted probabilities for single and dual choices.

        Args:
            X: Covariate matrix (N, K).
            flat_beta: Parameter vector of size (J-1)*K. Uses fitted coef_ if None.

        Returns:
            single_probs: (N, J) softmax probabilities for single choices.
            dual_probs: (N, J, J) probabilities for unordered pairs.
                Only upper triangle (s < t) is populated.

        Raises:
            ValueError: If model is unfitted and no parameters provided.
        """
        if flat_beta is None:
            if self.coef_ is None:
                raise ValueError(
                    "Model is not fitted. Provide flat_beta or call fit() first."
                )
            flat_beta = self.coef_.flatten()

        beta = self._transform_params(flat_beta)
        V, a = self._calculate_utilities(X, beta)

        # Single choice probabilities (softmax)
        single_probs = a / np.sum(a, axis=1, keepdims=True)

        # Dual choice probabilities: compute for each pair s<t
        N, J = X.shape[0], self.J
        dual_probs = np.zeros((N, J, J))
        a_sum = np.sum(a, axis=1)

        for s in range(J - 1):
            a_s = a[:, s]
            for t in range(s + 1, J):
                a_t = a[:, t]
                R = a_sum - a_s - a_t
                D1 = a_s + R
                D2 = a_t + R
                D3 = a_s + a_t + R
                probs = (a_s / D1) + (a_t / D2) - ((a_s + a_t) / D3)
                dual_probs[:, s, t] = np.maximum(probs, CLIP_THRESHOLD)

        return single_probs, dual_probs

    def log_likelihood_contributions(
        self,
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
        flat_beta: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute per-observation log-likelihood contributions.

        Useful for diagnostics, cross-validation, or computing information criteria.

        Args:
            X: Covariate matrix (N, K).
            y_single: Single-choice indicators (N, J).
            y_dual: Dual-choice indicators (dense, sparse, or index triplet).
            flat_beta: Parameter vector of size (J-1)*K. Uses fitted coef_ if None.

        Returns:
            Vector of per-observation log-likelihood contributions (N,).
        """
        if flat_beta is None:
            if self.coef_ is None:
                raise ValueError(
                    "Model is not fitted. Provide flat_beta or call fit() first."
                )
            flat_beta = self.coef_.flatten()

        single_indices, dual_indices = self._validate_data(X, y_single, y_dual)

        beta = self._transform_params(flat_beta)
        V, a = self._calculate_utilities(X, beta)
        contrib = np.zeros(X.shape[0])

        if len(single_indices[0]) > 0:
            row_idx, col_idx = single_indices
            V_chosen = V[row_idx, col_idx]
            log_sum = logsumexp(V[row_idx], axis=1)
            contrib[row_idx] = V_chosen - log_sum

        if len(dual_indices[0]) > 0:
            i_idx, s_idx, t_idx = dual_indices
            a_s = a[i_idx, s_idx]
            a_t = a[i_idx, t_idx]
            a_sum = np.sum(a[i_idx], axis=1)
            R = a_sum - a_s - a_t
            probs = (
                (a_s / (a_s + R)) + (a_t / (a_t + R)) - ((a_s + a_t) / (a_s + a_t + R))
            )
            contrib[i_idx] = np.log(np.maximum(probs, CLIP_THRESHOLD))
            # contrib fills per-observation slots; untouched rows remain zero if no dual/single

        return contrib

    def log_likelihood(
        self,
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
        flat_beta: Optional[npt.NDArray[np.float64]] = None,
    ) -> float:
        """
        Compute the total log-likelihood of the model.

        Args:
            X: Covariate matrix (N, K).
            y_single: Single-choice indicators (N, J).
            y_dual: Dual-choice indicators (dense, sparse, or index triplet).
            flat_beta: Parameter vector of size (J-1)*K. Uses fitted coef_ if None.

        Returns:
            Scalar log-likelihood (not negated).
        """
        return float(
            np.sum(self.log_likelihood_contributions(X, y_single, y_dual, flat_beta))
        )
