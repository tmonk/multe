"""
Multichoice Logit Model

Vectorized implementation for fast and accurate MLE estimation.
"""

from __future__ import annotations

import typing
from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp

# Numerical constants for stability and accuracy
CLIP_THRESHOLD = 1e-10  # Minimum probability value (avoid log(0))
HESSIAN_EPSILON = 1e-5  # Step size for Hessian finite differences

DualInput = (
    npt.NDArray[np.int8]
    | npt.NDArray[np.int64]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
    | sp.spmatrix
)


class MultichoiceLogit:
    """
    Multichoice Logit discrete choice model with vectorized operations.

    Attributes:
        J (int): Total number of alternatives available.
        K (int): Number of covariates (features) for each alternative.
        coef_ (np.ndarray): Fitted coefficients of shape (J-1, K). Available after fit().
        optimization_result_ (OptimizeResult): Full optimization result. Available after fit().
    """

    def __init__(self, num_alternatives: int, num_covariates: int) -> None:
        """
        Initialize the Multichoice Logit model dimensions.

        Args:
            num_alternatives (int): Total number of choices (J).
            num_covariates (int): Number of independent variables/features (K).

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

    def transform_params(
        self, flat_beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Reshapes a flat parameter vector into a (J, K) matrix, handling identification.

        Args:
            flat_beta (np.ndarray): 1D array of learnable parameters.
                                    Size should be (J-1) * K.

        Returns:
            np.ndarray: Matrix of shape (J, K) where row 0 is all zeros and
                        rows 1..J-1 contain the learned parameters.

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

    def calculate_utilities(
        self, X: npt.NDArray[np.float64], beta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Computes the deterministic utility V and the exponentiated utility a.

        Args:
            X (np.ndarray): Covariate matrix of shape (N, K).
            beta (np.ndarray): Parameter matrix of shape (J, K).

        Returns:
            tuple:
                - V (np.ndarray): Deterministic utilities (N, J).
                - a (np.ndarray): Exponentiated utilities exp(V) (N, J).
        """
        V = X @ beta.T
        # Numerical stability: subtract max V per row to avoid overflow
        V_stable = V - np.max(V, axis=1, keepdims=True)
        a = np.exp(V_stable)
        return V, a

    def _normalize_dual_indices(
        self, y_dual: DualInput, *, N: int, J: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert dual choice inputs into index triplets (rows, s, t).
        Accepts dense tensors, scipy sparse matrices (shape N x J*J),
        or explicit index tuples.

        Sparse flattening uses row-major order: column = s * J + t.
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

        raise TypeError("Unsupported y_dual type")

    def _validate_binary(self, arr: np.ndarray, name: str) -> None:
        """Ensure array contains only 0/1 values."""
        if not np.isin(arr, (0, 1)).all():
            raise ValueError(f"{name} must be binary (contain only 0 or 1).")

    def _validate_data(
        self,
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
        *,
        dual_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate input data dimensions and constraints."""
        N = X.shape[0]

        if X.shape[1] != self.K:
            raise ValueError(f"X must have {self.K} columns, got {X.shape[1]}")

        if y_single.shape != (N, self.J):
            raise ValueError(
                f"y_single must have shape ({N}, {self.J}), got {y_single.shape}"
            )

        self._validate_binary(y_single, "y_single")

        # Normalize dual indices for validation
        if dual_indices is None:
            dual_rows, dual_s, dual_t = self._normalize_dual_indices(
                y_dual, N=N, J=self.J
            )
        else:
            dual_rows, dual_s, dual_t = dual_indices

        # Bounds checks
        if len(dual_rows) > 0:
            if dual_rows.max() >= N or dual_rows.min() < 0:
                raise ValueError("y_dual row indices out of bounds.")
            if dual_s.max() >= self.J or dual_t.max() >= self.J or dual_s.min() < 0:
                raise ValueError("y_dual alternative indices out of bounds.")

        # Enforce upper triangle only and no diagonal
        if np.any(dual_s == dual_t):
            raise ValueError("y_dual diagonal must be zero.")
        if np.any(dual_s > dual_t):
            # Check that dual choices are in upper triangle (s < t)
            raise ValueError("y_dual must only have entries in upper triangle (s < t).")
        # Check that dual choices are in upper triangle (s < t)

        # Binary checks for dense/sparse
        if isinstance(y_dual, np.ndarray):
            if y_dual.shape != (N, self.J, self.J):
                raise ValueError(
                    f"y_dual must have shape ({N}, {self.J}, {self.J}), got {y_dual.shape}"
                )
            self._validate_binary(y_dual, "y_dual")
        elif sp.issparse(y_dual):
            if y_dual.shape != (N, self.J * self.J):
                raise ValueError(
                    f"sparse y_dual must have shape ({N}, {self.J * self.J})"
                )
            if y_dual.data.size and not np.isin(y_dual.data, (0, 1)).all():
                raise ValueError("Sparse y_dual must be binary.")

        # Check that each agent has exactly one choice
        single_rows = np.nonzero(y_single)[0]
        counts = np.bincount(single_rows, minlength=N)
        if len(dual_rows) > 0:
            counts += np.bincount(dual_rows, minlength=N)

        # Check that each agent has exactly one choice
        if not np.all(counts == 1):
            invalid = np.where(counts != 1)[0]
            raise ValueError(
                f"Each agent must have exactly one choice. "
                f"Agents with invalid choices: {invalid[:10]}"
                + ("..." if len(invalid) > 10 else "")
            )
        return dual_rows, dual_s, dual_t

    def _prepare_data(
        self,
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
        *,
        N: Optional[int] = None,
        J: Optional[int] = None,
        dual_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Pre-process data into sparse index format for faster iteration.
        Supports dense (N,J,J) tensors, scipy sparse matrices of shape (N, J*J),
        or explicit index tuples (rows, s, t).
        """
        single_rows, single_cols = np.nonzero(y_single)
        if dual_indices is None:
            dual_rows, dual_s, dual_t = self._normalize_dual_indices(
                y_dual, N=N if N is not None else y_single.shape[0], J=J or self.J
            )
        else:
            dual_rows, dual_s, dual_t = dual_indices
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
    ) -> "MultichoiceLogit":
        """
        Fit the multichoice logit model using maximum likelihood estimation.

        This is a convenience method that wraps scipy.optimize.minimize with sensible
        defaults. For more control over optimization, you can call neg_log_likelihood
        and gradient directly with your own optimizer.

        Args:
            X (np.ndarray): Covariate matrix of shape (N, K).
            y_single (np.ndarray): Binary matrix (N, J). y_single[i, j] = 1 if i chose j alone.
            y_dual (DualInput): Binary tensor (N, J, J), sparse matrix (N, J*J),
                                or index triplet (rows, s, t) for dual choices.
            init_beta (np.ndarray, optional): Initial parameter values (flat array of size (J-1)*K).
                                              If None, initializes to zeros.
            method (str): Optimization method for scipy.optimize.minimize. Default is 'L-BFGS-B'.
                         Other good options: 'BFGS', 'Newton-CG'.
            options (dict, optional): Additional options to pass to the optimizer.
                                     Default is {'gtol': 1e-5, 'maxiter': 1000}.
            bounds (Sequence, optional): Bounds to pass to scipy.optimize.minimize.
            constraints (Sequence, optional): Constraints to pass to minimize.
            num_restarts (int): Number of random restarts to perform beyond init_beta.
            restart_scale (float): Scale of normal noise for restart initialization.
            rng (np.random.Generator, optional): Random generator for restarts.

        Returns:
            self: Returns the instance itself for method chaining.

        Raises:
            ValueError: If data dimensions are incompatible or constraints are violated.
            RuntimeError: If optimization fails to converge.

        Example:
            >>> from multe import MultichoiceLogit, simulate_data
            >>> X, y_single, y_dual, _ = simulate_data(N=1000, J=3, K=2)
            >>> model = MultichoiceLogit(num_alternatives=3, num_covariates=2)
            >>> model.fit(X, y_single, y_dual)
            >>> print(model.coef_)  # Estimated coefficients
        """
        # Set default options
        if options is None:
            options = {"gtol": 1e-5, "maxiter": 1000}

        # Validate data and prepare indices once
        dual_indices = self._validate_data(X, y_single, y_dual)
        single_indices, dual_indices = self._prepare_data(
            y_single, y_dual, N=X.shape[0], J=self.J, dual_indices=dual_indices
        )

        # Initialize parameters
        expected_size = (self.J - 1) * self.K
        if init_beta is None:
            init_beta = np.zeros(expected_size)
        else:
            # Validate initial parameters
            if init_beta.size != expected_size:
                raise ValueError(
                    f"init_beta must have size {expected_size}, got {init_beta.size}"
                )

        rng = rng or np.random.default_rng()

        def run_optimization(start_beta: npt.NDArray[np.float64]) -> OptimizeResult:
            # Run optimization for a given starting point
            return minimize(
                fun=self._neg_log_likelihood_fast,
                jac=self._gradient_fast,
                x0=start_beta,
                args=(X, single_indices, dual_indices),
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )

        best_result: OptimizeResult | None = None
        start_points = [init_beta]
        if num_restarts > 0:
            noise = rng.normal(scale=restart_scale, size=(num_restarts, expected_size))
            start_points.extend(init_beta + noise_i for noise_i in noise)

        for start in start_points:
            # Run optimization
            result = run_optimization(start)
            if best_result is None or result.fun < best_result.fun:
                best_result = result

        assert best_result is not None

        # Check convergence
        if not best_result.success:
            raise RuntimeError(
                f"Optimization failed to converge: {best_result.message}\n"
                f"Try a different optimization method or adjust tolerance."
            )

        # Store results
        self.coef_ = best_result.x.reshape(self.J - 1, self.K)
        self.optimization_result_ = best_result

        return self

    def _neg_log_likelihood_fast(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        single_indices: tuple[np.ndarray, np.ndarray],
        dual_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        """
        Internal optimized NLL using pre-computed indices.
        """
        beta = self.transform_params(flat_beta)
        V, a = self.calculate_utilities(X, beta)

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
            # Vectorized probability computation
            D1 = a_s + R
            D2 = a_t + R
            D3 = a_s + a_t + R

            probs = (a_s / D1) + (a_t / D2) - ((a_s + a_t) / D3)

            # Safety clipping for numerical stability
            probs = np.maximum(probs, CLIP_THRESHOLD)  # Clipped for likelihood

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
        Computes the negative log-likelihood of the model.
        Wrapper for public API compliance that computes indices on the fly.

        Args:
            flat_beta: Parameter vector ((J-1)*K).
            X: Covariate matrix (N, K).
            y_single: Single-choice indicators (N, J).
            y_dual: Dual-choice indicators (dense, sparse, or index triplet).

        Returns:
            Scalar negative log-likelihood.
        """
        dual_indices = self._validate_data(X, y_single, y_dual)
        single_indices, dual_indices = self._prepare_data(
            y_single, y_dual, N=X.shape[0], J=self.J, dual_indices=dual_indices
        )
        return self._neg_log_likelihood_fast(flat_beta, X, single_indices, dual_indices)

    def _gradient_fast(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        single_indices: tuple[np.ndarray, np.ndarray],
        dual_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> npt.NDArray[np.float64]:
        """
        Internal optimized gradient using pre-computed indices and matrix multiplication.
        """
        beta = self.transform_params(flat_beta)
        V, a = self.calculate_utilities(X, beta)

        # Gradient buffer for all parameters (J, K); flattened later to ((J-1)*K,)
        # Gradient buffer for all parameters (J, K)
        grad = np.zeros((self.J, self.K))

        # 1. Single Choice Gradient (Standard MNL)
        if len(single_indices[0]) > 0:
            row_idx, col_idx = single_indices

            X_sub = X[row_idx]
            a_sub = a[row_idx]

            # Probabilities P(y=j) = exp(V_j) / sum(exp(V_k))
            probs = a_sub / np.sum(a_sub, axis=1, keepdims=True)

            # Add positive term for chosen alternatives (y=1)
            np.add.at(grad, col_idx, X_sub)

            # Subtract prob term for all alternatives
            grad -= probs.T @ X_sub
        # 2. Dual Choice Gradient
        if len(dual_indices[0]) > 0:
            i_idx, s_idx, t_idx = dual_indices
            n_dual = len(i_idx)

            # Covariates/utilities for dual choices
            X_i = X[i_idx]  # (n_dual, K)
            a_s = a[i_idx, s_idx]  # (n_dual,)
            a_t = a[i_idx, t_idx]  # (n_dual,)

            a_sum = np.sum(a[i_idx], axis=1)  # (n_dual,)
            R = a_sum - a_s - a_t

            D1 = a_s + R
            D2 = a_t + R
            D3 = a_s + a_t + R

            # Probability (unclipped for gradient computation)
            P_raw = (a_s / D1) + (a_t / D2) - ((a_s + a_t) / D3)

            clipped_mask = P_raw >= CLIP_THRESHOLD
            inv_P = np.zeros_like(P_raw)
            inv_P[clipped_mask] = 1.0 / P_raw[clipped_mask]

            # Derivatives
            dP_dVs = a_s * R * (1 / (D1**2) - 1 / (D3**2))  # (n_dual,)
            dP_dVt = a_t * R * (1 / (D2**2) - 1 / (D3**2))  # (n_dual,)
            common_r = (
                (a_s + a_t) / (D3**2) - a_s / (D1**2) - a_t / (D2**2)
            )  # (n_dual,)

            w_s = inv_P * dP_dVs  # (n_dual,)
            w_t = inv_P * dP_dVt  # (n_dual,)
            w_r = inv_P * common_r  # (n_dual,)
            # Complexity: O(n_dual) weight computations; still cheaper than looping over J.

            # 'r' alternatives (neither s nor t): grad += sum_i (w_r_i * a_ij) * X_i
            a_i = a[i_idx]  # (n_dual, J)
            M = w_r[:, np.newaxis] * a_i
            rows = np.arange(n_dual)
            M[rows, s_idx] = 0.0
            M[rows, t_idx] = 0.0

            # This is an O(n_dual * J * K) dense step; faster than looping for typical J.
            grad += M.T @ X_i

            # Add contributions from s and t (O(n_dual))
            grad_contrib_s = w_s[:, np.newaxis] * X_i  # (n_dual, K)
            grad_contrib_t = w_t[:, np.newaxis] * X_i  # (n_dual, K)

            np.add.at(grad, s_idx, grad_contrib_s)
            np.add.at(grad, t_idx, grad_contrib_t)
        # Return negative gradient for minimization, remove fixed class 0
        return -grad[1:].flatten()

    def gradient(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
    ) -> npt.NDArray[np.float64]:
        """
        Computes the analytical gradient (Jacobian).
        Wrapper for public API compliance that computes indices on the fly.

        Args:
            flat_beta: Parameter vector ((J-1)*K).
            X: Covariate matrix (N, K).
            y_single: Single-choice indicators (N, J).
            y_dual: Dual-choice indicators (dense, sparse, or index triplet).

        Returns:
            Gradient vector ((J-1)*K,).
        """
        dual_indices = self._validate_data(X, y_single, y_dual)
        single_indices, dual_indices = self._prepare_data(
            y_single, y_dual, N=X.shape[0], J=self.J, dual_indices=dual_indices
        )
        return self._gradient_fast(flat_beta, X, single_indices, dual_indices)

    def compute_standard_errors(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
        *,
        epsilon: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Computes standard errors by approximating the Hessian of the negative log-likelihood
        using central finite differences of the analytical gradient.

        Args:
            flat_beta: Parameter vector ((J-1)*K).
            X: Covariate matrix (N, K).
            y_single: Single-choice indicators (N, J).
            y_dual: Dual-choice indicators (dense, sparse, or index triplet).
            epsilon: Optional finite-difference step; defaults to HESSIAN_EPSILON.

        Returns:
            1D array of standard errors ((J-1)*K,).

        Raises:
            ValueError: If data validation fails.
        """
        dual_indices = self._validate_data(X, y_single, y_dual)
        single_indices, dual_indices = self._prepare_data(
            y_single, y_dual, N=X.shape[0], J=self.J, dual_indices=dual_indices
        )

        n_params = len(flat_beta)
        hessian = np.zeros((n_params, n_params))
        step = HESSIAN_EPSILON if epsilon is None else float(epsilon)

        # Central finite differences for Hessian (more accurate than forward)
        for j in range(n_params):
            beta_plus = flat_beta.copy()
            beta_plus[j] += step

            beta_minus = flat_beta.copy()
            beta_minus[j] -= step

            grad_plus = self._gradient_fast(beta_plus, X, single_indices, dual_indices)
            grad_minus = self._gradient_fast(
                beta_minus, X, single_indices, dual_indices
            )

            # Central difference: O(ε²) accuracy
            hessian[:, j] = (grad_plus - grad_minus) / (2 * step)

        # Variance-Covariance Matrix is inverse of Hessian
        # Use pinv for stability with near-singular Hessians
        cov_matrix = np.linalg.pinv(hessian)

        # Check if diagonal elements are positive (valid variance)
        diag_cov = np.diag(cov_matrix)

        # If any variance is negative (numerical noise with singular hessian), warn
        if np.any(diag_cov < 0):
            import warnings

            warnings.warn(
                "Hessian inverse has negative diagonal elements. Standard errors may be unreliable.",
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
        Compute predicted probabilities for single and dual choices for new data.

        Args:
            X: Covariate matrix (N, K)
            flat_beta: Optional parameter vector ((J-1)*K). Uses fitted coef_ if None.

        Returns:
            single_probs: (N, J) softmax probabilities for single choices
            dual_probs: (N, J, J) probabilities for unordered pairs (upper triangle populated)

        Raises:
            ValueError: If the model is unfitted and no parameters are provided.
        """
        if flat_beta is None:
            if self.coef_ is None:
                raise ValueError("Model is not fitted. Provide flat_beta or call fit.")
            flat_beta = self.coef_.flatten()

        beta = self.transform_params(flat_beta)
        V, a = self.calculate_utilities(X, beta)

        single_probs = a / np.sum(a, axis=1, keepdims=True)

        # Dual probabilities: compute for each pair s<t
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
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
    ) -> npt.NDArray[np.float64]:
        """
        Return per-observation log-likelihood contributions.

        Args:
            flat_beta: Parameter vector ((J-1)*K).
            X: Covariate matrix (N, K).
            y_single: Single-choice indicators (N, J).
            y_dual: Dual-choice indicators (dense, sparse, or index triplet).

        Returns:
            Vector of per-observation log-likelihood contributions (N,).
        """
        dual_indices = self._validate_data(X, y_single, y_dual)
        single_indices, dual_indices = self._prepare_data(
            y_single, y_dual, N=X.shape[0], J=self.J, dual_indices=dual_indices
        )
        beta = self.transform_params(flat_beta)
        V, a = self.calculate_utilities(X, beta)
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
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: DualInput,
    ) -> float:
        """Convenience wrapper returning the sum of log-likelihood contributions."""
        return float(
            np.sum(self.log_likelihood_contributions(flat_beta, X, y_single, y_dual))
        )
