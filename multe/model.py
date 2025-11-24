"""
Multichoice Logit Model

Vectorized implementation for fast and accurate MLE estimation.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp

# Numerical constants for stability and accuracy
CLIP_THRESHOLD = 1e-10  # Minimum probability value (avoid log(0))
HESSIAN_EPSILON = 1e-5  # Step size for Hessian finite differences


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
        self.coef_: Optional[npt.NDArray[np.float64]] = None
        self.optimization_result_: Optional[OptimizeResult] = None

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

    def _prepare_data(
        self,
        y_single: npt.NDArray[np.int8],
        y_dual: npt.NDArray[np.int8],
    ) -> tuple[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Pre-process data into sparse index format for faster iteration.
        """
        single_rows, single_cols = np.nonzero(y_single)
        dual_rows, dual_s, dual_t = np.nonzero(y_dual)
        return (single_rows, single_cols), (dual_rows, dual_s, dual_t)

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: npt.NDArray[np.int8],
        init_beta: Optional[npt.NDArray[np.float64]] = None,
        method: str = "L-BFGS-B",
        options: Optional[dict[str, Any]] = None,
    ) -> "MultichoiceLogit":
        """
        Fit the multichoice logit model using maximum likelihood estimation.

        This is a convenience method that wraps scipy.optimize.minimize with sensible
        defaults. For more control over optimization, you can call neg_log_likelihood
        and gradient directly with your own optimizer.

        Args:
            X (np.ndarray): Covariate matrix of shape (N, K).
            y_single (np.ndarray): Binary matrix (N, J). y_single[i, j] = 1 if i chose j alone.
            y_dual (np.ndarray): Binary tensor (N, J, J). y_dual[i, s, t] = 1 if i chose pair {s, t}.
            init_beta (np.ndarray, optional): Initial parameter values (flat array of size (J-1)*K).
                                              If None, initializes to zeros.
            method (str): Optimization method for scipy.optimize.minimize. Default is 'L-BFGS-B'.
                         Other good options: 'BFGS', 'Newton-CG'.
            options (dict, optional): Additional options to pass to the optimizer.
                                     Default is {'gtol': 1e-5, 'maxiter': 1000}.

        Returns:
            self: Returns the instance itself for method chaining.

        Raises:
            ValueError: If data dimensions are incompatible or constraints are violated.
            RuntimeError: If optimization fails to converge.
        """
        # Set default options
        if options is None:
            options = {"gtol": 1e-5, "maxiter": 1000}

        # Validate data
        self._validate_data(X, y_single, y_dual)

        # Prepare data indices once
        single_indices, dual_indices = self._prepare_data(y_single, y_dual)

        # Initialize parameters
        if init_beta is None:
            init_beta = np.zeros((self.J - 1) * self.K)
        else:
            # Validate initial parameters
            expected_size = (self.J - 1) * self.K
            if init_beta.size != expected_size:
                raise ValueError(
                    f"init_beta must have size {expected_size}, got {init_beta.size}"
                )

        # Run optimization using optimized internal functions
        result = minimize(
            fun=self._neg_log_likelihood_fast,
            jac=self._gradient_fast,
            x0=init_beta,
            args=(X, single_indices, dual_indices),
            method=method,
            options=options,
        )

        # Check convergence
        if not result.success:
            raise RuntimeError(
                f"Optimization failed to converge: {result.message}\n"
                f"Try a different optimization method or adjust tolerance."
            )

        # Store results
        self.coef_ = result.x.reshape(self.J - 1, self.K)
        self.optimization_result_ = result

        return self

    def _validate_data(
        self,
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: npt.NDArray[np.int8],
    ) -> None:
        """Validate input data dimensions and constraints."""
        N = X.shape[0]

        if X.shape[1] != self.K:
            raise ValueError(f"X must have {self.K} columns, got {X.shape[1]}")

        if y_single.shape != (N, self.J):
            raise ValueError(
                f"y_single must have shape ({N}, {self.J}), got {y_single.shape}"
            )

        if y_dual.shape != (N, self.J, self.J):
            raise ValueError(
                f"y_dual must have shape ({N}, {self.J}, {self.J}), got {y_dual.shape}"
            )

        # Check that each agent has exactly one choice
        single_choices = y_single.sum(axis=1)
        dual_choices = y_dual.sum(axis=(1, 2))
        total_choices = single_choices + dual_choices

        if not np.all(total_choices == 1):
            invalid = np.where(total_choices != 1)[0]
            raise ValueError(
                f"Each agent must have exactly one choice. "
                f"Agents with invalid choices: {invalid[:10]}"
                + ("..." if len(invalid) > 10 else "")
            )

        # Check that dual choices are in upper triangle (s < t)
        if np.any(y_dual):
            lower_triangle = np.tril_indices(self.J, k=0)
            if np.any(y_dual[:, lower_triangle[0], lower_triangle[1]]):
                raise ValueError(
                    "y_dual must only have entries in upper triangle (s < t)"
                )

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
            # We could use row_idx directly but logsumexp over all J is needed
            V_sub = V[row_idx]

            # Use logsumexp for numerical stability
            log_sum = logsumexp(V_sub, axis=1)

            log_lik += np.sum(V_chosen - log_sum)

        # 2. Handle Dual Choices
        if len(dual_indices[0]) > 0:
            i_idx, s_idx, t_idx = dual_indices

            # Get utilities for all dual choices at once
            a_s = a[i_idx, s_idx]  # Shape: (n_dual,)
            a_t = a[i_idx, t_idx]  # Shape: (n_dual,)

            # Compute R = sum(a) - a_s - a_t for all dual choices
            a_sum = np.sum(a[i_idx], axis=1)  # Shape: (n_dual,)
            R = a_sum - a_s - a_t  # Shape: (n_dual,)

            # Vectorized probability computation
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
        y_dual: npt.NDArray[np.int8],
    ) -> float:
        """
        Computes the negative log-likelihood of the model.
        Wrapper for public API compliance that computes indices on the fly.
        """
        self._validate_data(X, y_single, y_dual)
        single_indices, dual_indices = self._prepare_data(y_single, y_dual)
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

        # Gradient buffer for all parameters (J, K)
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
            # We can do this by subtracting p from y, but y is sparse indices
            # Easier: grad += X_sub.T @ (y_onehot - probs)
            # But creating y_onehot is (N_sub, J).
            # Memory efficient: grad += sum_i (delta_ij - p_ij) * x_i
            # grad += X_sub[y=1] - X_sub.T @ probs

            # Add positive term for chosen alternatives (y=1)
            # X_sub corresponds to row_idx.
            # We need to add X_i to grad[j] where j is chosen
            # Use np.add.at for sparse addition
            np.add.at(grad, col_idx, X_sub)

            # Subtract prob term for all alternatives
            # grad -= probs.T @ X_sub
            grad -= probs.T @ X_sub

        # 2. Dual Choice Gradient
        if len(dual_indices[0]) > 0:
            i_idx, s_idx, t_idx = dual_indices
            n_dual = len(i_idx)

            # Get covariates and utilities
            X_i = X[i_idx]  # Shape: (n_dual, K)
            a_s = a[i_idx, s_idx]  # Shape: (n_dual,)
            a_t = a[i_idx, t_idx]  # Shape: (n_dual,)

            # Compute R and denominators
            a_sum = np.sum(a[i_idx], axis=1)  # Shape: (n_dual,)
            R = a_sum - a_s - a_t

            D1 = a_s + R
            D2 = a_t + R
            D3 = a_s + a_t + R

            # Probability
            P_raw = (a_s / D1) + (a_t / D2) - ((a_s + a_t) / D3)

            # Clip mask
            clipped_mask = P_raw >= CLIP_THRESHOLD

            # Safe division
            inv_P = np.zeros_like(P_raw)
            inv_P[clipped_mask] = 1.0 / P_raw[clipped_mask]

            # Derivatives
            dP_dVs = a_s * R * (1 / (D1**2) - 1 / (D3**2))
            dP_dVt = a_t * R * (1 / (D2**2) - 1 / (D3**2))
            common_r = (a_s + a_t) / (D3**2) - a_s / (D1**2) - a_t / (D2**2)

            # Weights for s and t
            w_s = inv_P * dP_dVs
            w_t = inv_P * dP_dVt

            # Weights for r (all alternatives)
            w_r = inv_P * common_r

            # For 'r' alternatives: grad += sum_i (w_r_i * a_ij) * X_i
            # But we must exclude j=s and j=t

            # 1. Compute M = w_r[:, None] * a[i_idx]   (Shape: n_dual, J)
            M = w_r[:, np.newaxis] * a[i_idx]

            # 2. Zero out columns s and t for each row
            # Advanced indexing to set specific elements to 0
            rows = np.arange(n_dual)
            M[rows, s_idx] = 0.0
            M[rows, t_idx] = 0.0

            # 3. Compute gradient contribution from r-terms using matrix multiplication
            # grad += M.T @ X_i
            grad += M.T @ X_i

            # 4. Add contributions from s and t
            # grad[s] += sum(w_s * X_i)
            # grad[t] += sum(w_t * X_i)

            grad_contrib_s = w_s[:, np.newaxis] * X_i
            grad_contrib_t = w_t[:, np.newaxis] * X_i

            np.add.at(grad, s_idx, grad_contrib_s)
            np.add.at(grad, t_idx, grad_contrib_t)

        # Return negative gradient, remove fixed class 0
        return -grad[1:].flatten()

    def gradient(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: npt.NDArray[np.int8],
    ) -> npt.NDArray[np.float64]:
        """
        Computes the analytical gradient (Jacobian).
        Wrapper for public API compliance that computes indices on the fly.
        """
        self._validate_data(X, y_single, y_dual)
        single_indices, dual_indices = self._prepare_data(y_single, y_dual)
        return self._gradient_fast(flat_beta, X, single_indices, dual_indices)

    def compute_standard_errors(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: npt.NDArray[np.int8],
    ) -> npt.NDArray[np.float64]:
        """
        Computes standard errors by approximating the Hessian of the negative log-likelihood
        using central finite differences of the analytical gradient.
        """
        self._validate_data(X, y_single, y_dual)
        single_indices, dual_indices = self._prepare_data(y_single, y_dual)

        n_params = len(flat_beta)
        hessian = np.zeros((n_params, n_params))

        # Central finite differences for Hessian
        for j in range(n_params):
            beta_plus = flat_beta.copy()
            beta_plus[j] += HESSIAN_EPSILON

            beta_minus = flat_beta.copy()
            beta_minus[j] -= HESSIAN_EPSILON

            grad_plus = self._gradient_fast(beta_plus, X, single_indices, dual_indices)
            grad_minus = self._gradient_fast(
                beta_minus, X, single_indices, dual_indices
            )

            # Central difference: O(ε²) accuracy
            hessian[:, j] = (grad_plus - grad_minus) / (2 * HESSIAN_EPSILON)

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

        return std_errs
