"""
Multichoice Logit Model

Vectorized implementation for fast and accurate MLE estimation.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, OptimizeResult
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

    def transform_params(self, flat_beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: npt.NDArray[np.int8],
        init_beta: Optional[npt.NDArray[np.float64]] = None,
        method: str = "L-BFGS-B",
        options: Optional[Dict[str, Any]] = None,
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
                                     Default is {'gtol': 1e-5, 'maxiter': 1000, 'disp': False}.

        Returns:
            self: Returns the instance itself for method chaining.

        Raises:
            ValueError: If data dimensions are incompatible or constraints are violated.
            RuntimeError: If optimization fails to converge.

        Example:
            >>> from multe import MultichoiceLogit, simulate_data
            >>> X, y_single, y_dual, true_beta = simulate_data(N=1000, J=3, K=2)
            >>> model = MultichoiceLogit(num_alternatives=3, num_covariates=2)
            >>> model.fit(X, y_single, y_dual)
            >>> print(model.coef_)  # Estimated coefficients
        """
        # Set default options
        if options is None:
            options = {"gtol": 1e-5, "maxiter": 1000, "disp": False}

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

        # Run optimization
        result = minimize(
            fun=self.neg_log_likelihood,
            jac=self.gradient,
            x0=init_beta,
            args=(X, y_single, y_dual),
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

    def neg_log_likelihood(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: npt.NDArray[np.int8],
    ) -> float:
        """
        Computes the negative log-likelihood of the model for minimization.
        Vectorized version for faster computation.

        Args:
            flat_beta (np.ndarray): 1D array of parameters optimization is performed on.
            X (np.ndarray): Covariates of shape (N, K).
            y_single (np.ndarray): Binary matrix (N, J). y_single[i, j] = 1 if i chose j alone.
            y_dual (np.ndarray): Binary tensor (N, J, J). y_dual[i, s, t] = 1 if i chose pair {s, t}.

        Returns:
            float: The negative log-likelihood value (scalar).

        Raises:
            ValueError: If data dimensions are incompatible or constraints are violated.
        """
        self._validate_data(X, y_single, y_dual)
        beta = self.transform_params(flat_beta)
        V, a = self.calculate_utilities(X, beta)

        log_lik = 0.0

        # 1. Handle Single Choices (MNL)
        single_choice_mask = y_single.sum(axis=1) > 0
        if np.any(single_choice_mask):
            V_sub = V[single_choice_mask]
            y_sub = y_single[single_choice_mask]
            term1 = np.sum(y_sub * V_sub)
            # Use logsumexp for numerical stability (avoids overflow in exp)
            term2 = np.sum(logsumexp(V_sub, axis=1))
            log_lik += (term1 - term2)

        # 2. Handle Dual Choices
        dual_indices = np.argwhere(y_dual > 0)

        if len(dual_indices) > 0:
            # Extract all dual choice indices at once
            i_idx = dual_indices[:, 0]
            s_idx = dual_indices[:, 1]
            t_idx = dual_indices[:, 2]

            # Get utilities for all dual choices at once
            a_s = a[i_idx, s_idx]  # Shape: (n_dual,)
            a_t = a[i_idx, t_idx]  # Shape: (n_dual,)

            # Compute R = sum(a) - a_s - a_t for all dual choices
            a_sum = np.sum(a[i_idx], axis=1)  # Shape: (n_dual,)
            R = a_sum - a_s - a_t  # Shape: (n_dual,)

            # Vectorized probability computation
            # P = a_s/(a_s+R) + a_t/(a_t+R) - (a_s+a_t)/(a_s+a_t+R)
            p1 = a_s / (a_s + R)
            p2 = a_t / (a_t + R)
            p3 = (a_s + a_t) / (a_s + a_t + R)

            probs = p1 + p2 - p3

            # Safety clipping for numerical stability
            probs = np.maximum(probs, CLIP_THRESHOLD)

            # Sum log probabilities
            log_lik += np.sum(np.log(probs))

        return -log_lik

    def gradient(
        self,
        flat_beta: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y_single: npt.NDArray[np.int8],
        y_dual: npt.NDArray[np.int8],
    ) -> npt.NDArray[np.float64]:
        """
        Computes the analytical gradient (Jacobian) of the negative log-likelihood.

        Args:
            flat_beta (np.ndarray): 1D array of parameters.
            X (np.ndarray): Covariates of shape (N, K).
            y_single (np.ndarray): Single choice indicators (N, J).
            y_dual (np.ndarray): Dual choice indicators (N, J, J).

        Returns:
            np.ndarray: Flattened gradient vector of shape ((J-1)*K, ).

        Raises:
            ValueError: If data dimensions are incompatible or constraints are violated.
        """
        self._validate_data(X, y_single, y_dual)
        beta = self.transform_params(flat_beta)
        V, a = self.calculate_utilities(X, beta)

        # Gradient buffer for all parameters (J, K)
        grad = np.zeros((self.J, self.K))

        # 1. Single Choice Gradient (Standard MNL) - already vectorized
        single_mask = y_single.sum(axis=1) > 0
        if np.any(single_mask):
            X_sub = X[single_mask]
            a_sub = a[single_mask]
            y_sub = y_single[single_mask]
            probs = a_sub / np.sum(a_sub, axis=1, keepdims=True)
            error = y_sub - probs
            grad += error.T @ X_sub

        # 2. Dual Choice Gradient
        dual_indices = np.argwhere(y_dual > 0)

        if len(dual_indices) > 0:
            # Extract indices
            i_idx = dual_indices[:, 0]
            s_idx = dual_indices[:, 1]
            t_idx = dual_indices[:, 2]

            n_dual = len(i_idx)

            # Get covariates and utilities for all dual choices
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
            P_raw = (a_s/D1) + (a_t/D2) - ((a_s+a_t)/D3)

            # Clip for numerical stability (use module constant)
            P = np.maximum(P_raw, CLIP_THRESHOLD)  # Clipped for likelihood

            # Mask for where clipping occurred (gradient should be 0)
            clipped_mask = P_raw >= CLIP_THRESHOLD

            # Precompute derivatives
            dP_dVs = a_s * R * (1/(D1**2) - 1/(D3**2))  # Shape: (n_dual,)
            dP_dVt = a_t * R * (1/(D2**2) - 1/(D3**2))  # Shape: (n_dual,)
            common_r = (a_s+a_t)/(D3**2) - a_s/(D1**2) - a_t/(D2**2)  # Shape: (n_dual,)

            # Compute gradient contributions for s and t
            # For alternative s (gradient = 0 if probability was clipped)
            grad_weight_s = np.where(clipped_mask, (1/P_raw) * dP_dVs, 0.0)  # Shape: (n_dual,)
            grad_contrib_s = grad_weight_s[:, np.newaxis] * X_i  # Shape: (n_dual, K)

            # For alternative t (gradient = 0 if probability was clipped)
            grad_weight_t = np.where(clipped_mask, (1/P_raw) * dP_dVt, 0.0)  # Shape: (n_dual,)
            grad_contrib_t = grad_weight_t[:, np.newaxis] * X_i  # Shape: (n_dual, K)

            # Accumulate gradient for s alternatives
            for j in range(self.J):
                mask_s = (s_idx == j)
                if np.any(mask_s):
                    grad[j] += np.sum(grad_contrib_s[mask_s], axis=0)

                mask_t = (t_idx == j)
                if np.any(mask_t):
                    grad[j] += np.sum(grad_contrib_t[mask_t], axis=0)

            # For alternatives that are neither s nor t (the 'r' alternatives)
            # Create mask for each alternative
            a_i = a[i_idx]  # Shape: (n_dual, J)
            # Gradient = 0 if probability was clipped
            grad_weight_r = np.where(clipped_mask, (1/P_raw) * common_r, 0.0)  # Shape: (n_dual,)

            for j in range(self.J):
                # Mask where j is neither s nor t
                mask_r = (s_idx != j) & (t_idx != j)
                if np.any(mask_r):
                    # Get a_j values where j is 'r'
                    a_j = a_i[mask_r, j]  # Shape: (n_mask,)
                    grad_contrib_r = (grad_weight_r[mask_r] * a_j)[:, np.newaxis] * X_i[mask_r]
                    grad[j] += np.sum(grad_contrib_r, axis=0)

        # Return negative gradient for minimization, remove fixed class 0
        return -grad[1:].flatten()

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

        Uses central differences for O(ε²) accuracy vs O(ε) for forward differences.

        Args:
            flat_beta (np.ndarray): Optimal parameters (flattened).
            X, y_single, y_dual: Data required for gradient calculation.

        Returns:
            np.ndarray: Standard errors for the parameters.

        Raises:
            ValueError: If data dimensions are incompatible or constraints are violated.
            RuntimeWarning: If Hessian is singular or ill-conditioned.
        """
        self._validate_data(X, y_single, y_dual)
        n_params = len(flat_beta)
        hessian = np.zeros((n_params, n_params))

        # Central finite differences for Hessian (more accurate than forward)
        for j in range(n_params):
            beta_plus = flat_beta.copy()
            beta_plus[j] += HESSIAN_EPSILON

            beta_minus = flat_beta.copy()
            beta_minus[j] -= HESSIAN_EPSILON

            grad_plus = self.gradient(beta_plus, X, y_single, y_dual)
            grad_minus = self.gradient(beta_minus, X, y_single, y_dual)

            # Central difference: O(ε²) accuracy
            hessian[:, j] = (grad_plus - grad_minus) / (2 * HESSIAN_EPSILON)

        # Variance-Covariance Matrix is inverse of Hessian
        try:
            cov_matrix = np.linalg.inv(hessian)
            std_errs = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError as e:
            import warnings

            warnings.warn(
                "Hessian is singular or ill-conditioned. Standard errors may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
            std_errs = np.full(n_params, np.nan)

        return std_errs
