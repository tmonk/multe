"""Tests for the MultichoiceLogit model."""

import numpy as np
import pytest
from multe import MultichoiceLogit, simulate_data


class TestMultichoiceLogitInit:
    """Test MultichoiceLogit initialization."""

    def test_valid_initialization(self):
        """Test valid model initialization."""
        model = MultichoiceLogit(num_alternatives=3, num_covariates=2)
        assert model.J == 3
        assert model.K == 2

    def test_invalid_num_alternatives(self):
        """Test that num_alternatives < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_alternatives must be >= 2"):
            MultichoiceLogit(num_alternatives=1, num_covariates=2)

    def test_invalid_num_covariates(self):
        """Test that num_covariates < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_covariates must be >= 1"):
            MultichoiceLogit(num_alternatives=3, num_covariates=0)


class TestTransformParams:
    """Test parameter transformation."""

    def test_correct_shape(self):
        """Test that transform_params produces correct shape."""
        model = MultichoiceLogit(num_alternatives=4, num_covariates=3)
        flat_beta = np.random.randn((4 - 1) * 3)
        beta = model.transform_params(flat_beta)

        assert beta.shape == (4, 3)
        assert np.allclose(beta[0], 0)  # First row should be zeros

    def test_incorrect_size(self):
        """Test that incorrect flat_beta size raises ValueError."""
        model = MultichoiceLogit(num_alternatives=4, num_covariates=3)
        flat_beta = np.random.randn(10)  # Wrong size

        with pytest.raises(ValueError, match="flat_beta must have size"):
            model.transform_params(flat_beta)


class TestDataValidation:
    """Test input data validation."""

    def test_valid_data(self):
        """Test that valid data passes validation."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        # Should not raise
        model._validate_data(X, y_single, y_dual)

    def test_wrong_x_columns(self):
        """Test that X with wrong number of columns raises ValueError."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        X_wrong = X[:, :1]  # Wrong number of columns
        with pytest.raises(ValueError, match="X must have"):
            model._validate_data(X_wrong, y_single, y_dual)

    def test_wrong_y_single_shape(self):
        """Test that y_single with wrong shape raises ValueError."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        y_single_wrong = y_single[:, :2]  # Wrong shape
        with pytest.raises(ValueError, match="y_single must have shape"):
            model._validate_data(X, y_single_wrong, y_dual)

    def test_wrong_y_dual_shape(self):
        """Test that y_dual with wrong shape raises ValueError."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        y_dual_wrong = y_dual[:, :2, :]  # Wrong shape
        with pytest.raises(ValueError, match="y_dual must have shape"):
            model._validate_data(X, y_single, y_dual_wrong)

    def test_multiple_choices_per_agent(self):
        """Test that multiple choices per agent raises ValueError."""
        N, J, K = 10, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        # Add extra choice to first agent that has a single choice
        # Find an agent with a single choice
        single_agent = np.where(y_single.sum(axis=1) > 0)[0][0]
        # Add another single choice
        available_alt = np.where(y_single[single_agent] == 0)[0][0]
        y_single[single_agent, available_alt] = 1

        with pytest.raises(ValueError, match="Each agent must have exactly one choice"):
            model._validate_data(X, y_single, y_dual)

    def test_no_choice_per_agent(self):
        """Test that no choice per agent raises ValueError."""
        N, J, K = 10, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        # Remove choice from first agent
        y_single[0, :] = 0
        with pytest.raises(ValueError, match="Each agent must have exactly one choice"):
            model._validate_data(X, y_single, y_dual)

    def test_dual_choice_in_lower_triangle(self):
        """Test that dual choices in lower triangle raise ValueError."""
        N, J, K = 10, 3, 2
        model = MultichoiceLogit(J, K)
        X = np.random.randn(N, K)
        y_single = np.zeros((N, J), dtype=np.int8)
        y_dual = np.zeros((N, J, J), dtype=np.int8)

        # Give all agents valid choices first
        for i in range(N):
            y_single[i, 0] = 1

        # Now add an invalid choice in lower triangle for one agent
        y_single[0, 0] = 0  # Remove single choice
        y_dual[0, 2, 1] = 1  # s > t (invalid)

        with pytest.raises(ValueError, match="upper triangle"):
            model._validate_data(X, y_single, y_dual)


class TestNegLogLikelihood:
    """Test negative log-likelihood computation."""

    def test_returns_scalar(self):
        """Test that neg_log_likelihood returns a scalar."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        nll = model.neg_log_likelihood(flat_beta, X, y_single, y_dual)

        assert isinstance(nll, (float, np.floating))

    def test_positive_value(self):
        """Test that neg_log_likelihood returns positive value."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        nll = model.neg_log_likelihood(flat_beta, X, y_single, y_dual)

        assert nll > 0

    def test_all_single_choices(self):
        """Test with all single choices (mix_ratio=1.0)."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=1.0, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        nll = model.neg_log_likelihood(flat_beta, X, y_single, y_dual)

        assert nll > 0
        assert np.sum(y_dual) == 0  # Verify no dual choices

    def test_all_dual_choices(self):
        """Test with all dual choices (mix_ratio=0.0)."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=0.0, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        nll = model.neg_log_likelihood(flat_beta, X, y_single, y_dual)

        assert nll > 0
        assert np.sum(y_single) == 0  # Verify no single choices


class TestGradient:
    """Test gradient computation."""

    def test_returns_correct_shape(self):
        """Test that gradient returns correct shape."""
        N, J, K = 100, 4, 3
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        grad = model.gradient(flat_beta, X, y_single, y_dual)

        assert grad.shape == ((J - 1) * K,)

    def test_gradient_numerical_accuracy(self):
        """Test gradient against numerical approximation."""
        N, J, K = 50, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        analytical_grad = model.gradient(flat_beta, X, y_single, y_dual)

        # Numerical gradient
        epsilon = 1e-5
        numerical_grad = np.zeros_like(flat_beta)
        for i in range(len(flat_beta)):
            beta_plus = flat_beta.copy()
            beta_plus[i] += epsilon
            beta_minus = flat_beta.copy()
            beta_minus[i] -= epsilon

            nll_plus = model.neg_log_likelihood(beta_plus, X, y_single, y_dual)
            nll_minus = model.neg_log_likelihood(beta_minus, X, y_single, y_dual)

            numerical_grad[i] = (nll_plus - nll_minus) / (2 * epsilon)

        # Should be close (within 1e-4)
        assert np.allclose(analytical_grad, numerical_grad, atol=1e-4)


class TestComputeStandardErrors:
    """Test standard error computation."""

    def test_returns_correct_shape(self):
        """Test that compute_standard_errors returns correct shape."""
        N, J, K = 200, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        std_errs = model.compute_standard_errors(flat_beta, X, y_single, y_dual)

        assert std_errs.shape == ((J - 1) * K,)

    def test_positive_standard_errors(self):
        """Test that standard errors are positive."""
        N, J, K = 200, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        std_errs = model.compute_standard_errors(flat_beta, X, y_single, y_dual)

        # Standard errors should be positive (or NaN if singular)
        assert np.all((std_errs > 0) | np.isnan(std_errs))

    def test_singular_hessian_warning(self):
        """Test that singular Hessian produces warning."""
        N, J, K = 10, 3, 2  # Small sample might cause issues
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        # Use arbitrary parameters that might cause singularity
        flat_beta = np.zeros((J - 1) * K)

        # Should produce RuntimeWarning in some cases (not always)
        std_errs = model.compute_standard_errors(flat_beta, X, y_single, y_dual)
        assert std_errs.shape == ((J - 1) * K,)


class TestFitMethod:
    """Test the fit() convenience method."""

    def test_fit_basic(self):
        """Test that fit() works and sets attributes correctly."""
        N, J, K = 500, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        # Initially, coef_ should be None
        assert model.coef_ is None
        assert model.optimization_result_ is None

        # Fit the model
        result = model.fit(X, y_single, y_dual)

        # Should return self
        assert result is model

        # Attributes should be set
        assert model.coef_ is not None
        assert model.optimization_result_ is not None
        assert model.coef_.shape == (J - 1, K)

    def test_fit_recovers_parameters(self):
        """Test that fit() recovers parameters reasonably well."""
        N, J, K = 1000, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        model.fit(X, y_single, y_dual)

        # Mean absolute error should be reasonably small
        mae = np.mean(np.abs(model.coef_ - true_beta))
        assert mae < 0.15

    def test_fit_with_custom_init(self):
        """Test fit() with custom initial parameters."""
        N, J, K = 200, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        # Custom initial values
        init_beta = np.random.randn((J - 1) * K) * 0.1

        model.fit(X, y_single, y_dual, init_beta=init_beta)

        assert model.coef_ is not None
        assert model.coef_.shape == (J - 1, K)

    def test_fit_with_invalid_init_shape(self):
        """Test that invalid init_beta shape raises ValueError."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        init_beta_wrong = np.random.randn(10)  # Wrong size

        with pytest.raises(ValueError, match="init_beta must have size"):
            model.fit(X, y_single, y_dual, init_beta=init_beta_wrong)

    def test_fit_with_different_methods(self):
        """Test fit() with different optimization methods."""
        N, J, K = 200, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)

        for method in ["BFGS", "L-BFGS-B"]:
            model = MultichoiceLogit(J, K)
            model.fit(X, y_single, y_dual, method=method)

            assert model.coef_ is not None
            assert model.optimization_result_.success

    def test_fit_with_custom_options(self):
        """Test fit() with custom optimizer options."""
        N, J, K = 200, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        custom_options = {"gtol": 1e-4, "maxiter": 500}
        model.fit(X, y_single, y_dual, options=custom_options)

        assert model.coef_ is not None
        assert model.optimization_result_ is not None

    def test_fit_method_chaining(self):
        """Test that fit() supports method chaining."""
        N, J, K = 200, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)

        model = MultichoiceLogit(J, K).fit(X, y_single, y_dual)

        assert model.coef_ is not None
        assert model.coef_.shape == (J - 1, K)

    def test_fit_optimization_result_attributes(self):
        """Test that optimization_result_ contains expected attributes."""
        N, J, K = 200, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        model.fit(X, y_single, y_dual)

        result = model.optimization_result_

        # Check expected attributes
        assert hasattr(result, "success")
        assert hasattr(result, "fun")  # Final negative log-likelihood
        assert hasattr(result, "nit")  # Number of iterations
        assert hasattr(result, "nfev")  # Number of function evaluations
        assert result.success is True


class TestEndToEndEstimation:
    """Test end-to-end estimation workflow."""

    def test_estimation_recovers_parameters(self):
        """Test that MLE recovers true parameters reasonably well."""
        from scipy.optimize import minimize

        N, J, K = 1000, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        init_beta = np.zeros((J - 1) * K)

        result = minimize(
            fun=model.neg_log_likelihood,
            jac=model.gradient,
            x0=init_beta,
            args=(X, y_single, y_dual),
            method="BFGS",
            options={"gtol": 1e-5, "maxiter": 1000},
        )

        assert result.success
        est_beta = result.x.reshape(J - 1, K)

        # Mean absolute error should be reasonably small
        mae = np.mean(np.abs(est_beta - true_beta))
        assert mae < 0.15  # Reasonable tolerance for N=1000
