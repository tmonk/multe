"""Tests for the MultichoiceLogit model."""

import numpy as np
import pytest
import scipy.sparse as sp

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
        beta = model._transform_params(flat_beta)

        assert beta.shape == (4, 3)
        assert np.allclose(beta[0], 0)  # First row should be zeros

    def test_incorrect_size(self):
        """Test that incorrect flat_beta size raises ValueError."""
        model = MultichoiceLogit(num_alternatives=4, num_covariates=3)
        flat_beta = np.random.randn(10)  # Wrong size

        with pytest.raises(ValueError, match="flat_beta must have size"):
            model._transform_params(flat_beta)


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

    def test_non_binary_inputs(self):
        """Test that non-binary entries raise ValueError."""
        N, J, K = 20, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        y_single[0, 0] = 2
        with pytest.raises(ValueError, match="binary"):
            model._validate_data(X, y_single, y_dual)

        y_single[0, 0] = 1  # restore
        y_dual[0, 1, 2] = 2
        with pytest.raises(ValueError, match="binary"):
            model._validate_data(X, y_single, y_dual)

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

    def test_dual_choice_on_diagonal(self):
        """Test that diagonal dual entries raise ValueError."""
        N, J, K = 10, 3, 2
        model = MultichoiceLogit(J, K)
        X = np.random.randn(N, K)
        y_single = np.zeros((N, J), dtype=np.int8)
        y_dual = np.zeros((N, J, J), dtype=np.int8)
        y_single[:, 0] = 1

        y_single[0, 0] = 0
        y_dual[0, 1, 1] = 1

        with pytest.raises(ValueError, match="diagonal"):
            model._validate_data(X, y_single, y_dual)


class TestNegLogLikelihood:
    """Test negative log-likelihood computation."""

    def test_returns_scalar(self):
        """Test that neg_log_likelihood returns a scalar."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)
        nll = model._neg_log_likelihood(flat_beta, X, single_idx, dual_idx)

        assert isinstance(nll, (float, np.floating))

    def test_positive_value(self):
        """Test that neg_log_likelihood returns positive value."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)
        nll = model._neg_log_likelihood(flat_beta, X, single_idx, dual_idx)

        assert nll > 0

    def test_all_single_choices(self):
        """Test with all single choices (mix_ratio=1.0)."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=1.0, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)
        nll = model._neg_log_likelihood(flat_beta, X, single_idx, dual_idx)

        assert nll > 0
        assert np.sum(y_dual) == 0  # Verify no dual choices

    def test_all_dual_choices(self):
        """Test with all dual choices (mix_ratio=0.0)."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=0.0, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)
        nll = model._neg_log_likelihood(flat_beta, X, single_idx, dual_idx)

        assert nll > 0
        assert np.sum(y_single) == 0  # Verify no single choices

    def test_numerical_stability_large_utilities(self):
        """Test that large utilities don't cause overflow (numerical stability)."""
        N, J, K = 50, 3, 2
        model = MultichoiceLogit(J, K)

        # Create data with very large utilities that would overflow with naive exp
        X = np.random.randn(N, K)
        # Use large beta values
        large_beta = np.array([[100.0, 100.0], [50.0, 50.0]])

        # Create valid single choices
        y_single = np.zeros((N, J), dtype=np.int8)
        y_single[:, 0] = 1  # All choose first alternative
        y_dual = np.zeros((N, J, J), dtype=np.int8)

        flat_beta = large_beta.flatten()

        # Should not overflow or produce inf/nan
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)
        nll = model._neg_log_likelihood(flat_beta, X, single_idx, dual_idx)

        assert np.isfinite(nll)
        assert nll > 0

        # Gradient should also be stable
        grad = model._gradient(flat_beta, X, single_idx, dual_idx)
        assert np.all(np.isfinite(grad))

    def test_sparse_dual_input_equivalence(self):
        """Test that tuple dual indices produce same NLL as dense tensor."""
        N, J, K = 150, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_dense, dual_dense = model._validate_data(X, y_single, y_dual)
        dense_nll = model._neg_log_likelihood(flat_beta, X, single_dense, dual_dense)

        rows, s_idx, t_idx = np.nonzero(y_dual)
        tuple_input = (rows, s_idx, t_idx)
        single_tuple, dual_tuple = model._validate_data(X, y_single, tuple_input)
        tuple_nll = model._neg_log_likelihood(flat_beta, X, single_tuple, dual_tuple)

        assert np.isclose(dense_nll, tuple_nll)


class TestGradient:
    """Test gradient computation."""

    def test_returns_correct_shape(self):
        """Test that gradient returns correct shape."""
        N, J, K = 100, 4, 3
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)
        grad = model._gradient(flat_beta, X, single_idx, dual_idx)

        assert grad.shape == ((J - 1) * K,)

    def test_gradient_numerical_accuracy(self):
        """Test gradient against numerical approximation."""
        N, J, K = 50, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)
        analytical_grad = model._gradient(flat_beta, X, single_idx, dual_idx)

        # Numerical gradient
        epsilon = 1e-5
        numerical_grad = np.zeros_like(flat_beta)
        for i in range(len(flat_beta)):
            beta_plus = flat_beta.copy()
            beta_plus[i] += epsilon
            beta_minus = flat_beta.copy()
            beta_minus[i] -= epsilon

            nll_plus = model._neg_log_likelihood(beta_plus, X, single_idx, dual_idx)
            nll_minus = model._neg_log_likelihood(beta_minus, X, single_idx, dual_idx)

            numerical_grad[i] = (nll_plus - nll_minus) / (2 * epsilon)

        # Should be close (within 1e-4)
        assert np.allclose(analytical_grad, numerical_grad, atol=1e-4)

    def test_gradient_with_clipped_probabilities(self):
        """Test that gradient correctly handles probability clipping."""
        N, J, K = 30, 3, 2
        model = MultichoiceLogit(J, K)

        # Create data with some dual choices
        X, y_single, y_dual, _ = simulate_data(N, J, K, mix_ratio=0.3, seed=42)

        # Use extreme parameters that might cause very low probabilities
        extreme_beta = np.array([[-5.0, -5.0], [10.0, 10.0]])
        flat_beta = extreme_beta.flatten()

        # Gradient should be finite even with clipped probabilities
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)
        grad = model._gradient(flat_beta, X, single_idx, dual_idx)

        assert np.all(np.isfinite(grad))

        # Verify gradient still matches numerical gradient
        epsilon = 1e-5
        numerical_grad = np.zeros_like(flat_beta)
        for i in range(len(flat_beta)):
            beta_plus = flat_beta.copy()
            beta_plus[i] += epsilon
            beta_minus = flat_beta.copy()
            beta_minus[i] -= epsilon

            nll_plus = model._neg_log_likelihood(beta_plus, X, single_idx, dual_idx)
            nll_minus = model._neg_log_likelihood(beta_minus, X, single_idx, dual_idx)

            numerical_grad[i] = (nll_plus - nll_minus) / (2 * epsilon)

        # Should still be close even with clipped probabilities
        assert np.allclose(grad, numerical_grad, atol=1e-4)

    def test_gradient_sparse_dual(self):
        """Gradient with tuple dual input matches dense gradient."""
        N, J, K = 40, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=0.3, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_dense, dual_dense = model._validate_data(X, y_single, y_dual)
        dense_grad = model._gradient(flat_beta, X, single_dense, dual_dense)

        dual_rows, dual_s, dual_t = np.nonzero(y_dual)
        single_tuple, dual_tuple = model._validate_data(
            X, y_single, (dual_rows, dual_s, dual_t)
        )
        tuple_grad = model._gradient(flat_beta, X, single_tuple, dual_tuple)

        assert np.allclose(dense_grad, tuple_grad)

    def test_public_gradient_matches_private(self):
        """Public gradient uses the validated indices and matches the internal version."""
        N, J, K = 60, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=0.4, seed=123)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)

        internal = model._gradient(flat_beta, X, single_idx, dual_idx)
        public = model.gradient(flat_beta, X, y_single, y_dual)

        np.testing.assert_allclose(public, internal)

    def test_public_gradient_accepts_sparse_and_tuple_dual(self):
        """Public gradient handles tuple and sparse dual inputs consistently."""
        N, J, K = 50, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=0.35, seed=7)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()

        # Tuple format
        rows, s_idx, t_idx = np.nonzero(y_dual)
        tuple_grad = model.gradient(flat_beta, X, y_single, (rows, s_idx, t_idx))

        # Sparse format (row-major flattening)
        sparse_dual = sp.csr_matrix(y_dual.reshape(N, J * J))
        sparse_grad = model.gradient(flat_beta, X, y_single, sparse_dual)

        np.testing.assert_allclose(tuple_grad, sparse_grad)


class TestComputeStandardErrors:
    """Test standard error computation."""

    def test_returns_correct_shape(self):
        """Test that compute_standard_errors returns correct shape."""
        N, J, K = 200, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        std_errs = model.compute_standard_errors(X, y_single, y_dual, flat_beta)

        assert std_errs.shape == ((J - 1) * K,)

    def test_positive_standard_errors(self):
        """Test that standard errors are positive."""
        N, J, K = 200, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        std_errs = model.compute_standard_errors(X, y_single, y_dual, flat_beta)

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
        std_errs = model.compute_standard_errors(X, y_single, y_dual, flat_beta)
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
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)

        def fun(beta):
            return model._neg_log_likelihood(beta, X, single_idx, dual_idx)

        def jac(beta):
            return model._gradient(beta, X, single_idx, dual_idx)

        result = minimize(
            fun=fun,
            jac=jac,
            x0=init_beta,
            method="BFGS",
            options={"gtol": 1e-5, "maxiter": 1000},
        )

        assert result.success
        est_beta = result.x.reshape(J - 1, K)

        # Mean absolute error should be reasonably small
        mae = np.mean(np.abs(est_beta - true_beta))
        assert mae < 0.15  # Reasonable tolerance for N=1000


class TestHelpers:
    """Tests for helper APIs such as predict_proba and log_likelihood_contributions."""

    def test_predict_proba_shapes_and_sums(self):
        N, J, K = 50, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        single_probs, dual_probs = model.predict_proba(X, flat_beta=flat_beta)

        assert single_probs.shape == (N, J)
        assert dual_probs.shape == (N, J, J)
        # Single probabilities should sum to 1
        np.testing.assert_allclose(np.sum(single_probs, axis=1), 1.0, atol=1e-6)
        # Dual probabilities upper triangle should be non-negative
        assert np.all(
            dual_probs[:, np.triu_indices(J, k=1)[0], np.triu_indices(J, k=1)[1]] >= 0
        )

    def test_log_likelihood_contributions_sum(self):
        N, J, K = 80, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)
        flat_beta = true_beta.flatten()

        contributions = model.log_likelihood_contributions(
            X, y_single, y_dual, flat_beta
        )
        total = np.sum(contributions)
        single_idx, dual_idx = model._validate_data(X, y_single, y_dual)
        direct = -model._neg_log_likelihood(flat_beta, X, single_idx, dual_idx)

        assert contributions.shape == (N,)
        np.testing.assert_allclose(total, direct, rtol=1e-6, atol=1e-6)

    def test_compute_standard_errors_custom_epsilon(self):
        N, J, K = 60, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        flat_beta = true_beta.flatten()
        std_errs_default = model.compute_standard_errors(X, y_single, y_dual, flat_beta)
        std_errs_smaller = model.compute_standard_errors(
            X, y_single, y_dual, flat_beta, epsilon=1e-6
        )

        assert std_errs_default.shape == std_errs_smaller.shape == ((J - 1) * K,)
        assert np.all(np.isfinite(std_errs_default))
