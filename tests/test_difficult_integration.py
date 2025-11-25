"""
Integration tests with difficult data scenarios.
Tests collinearity, separation, tiny samples, and edge case distributions.
"""

import numpy as np
import pytest

from multe import MultichoiceLogit, simulate_data


class TestDifficultIntegration:
    """Integration tests for difficult data scenarios."""

    def test_perfect_collinearity(self):
        """
        Test estimation with perfectly collinear features.
        The Hessian will be singular.
        """
        N, J, K = 500, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)

        # Make column 1 exactly equal to column 0
        X[:, 1] = X[:, 0]

        model = MultichoiceLogit(J, K)
        # This should run without error due to pinv usage, though coefficients are unidentified
        model.fit(X, y_single, y_dual)

        # Should run without error due to pinv usage
        # Note: We don't strictly expect a warning here because pinv finds a minimum norm solution
        # which might have valid positive diagonal elements even if unidentified.
        std_errs = model.compute_standard_errors(
            X, y_single, y_dual, flat_beta=model.coef_.flatten()
        )
        assert std_errs is not None
        assert len(std_errs) == (J - 1) * K

    def test_near_collinearity(self):
        """
        Test estimation with highly correlated features.
        Optimizer should struggle but eventually return.
        """
        N, J, K = 1000, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)

        # Make column 1 very close to column 0
        X[:, 1] = X[:, 0] + np.random.normal(0, 1e-6, N)

        model = MultichoiceLogit(J, K)
        model.fit(X, y_single, y_dual)

        # Should converge successfully
        assert model.optimization_result_.success

    def test_quasi_separation(self):
        """
        Test data where a covariate is a very strong predictor (quasi-separation).
        This typically drives coefficients to be very large.
        """
        N, J, K = 200, 3, 1
        X = np.random.normal(0, 1, (N, K))

        # Construct choices based on X to create strong separation
        y_single = np.zeros((N, J), dtype=np.int8)
        y_dual = np.zeros((N, J, J), dtype=np.int8)

        for i in range(N):
            if X[i, 0] > 0.5:
                y_single[i, 0] = 1
            elif X[i, 0] < -0.5:
                y_single[i, 1] = 1
            else:
                y_single[i, 2] = 1

        model = MultichoiceLogit(J, K)
        model.fit(X, y_single, y_dual)

        assert model.optimization_result_.success

        # Check if coefficients are finite (even if large)
        assert np.all(np.isfinite(model.coef_))

    def test_tiny_sample(self):
        """
        Test with very small sample size (N < Parameters).
        Identification is impossible, but code should run.
        """
        N = 5
        J = 4
        K = 3
        # Total params = (J-1)*K = 3*3 = 9. N=5 < 9.

        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)

        model = MultichoiceLogit(J, K)
        model.fit(X, y_single, y_dual)

        assert model.coef_ is not None
        # Likely warns on SE calculation
        with pytest.warns(RuntimeWarning, match="Hessian inverse"):
            model.compute_standard_errors(
                X, y_single, y_dual, flat_beta=model.coef_.flatten()
            )

    @pytest.mark.slow
    def test_large_scale_synthetic(self):
        """
        Test with a large N=500,000 and J=20.
        Marked as slow. Needs --run-slow to execute.
        """
        N = 500000
        J = 20
        K = 5
        # This will allocate ~200MB for y_dual (int8)
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)

        model = MultichoiceLogit(J, K)
        model.fit(X, y_single, y_dual)

        assert model.coef_ is not None
        assert model.optimization_result_.success

    @pytest.mark.slow
    def test_massive_scale_synthetic(self):
        """
        Test with a massive N=1,000,000 and J=20.
        Marked as slow. Needs --run-slow to execute.
        """
        N = 1000000
        J = 20
        K = 10
        # This will allocate ~400MB for y_dual
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)

        model = MultichoiceLogit(J, K)
        model.fit(X, y_single, y_dual)

        assert model.coef_ is not None
        assert model.optimization_result_.success

    def test_all_dual_choices(self):
        """
        Test where every observation is a dual choice.
        """
        N, J, K = 100, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, mix_ratio=0.0, seed=42)

        assert np.sum(y_single) == 0
        assert np.sum(y_dual) == N

        model = MultichoiceLogit(J, K)
        model.fit(X, y_single, y_dual)
        assert model.optimization_result_.success

    def test_no_dual_choices(self):
        """
        Test where every observation is a single choice (standard MNL).
        """
        N, J, K = 100, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, mix_ratio=1.0, seed=42)

        assert np.sum(y_dual) == 0
        assert np.sum(y_single) == N

        model = MultichoiceLogit(J, K)
        model.fit(X, y_single, y_dual)
        assert model.optimization_result_.success
