"""Tests for data simulation."""

import numpy as np
import pytest

from multe import simulate_data


class TestSimulateDataValidation:
    """Test input validation for simulate_data."""

    def test_valid_parameters(self):
        """Test that valid parameters work."""
        X, y_single, y_dual, true_beta = simulate_data(N=100, J=3, K=2, seed=42)
        assert X.shape == (100, 2)
        assert y_single.shape == (100, 3)
        assert y_dual.shape == (100, 3, 3)
        assert true_beta.shape == (2, 2)

    def test_invalid_N(self):
        """Test that N < 1 raises ValueError."""
        with pytest.raises(ValueError, match="N must be >= 1"):
            simulate_data(N=0, J=3, K=2)

    def test_invalid_J(self):
        """Test that J < 2 raises ValueError."""
        with pytest.raises(ValueError, match="J must be >= 2"):
            simulate_data(N=100, J=1, K=2)

    def test_invalid_K(self):
        """Test that K < 1 raises ValueError."""
        with pytest.raises(ValueError, match="K must be >= 1"):
            simulate_data(N=100, J=3, K=0)

    def test_invalid_mix_ratio_negative(self):
        """Test that negative mix_ratio raises ValueError."""
        with pytest.raises(ValueError, match="mix_ratio must be in"):
            simulate_data(N=100, J=3, K=2, mix_ratio=-0.1)

    def test_invalid_mix_ratio_large(self):
        """Test that mix_ratio > 1 raises ValueError."""
        with pytest.raises(ValueError, match="mix_ratio must be in"):
            simulate_data(N=100, J=3, K=2, mix_ratio=1.5)

    def test_invalid_true_beta_shape(self):
        """Test that wrong true_beta shape raises ValueError."""
        wrong_beta = np.random.randn(3, 3)  # Should be (J-1, K) = (2, 2)
        with pytest.raises(ValueError, match="true_beta must have shape"):
            simulate_data(N=100, J=3, K=2, true_beta=wrong_beta)


class TestSimulateDataOutput:
    """Test output properties of simulate_data."""

    def test_output_shapes(self):
        """Test that output has correct shapes."""
        N, J, K = 200, 4, 3
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)

        assert X.shape == (N, K)
        assert y_single.shape == (N, J)
        assert y_dual.shape == (N, J, J)
        assert true_beta.shape == (J - 1, K)

    def test_each_agent_has_one_choice(self):
        """Test that each agent makes exactly one choice."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)

        single_choices = y_single.sum(axis=1)
        dual_choices = y_dual.sum(axis=(1, 2))
        total_choices = single_choices + dual_choices

        assert np.all(total_choices == 1)

    def test_dual_choices_upper_triangle(self):
        """Test that dual choices are only in upper triangle."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)

        # Check lower triangle (including diagonal) is all zeros
        for i in range(J):
            for j in range(i + 1):
                assert np.all(y_dual[:, i, j] == 0)

    def test_mix_ratio_all_single(self):
        """Test that mix_ratio=1.0 produces all single choices."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=1.0, seed=42)

        assert np.sum(y_single) == N
        assert np.sum(y_dual) == 0

    def test_mix_ratio_all_dual(self):
        """Test that mix_ratio=0.0 produces all dual choices."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=0.0, seed=42)

        assert np.sum(y_single) == 0
        assert np.sum(y_dual) == N

    def test_mix_ratio_approximately_correct(self):
        """Test that mix_ratio produces approximately correct proportions."""
        N, J, K = 1000, 3, 2
        mix_ratio = 0.7
        X, y_single, y_dual, true_beta = simulate_data(
            N, J, K, mix_ratio=mix_ratio, seed=42
        )

        actual_single_ratio = np.sum(y_single) / N

        # With large N, should be close (within 5%)
        assert abs(actual_single_ratio - mix_ratio) < 0.05

    def test_custom_true_beta(self):
        """Test that custom true_beta is used correctly."""
        N, J, K = 100, 3, 2
        custom_beta = np.array([[1.0, 2.0], [3.0, 4.0]])

        X, y_single, y_dual, true_beta = simulate_data(
            N, J, K, true_beta=custom_beta, seed=42
        )

        assert np.allclose(true_beta, custom_beta)

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        N, J, K = 100, 3, 2

        X1, y_single1, y_dual1, beta1 = simulate_data(N, J, K, seed=42)
        X2, y_single2, y_dual2, beta2 = simulate_data(N, J, K, seed=42)

        assert np.allclose(X1, X2)
        assert np.all(y_single1 == y_single2)
        assert np.all(y_dual1 == y_dual2)
        assert np.allclose(beta1, beta2)

    def test_rng_argument_reproducibility(self):
        """Test that supplying rng yields reproducible draws independent of seed."""
        N, J, K = 50, 3, 2
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        X1, y_single1, y_dual1, beta1 = simulate_data(N, J, K, rng=rng1, seed=None)
        X2, y_single2, y_dual2, beta2 = simulate_data(N, J, K, rng=rng2, seed=None)

        assert np.allclose(X1, X2)
        assert np.all(y_single1 == y_single2)
        assert np.all(y_dual1 == y_dual2)
        assert np.allclose(beta1, beta2)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        N, J, K = 100, 3, 2

        X1, y_single1, y_dual1, beta1 = simulate_data(N, J, K, seed=42)
        X2, y_single2, y_dual2, beta2 = simulate_data(N, J, K, seed=123)

        # At least one should be different
        assert not (
            np.allclose(X1, X2)
            and np.all(y_single1 == y_single2)
            and np.all(y_dual1 == y_dual2)
        )

    def test_dtype_control(self):
        """Test that dtype parameter controls output dtype."""
        N, J, K = 20, 3, 2
        X, y_single, y_dual, beta = simulate_data(N, J, K, seed=42, dtype=np.float32)

        assert X.dtype == np.float32
        assert beta.dtype == np.float32

    def test_binary_outputs(self):
        """Test that y_single and y_dual contain only 0 and 1."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)

        assert np.all((y_single == 0) | (y_single == 1))
        assert np.all((y_dual == 0) | (y_dual == 1))

    def test_y_dual_symmetric_property(self):
        """Test that y_dual represents unordered pairs correctly."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)

        # For each observation, if y_dual[i,s,t]=1, then y_dual[i,t,s] should be 0
        # (since we only store in upper triangle)
        for i in range(N):
            for s in range(J):
                for t in range(s + 1, J):
                    if y_dual[i, s, t] == 1:
                        assert y_dual[i, t, s] == 0


class TestSimulateDataStatisticalProperties:
    """Test statistical properties of simulated data."""

    def test_X_approximately_normal(self):
        """Test that X covariates are approximately N(0,1)."""
        N, J, K = 5000, 3, 2
        X, _, _, _ = simulate_data(N, J, K, seed=42)

        # Mean should be close to 0
        assert np.abs(np.mean(X)) < 0.1

        # Std should be close to 1
        assert np.abs(np.std(X) - 1.0) < 0.1

    def test_choices_depend_on_beta(self):
        """Test that choices depend on true_beta parameters."""
        N, J, K = 500, 3, 2

        # Simulate with different beta values
        beta1 = np.array([[1.0, 1.0], [2.0, 2.0]])
        beta2 = np.array([[-1.0, -1.0], [-2.0, -2.0]])

        _, y_single1, y_dual1, _ = simulate_data(N, J, K, true_beta=beta1, seed=42)
        _, y_single2, y_dual2, _ = simulate_data(N, J, K, true_beta=beta2, seed=43)

        # Choice distributions should be different
        choice_dist1 = y_single1.sum(axis=0)
        choice_dist2 = y_single2.sum(axis=0)

        # At least some difference in choices
        assert not np.allclose(choice_dist1, choice_dist2, rtol=0.1)
