"""
Tests for edge cases and error handling to ensure 100% coverage.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from scipy.optimize import OptimizeResult
from multe import MultichoiceLogit, simulate_data

class TestEdgeCases:
    """Test tricky edge cases and error conditions."""

    def test_optimization_failure_raises_runtime_error(self):
        """Test that a RuntimeError is raised when optimization fails."""
        N, J, K = 100, 3, 2
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        model = MultichoiceLogit(J, K)

        # Mock minimize to return a failed result
        mock_result = OptimizeResult(success=False, message="Optimization failed intentionally")
        
        with patch("multe.model.minimize", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Optimization failed to converge"):
                model.fit(X, y_single, y_dual)

    def test_negative_variance_warning(self):
        """
        Test that a warning is raised when the Hessian inverse has negative diagonal elements.
        This usually happens with numerical instability.
        """
        N, J, K = 50, 3, 2
        # Create perfectly collinear data to force singular/unstable Hessian
        X = np.zeros((N, K)) 
        # If X is zero, gradient is zero, Hessian is zero. 
        # Pinv of zero is zero. Diag is zero.
        # We need a case where pinv gives negative values.
        
        # Easier approach: Mock the gradient or just use specific parameters 
        # that are known to cause issues, OR modify the data manually.
        
        model = MultichoiceLogit(J, K)
        
        # Create data
        X, y_single, y_dual, _ = simulate_data(N, J, K, seed=42)
        
        # We can't easily force pinv to return negatives with standard data.
        # Let's mock the internal _gradient_fast to return values that lead to 
        # a Hessian with negative inverse diagonal elements.
        #
        # Alternatively, and much easier:
        # We can test `compute_standard_errors` but assume the optimization is done.
        # We just need `compute_standard_errors` to encounter a bad Hessian.
        
        # Let's force the Hessian calculation to produce something that inverts to negatives.
        # Or, we can just patch np.linalg.pinv to return a matrix with -1 on diagonal.
        
        flat_beta = np.zeros((J-1)*K)
        
        with patch("numpy.linalg.pinv", return_value=np.diag([-1.0] * len(flat_beta))):
            with pytest.warns(RuntimeWarning, match="Hessian inverse has negative diagonal elements"):
                std_errs = model.compute_standard_errors(flat_beta, X, y_single, y_dual)
                
        # Verify we got NaNs where variance was negative
        assert np.all(np.isnan(std_errs))
