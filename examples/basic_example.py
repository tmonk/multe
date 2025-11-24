"""
Basic Example: Simulate and Estimate a Multichoice Logit Model

This example demonstrates:
1. Simulating data from a multichoice logit model
2. Estimating model parameters using MLE
3. Computing standard errors and statistical tests
4. Comparing estimates to true parameters
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats

from multe import MultichoiceLogit, simulate_data


def main():
    # Settings
    N = 2000  # Number of observations
    J = 4  # Number of alternatives
    K = 3  # Number of covariates

    print(f"Simulating Data (N={N}, J={J}, K={K})...")
    X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=0.5, seed=42)

    print("Starting Estimation...")
    model = MultichoiceLogit(J, K)

    # Initial guess (zeros)
    init_beta = np.zeros((J - 1) * K)

    res = minimize(
        fun=model.neg_log_likelihood,
        jac=model.gradient,
        x0=init_beta,
        args=(X, y_single, y_dual),
        method="L-BFGS-B",
        options={"disp": True, "gtol": 1e-5},
    )

    print("\nOptimization Success:", res.success)

    # Compute Standard Errors
    print("Computing Standard Errors...")
    std_errs = model.compute_standard_errors(res.x, X, y_single, y_dual)

    est_beta = res.x.reshape(J - 1, K)
    std_errs_reshaped = std_errs.reshape(J - 1, K)

    print("\nComparison (Row 0 is fixed to 0, these are rows 1 to J-1):")
    # Header
    print("-" * 88)
    print(
        f"{'True':<10} | {'Est':<10} | {'SE':<10} | {'t-stat':<10} | {'p-val':<10} | {'95% CI':<15}"
    )
    print("-" * 88)

    for i in range(J - 1):
        print(f"Alternative {i + 1}:")
        for k in range(K):
            t_val = true_beta[i, k]
            e_val = est_beta[i, k]
            se = std_errs_reshaped[i, k]

            # t-statistic and p-value
            t_stat = e_val / se if se > 0 else 0
            p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))  # Two-tailed

            # 95% Confidence Interval
            ci_lower = e_val - 1.96 * se
            ci_upper = e_val + 1.96 * se
            ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]"

            print(
                f" {t_val:<9.4f} | {e_val:<9.4f}  | {se:<9.4f}  | {t_stat:<9.2f}  | {p_val:<9.4f}  | {ci_str:<15}"
            )
        print("-" * 88)

    mae = np.mean(np.abs(est_beta - true_beta))
    print(f"Mean Absolute Error: {mae:.4f}")


if __name__ == "__main__":
    main()
