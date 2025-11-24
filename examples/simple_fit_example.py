"""
Simple Example: Using the fit() Method

This example demonstrates the convenient fit() method for quick model estimation.
"""

import numpy as np

from multe import MultichoiceLogit, simulate_data


def main():
    print("=" * 70)
    print("Simple Example: Using fit() Method")
    print("=" * 70)

    # Generate data
    print("\n1. Simulating data...")
    N, J, K = 1000, 4, 3
    X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)
    print(f"   Generated N={N} observations, J={J} alternatives, K={K} covariates")

    # Fit model using the convenient fit() method
    print("\n2. Fitting model using fit() method...")
    model = MultichoiceLogit(num_alternatives=J, num_covariates=K)
    model.fit(X, y_single, y_dual)
    print(f"   ✓ Optimization converged in {model.optimization_result_.nit} iterations")

    # Access fitted coefficients
    print("\n3. Fitted coefficients (model.coef_):")
    print("-" * 70)
    print(f"   Shape: {model.coef_.shape}")
    print("\n   Values:")
    for j in range(J - 1):
        print(f"   Alternative {j + 1}: {model.coef_[j]}")

    # Compare to true parameters
    print("\n4. Comparison to true parameters:")
    print("-" * 70)
    mae = np.mean(np.abs(model.coef_ - true_beta))
    print(f"   Mean Absolute Error: {mae:.4f}")

    # Compute standard errors
    print("\n5. Computing standard errors...")
    flat_coef = model.coef_.flatten()
    std_errs = model.compute_standard_errors(X, y_single, y_dual, flat_coef)
    std_errs = std_errs.reshape(J - 1, K)

    print("\n   Coefficients with Standard Errors:")
    print("-" * 70)
    for j in range(J - 1):
        print(f"   Alternative {j + 1}:")
        for k in range(K):
            coef = model.coef_[j, k]
            se = std_errs[j, k]
            t_stat = coef / se
            sig = (
                "***"
                if abs(t_stat) > 2.576
                else (
                    "**" if abs(t_stat) > 1.96 else ("*" if abs(t_stat) > 1.645 else "")
                )
            )
            print(
                f"      Covariate {k}: {coef:7.4f} (SE: {se:6.4f}, t: {t_stat:6.2f}) {sig}"
            )

    print("\n" + "=" * 70)
    print("Example completed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
