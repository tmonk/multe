"""
Matrix Example: Simulate and estimate a multichoice logit model using matrix inputs.

Steps:
1) Simulate data (X, y_single, y_dual)
2) Fit via fit_matrix()
3) Compute standard errors and show the built-in summary
4) Inspect fit quality vs. true parameters and basic predictions
"""

from __future__ import annotations

import numpy as np

from multe import MultichoiceLogit, parse_choices, simulate_choices, simulate_data


def main() -> None:
    print("=" * 70)
    print("Matrix Workflow: fit_matrix() demonstration")
    print("=" * 70)

    # 1) Simulate data
    N, J, K = 1000, 4, 3
    print(f"\nSimulating data (N={N}, J={J}, K={K})...")
    X, y_single, y_dual, true_beta = simulate_data(N, J, K, seed=42)

    # Optional: if you start from choices, convert via parse_choices
    _, choices_demo, _ = simulate_choices(5, J, K, seed=0)
    y_single_demo, y_dual_demo = parse_choices(choices_demo, J=J)
    print(
        f"Converted a small choices demo to matrices: y_single_demo shape={y_single_demo.shape}, y_dual_demo shape={y_dual_demo.shape}"
    )

    # 2) Fit using matrix inputs
    print("\nFitting model with fit_matrix()...")
    model = MultichoiceLogit(num_alternatives=J, num_covariates=K)
    model.fit_matrix(
        X,
        y_single=y_single,
        y_dual=y_dual,
        method="L-BFGS-B",
        options={"gtol": 1e-5, "maxiter": 1000},
    )
    print(f"   ✓ Optimization success: {model.optimization_result_.success}")
    print(f"   ✓ Iterations: {model.optimization_result_.nit}")

    # 3) Standard errors + summary
    print("\nComputing standard errors and summary...")
    std_errs = model.compute_standard_errors(X, y_single, y_dual)
    result = model.get_result(standard_errors=std_errs)
    print(result.summary())

    # 4) Fit quality vs. truth
    mae = np.mean(np.abs(model.coef_ - true_beta))
    print(f"\nMean Absolute Error vs. true parameters: {mae:.4f}")

    # 5) Predict probabilities for a few rows
    single_probs, dual_probs = model.predict_proba(X[:5])
    print("\nPredicted single-choice probabilities (first 5 rows):")
    print(single_probs)
    print("\nPredicted dual-choice probabilities (first 5 rows, upper triangle):")
    tri_rows, tri_cols = np.triu_indices(J, k=1)
    print(dual_probs[:, tri_rows, tri_cols])


if __name__ == "__main__":
    main()
