"""
Example: Load Data from CSV and Estimate Multichoice Logit Model

This example demonstrates:
1. Saving simulated data to CSV files
2. Loading data from CSV files
3. Estimating the model with loaded data
4. Handling real-world data formats
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from multe import MultichoiceLogit, simulate_data


def save_data_to_csv(X, y_single, y_dual, filepath_prefix="data"):
    """
    Save data to CSV files.

    Args:
        X: Covariates (N, K)
        y_single: Single choice indicators (N, J)
        y_dual: Dual choice indicators (N, J, J)
        filepath_prefix: Prefix for output files
    """
    N, K = X.shape
    J = y_single.shape[1]

    # Save covariates
    covariate_df = pd.DataFrame(X, columns=[f"x_{k}" for k in range(K)])
    covariate_df.to_csv(f"{filepath_prefix}_covariates.csv", index=False)

    # Save choices in long format
    choices_list = []
    for i in range(N):
        # Check for single choice
        single_idx = np.where(y_single[i] == 1)[0]
        if len(single_idx) > 0:
            choices_list.append(
                {
                    "agent_id": i,
                    "choice_type": "single",
                    "alternative_1": single_idx[0],
                    "alternative_2": None,
                }
            )

        # Check for dual choice
        dual_idx = np.where(y_dual[i] > 0)
        if len(dual_idx[0]) > 0:
            s, t = dual_idx[0][0], dual_idx[1][0]
            choices_list.append(
                {
                    "agent_id": i,
                    "choice_type": "dual",
                    "alternative_1": s,
                    "alternative_2": t,
                }
            )

    choices_df = pd.DataFrame(choices_list)
    choices_df.to_csv(f"{filepath_prefix}_choices.csv", index=False)

    print(f"Data saved to {filepath_prefix}_covariates.csv and {filepath_prefix}_choices.csv")


def load_data_from_csv(filepath_prefix="data", J=None):
    """
    Load data from CSV files.

    Args:
        filepath_prefix: Prefix for input files
        J: Number of alternatives (required)

    Returns:
        X, y_single, y_dual: Same format as simulate_data()
    """
    if J is None:
        raise ValueError("J (number of alternatives) must be specified")

    # Load covariates
    covariate_df = pd.read_csv(f"{filepath_prefix}_covariates.csv")
    X = covariate_df.values
    N, K = X.shape

    # Load choices
    choices_df = pd.read_csv(f"{filepath_prefix}_choices.csv")

    # Initialize choice matrices
    y_single = np.zeros((N, J), dtype=np.int8)
    y_dual = np.zeros((N, J, J), dtype=np.int8)

    # Fill in choices
    for _, row in choices_df.iterrows():
        i = int(row["agent_id"])
        if row["choice_type"] == "single":
            j = int(row["alternative_1"])
            y_single[i, j] = 1
        elif row["choice_type"] == "dual":
            s = int(row["alternative_1"])
            t = int(row["alternative_2"])
            # Ensure s < t for upper triangle
            if s > t:
                s, t = t, s
            y_dual[i, s, t] = 1

    print(f"Data loaded: N={N}, J={J}, K={K}")
    print(f"Single choices: {np.sum(y_single)}, Dual choices: {np.sum(y_dual)}")

    return X, y_single, y_dual


def main():
    """Main example workflow."""
    print("=" * 80)
    print("Example: CSV Data Loading for Multichoice Logit Estimation")
    print("=" * 80)

    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic data...")
    N, J, K = 500, 4, 3
    X, y_single, y_dual, true_beta = simulate_data(N, J, K, mix_ratio=0.6, seed=42)
    print(f"Generated N={N} observations with J={J} alternatives and K={K} covariates")

    # Step 2: Save to CSV
    print("\n2. Saving data to CSV files...")
    save_data_to_csv(X, y_single, y_dual, filepath_prefix="example_data")

    # Step 3: Load from CSV
    print("\n3. Loading data from CSV files...")
    X_loaded, y_single_loaded, y_dual_loaded = load_data_from_csv(
        filepath_prefix="example_data", J=J
    )

    # Verify data matches
    assert np.allclose(X, X_loaded)
    assert np.all(y_single == y_single_loaded)
    assert np.all(y_dual == y_dual_loaded)
    print("✓ Data loaded successfully and matches original")

    # Step 4: Estimate model
    print("\n4. Estimating model with loaded data...")
    model = MultichoiceLogit(J, K)
    init_beta = np.zeros((J - 1) * K)

    result = minimize(
        fun=model.neg_log_likelihood,
        jac=model.gradient,
        x0=init_beta,
        args=(X_loaded, y_single_loaded, y_dual_loaded),
        method="BFGS",
        options={"disp": False, "gtol": 1e-5},
    )

    print(f"Optimization converged: {result.success}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Final negative log-likelihood: {result.fun:.4f}")

    # Step 5: Compare estimates to true parameters
    print("\n5. Comparing estimates to true parameters...")
    est_beta = result.x.reshape(J - 1, K)
    mae = np.mean(np.abs(est_beta - true_beta))

    print("\nTrue vs Estimated Parameters:")
    print("-" * 60)
    for j in range(J - 1):
        print(f"Alternative {j+1}:")
        for k in range(K):
            print(
                f"  Covariate {k}: True={true_beta[j,k]:7.4f}, "
                f"Est={est_beta[j,k]:7.4f}, "
                f"Error={abs(true_beta[j,k] - est_beta[j,k]):7.4f}"
            )
    print("-" * 60)
    print(f"Mean Absolute Error: {mae:.4f}")

    # Step 6: Compute standard errors
    print("\n6. Computing standard errors...")
    std_errs = model.compute_standard_errors(
        result.x, X_loaded, y_single_loaded, y_dual_loaded
    )
    std_errs_reshaped = std_errs.reshape(J - 1, K)

    print("\nParameter Estimates with Standard Errors:")
    print("-" * 60)
    for j in range(J - 1):
        print(f"Alternative {j+1}:")
        for k in range(K):
            t_stat = est_beta[j, k] / std_errs_reshaped[j, k]
            print(
                f"  Covariate {k}: {est_beta[j,k]:7.4f} "
                f"(SE: {std_errs_reshaped[j,k]:6.4f}, "
                f"t: {t_stat:6.2f})"
            )

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

    # Clean up
    import os

    try:
        os.remove("example_data_covariates.csv")
        os.remove("example_data_choices.csv")
        print("\nTemporary CSV files cleaned up.")
    except:
        pass


if __name__ == "__main__":
    main()
