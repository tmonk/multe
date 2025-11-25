"""
Benchmark script for Multichoice Logit estimation

Tests speed and accuracy across different problem sizes and optimization methods.
"""

import time

import numpy as np
from scipy.optimize import minimize

from multe import MultichoiceLogit, simulate_data


def benchmark_estimation(N, J, K, mix_ratio=0.5, method="BFGS", seed=42):
    """
    Benchmark a single estimation run.

    Returns:
        dict: Contains timing, accuracy, and convergence information
    """
    # Simulate data
    t0 = time.time()
    X, y_single, y_dual, true_beta = simulate_data(
        N, J, K, mix_ratio=mix_ratio, seed=seed
    )
    sim_time = time.time() - t0

    # Initialize model
    model = MultichoiceLogit(J, K)
    init_beta = np.zeros((J - 1) * K)

    # Prepare indices once for timing internals
    single_idx, dual_idx = model._validate_data(X, y_single, y_dual)

    # Time likelihood evaluation
    t0 = time.time()
    _ = model._neg_log_likelihood(init_beta, X, single_idx, dual_idx)
    likelihood_time = time.time() - t0

    # Time gradient evaluation
    t0 = time.time()
    _ = model._gradient(init_beta, X, single_idx, dual_idx)
    gradient_time = time.time() - t0

    # Optimize
    t0 = time.time()
    result = minimize(
        fun=lambda b: model._neg_log_likelihood(b, X, single_idx, dual_idx),
        jac=lambda b: model._gradient(b, X, single_idx, dual_idx),
        x0=init_beta,
        method=method,
        options={"disp": False, "gtol": 1e-5, "maxiter": 1000},
    )
    opt_time = time.time() - t0

    # Compute accuracy
    est_beta = result.x.reshape(J - 1, K)
    mae = np.mean(np.abs(est_beta - true_beta))
    rmse = np.sqrt(np.mean((est_beta - true_beta) ** 2))
    max_error = np.max(np.abs(est_beta - true_beta))

    # Compute standard errors
    t0 = time.time()
    # Compute standard errors (runtime tracked, values unused in benchmark output)
    _ = model.compute_standard_errors(X, y_single, y_dual, flat_beta=result.x)
    se_time = time.time() - t0

    return {
        "N": N,
        "J": J,
        "K": K,
        "method": method,
        "sim_time": sim_time,
        "likelihood_time": likelihood_time,
        "gradient_time": gradient_time,
        "opt_time": opt_time,
        "se_time": se_time,
        "total_time": sim_time + opt_time + se_time,
        "success": result.success,
        "nit": result.nit,
        "nfev": result.nfev,
        "final_nll": result.fun,
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
    }


def print_results(results):
    """Pretty print benchmark results."""
    print("\n" + "=" * 100)
    print(
        f"{'N':<7} {'J':<4} {'K':<4} {'Method':<10} {'Opt(s)':<8} {'SE(s)':<8} {'Total(s)':<9} "
        f"{'Iters':<6} {'FEval':<6} {'MAE':<8} {'RMSE':<8} {'MaxErr':<8} {'Success':<7}"
    )
    print("=" * 100)

    for r in results:
        print(
            f"{r['N']:<7} {r['J']:<4} {r['K']:<4} {r['method']:<10} "
            f"{r['opt_time']:<8.3f} {r['se_time']:<8.3f} {r['total_time']:<9.3f} "
            f"{r['nit']:<6} {r['nfev']:<6} "
            f"{r['mae']:<8.4f} {r['rmse']:<8.4f} {r['max_error']:<8.4f} "
            f"{'✓' if r['success'] else '✗':<7}"
        )
    print("=" * 100)


def main():
    print("Multichoice Logit Estimation Benchmark")
    print("=" * 100)

    # Test different problem sizes
    problem_sizes = [
        (500, 3, 2),
        (1000, 3, 2),
        (2000, 4, 3),
        (5000, 4, 3),
        (10000, 5, 4),
    ]

    # Test different optimization methods
    methods = ["BFGS", "L-BFGS-B", "Newton-CG"]

    results = []

    # Benchmark across problem sizes with BFGS
    print("\n1. Problem Size Scaling (BFGS)")
    print("-" * 100)
    for N, J, K in problem_sizes:
        print(f"Running: N={N}, J={J}, K={K}...", end=" ", flush=True)
        r = benchmark_estimation(N, J, K, method="BFGS")
        results.append(r)
        print(f"✓ ({r['opt_time']:.2f}s)")

    # Benchmark different methods on medium problem
    print("\n2. Optimization Method Comparison (N=2000, J=4, K=3)")
    print("-" * 100)
    N, J, K = 2000, 4, 3
    for method in methods:
        print(f"Running: {method}...", end=" ", flush=True)
        try:
            r = benchmark_estimation(N, J, K, method=method)
            results.append(r)
            print(f"✓ ({r['opt_time']:.2f}s)")
        except Exception as e:
            print(f"✗ Failed: {e}")

    # Test different mix ratios
    print("\n3. Mix Ratio Effect (N=2000, J=4, K=3)")
    print("-" * 100)
    for mix_ratio in [0.2, 0.5, 0.8]:
        print(f"Running: mix_ratio={mix_ratio}...", end=" ", flush=True)
        r = benchmark_estimation(2000, 4, 3, mix_ratio=mix_ratio, method="BFGS")
        r["mix_ratio"] = mix_ratio
        results.append(r)
        print(f"✓ ({r['opt_time']:.2f}s)")

    # Print all results
    print_results(results)

    # Summary statistics
    print("\nSummary:")
    print("-" * 100)
    bfgs_results = [
        r for r in results if r["method"] == "BFGS" and "mix_ratio" not in r
    ]
    if bfgs_results:
        avg_time_per_1k = np.mean(
            [r["opt_time"] / (r["N"] / 1000) for r in bfgs_results]
        )
        avg_mae = np.mean([r["mae"] for r in bfgs_results])
        avg_rmse = np.mean([r["rmse"] for r in bfgs_results])

        print(
            f"Average optimization time per 1000 observations: {avg_time_per_1k:.3f}s"
        )
        print(f"Average MAE: {avg_mae:.4f}")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(
            f"Success rate: {sum(r['success'] for r in bfgs_results) / len(bfgs_results) * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
