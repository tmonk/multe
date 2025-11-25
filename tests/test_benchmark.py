import importlib.util
from pathlib import Path


def load_benchmark_module():
    """Load examples/benchmark.py as a module for testing."""
    root = Path(__file__).resolve().parents[1]
    benchmark_path = root / "examples" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_module", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_estimation_runs_end_to_end():
    """
    Smoke test for the benchmark helper to ensure a full run (including
    standard error computation) succeeds and returns expected fields.
    """
    benchmark = load_benchmark_module()

    N, J, K = 40, 3, 2
    result = benchmark.benchmark_estimation(N=N, J=J, K=K, method="BFGS", seed=123)

    expected_keys = {
        "N",
        "J",
        "K",
        "method",
        "sim_time",
        "likelihood_time",
        "gradient_time",
        "opt_time",
        "se_time",
        "total_time",
        "success",
        "nit",
        "nfev",
        "final_nll",
        "mae",
        "rmse",
        "max_error",
    }

    assert expected_keys.issubset(result.keys())
    assert result["N"] == N and result["J"] == J and result["K"] == K
    assert result["success"]
    assert result["se_time"] >= 0
