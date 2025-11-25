"""Minimal quickstart for Multe using the choices-first workflow."""

from __future__ import annotations

import numpy as np

from multe import MultichoiceLogit, parse_choices, simulate_choices


def main() -> None:
    # Simulate data and get choices list directly
    X, choices, true_beta = simulate_choices(N=10000, J=4, K=2, seed=42)

    # - X: (N, K) covariate matrix.
    # - choices: length‑N list of choices (each element is an int for single or a (s, t) tuple for a dual).
    # - true_beta: (J-1, K) true coefficients (first alternative fixed to zero for identification).

    # Fit with the choices-first entrypoint
    model = MultichoiceLogit(num_alternatives=4, num_covariates=2)
    model.fit(X, choices)

    # Optionally compute standard errors
    y_single, y_dual = parse_choices(choices, J=4)
    std_errs = model.compute_standard_errors(X, y_single, y_dual)

    # Report
    result = model.get_result(standard_errors=std_errs)
    result.summary(verbose=True)
    print("\nTrue parameters (free):\n", np.array2string(true_beta, precision=4))


if __name__ == "__main__":
    main()
