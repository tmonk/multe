# Multe: Multichoice Logit Estimation

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/tmonk/multe/actions/workflows/tests.yml/badge.svg)](https://github.com/tmonk/multe/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/multe)](https://pypi.org/project/multe/)

A Python library for estimating discrete choice models where agents can select either single alternatives or unordered pairs of alternatives.

This implements the model as described in Ophem, H.V., Stam, P. and Praag, B.V., 1999. [Multichoice Logit: Modeling Incomplete Preference Rankings of Classical Concerts](https://www.tandfonline.com/doi/abs/10.1080/07350015.1999.10524801). Journal of Business & Economic Statistics, 17(1), pp.117-128.

Built by [Thomas Monk](https://tdmonk.com), London School of Economics.

## Citation

If you use this package, please cite it as:

```
@misc{monk2025multe,
  author = {Thomas Monk},
  title = {multe},
  howpublished = {\url{https://github.com/tmonk/multe}},
  year = {2025}
}
```

## Installation

Install from PyPI:
```bash
pip install multe
```

Or install from source for development:
```bash
git clone https://github.com/tmonk/multe.git
cd multe
pip install -e .
```

## Quick Start

```python
from multe import MultichoiceLogit, simulate_choices

# N: Number of observations
# J: Number of alternatives
# K: Number of covariates

X, choices, true_beta = simulate_choices(N=1000, J=4, K=3, seed=123)

model = MultichoiceLogit(num_alternatives=4, num_covariates=3)
result = model.fit(X, choices).get_result()
print(result.summary())
```

See `examples/quickstart.py` for a ready-to-run script or `examples/matrix_example.py` for the matrix workflow and deeper walkthrough.

## Data Format

- `choices` (recommended): Length-N list of either `int` (single choice) or `(s, t)` tuples (dual choice, unordered).
- `X`: Covariates of shape `(N, K)`.
- Each agent must have exactly one choice entry.
- Need matrices for another workflow? Jump to **Matrix inputs (y_single/y_dual)** below.

## Model

The full model derivation is available [here](https://raw.githubusercontent.com/tmonk/multe/refs/heads/master/documentation/multichoice.pdf).

Agents choose from J alternatives. The utility for agent i and alternative j is:

```
U_ij = V_ij + epsilon_ij
V_ij = X_i * beta_j
epsilon_ij ~ Gumbel(0, 1)
```

Each agent makes either:
- **Single choice**: Select the alternative with maximum utility
- **Dual choice**: Select the top two alternatives as an unordered pair

The mixing probability between single and dual choices is determined by the parameter `mix_ratio` in simulation.

### Probabilities

**Single choice** (multinomial logit):
```
P(choose j) = exp(V_ij) / sum_k exp(V_ik)
```

**Dual choice** (inclusion-exclusion principle):
```
P(choose {s,t}) = P(U_s > max_k≠s U_k ∪ U_t > max_k≠t U_k)
```

### Identification

The model fixes beta_0 = 0 for identification, so it estimates (J-1) × K parameters.

## Performance

Fully vectorized implementation for fast estimation and simulation:

**Estimation:**
- ~10ms per 1000 observations (optimization)
- Example: N=10,000, J=5, K=4 completes in ~0.1s

**Recommended optimizer:** L-BFGS-B

Run benchmarks: `python examples/benchmark.py`

## API Reference

- `MultichoiceLogit(num_alternatives, num_covariates)` – model class.
- `fit(X, choices, **kwargs)` – choices-first MLE.
- `get_result(standard_errors=None)` – `ModelResult` snapshot with `summary()`.
- `predict_proba(X, flat_beta=None)` – single/dual choice probabilities.
- `log_likelihood_contributions(X, y_single, y_dual, flat_beta=None)` – per-observation log-likelihoods (convert via `parse_choices` if starting from choices).
- `parse_choices(choices, J)` – convert choice list to `(y_single, y_dual)`.
- `simulate_choices(N, J, K, true_beta=None, mix_ratio=0.5, seed=42, rng=None, dtype=np.float64)` – generate `(X, choices, true_beta)`; `true_beta` shape is `(J-1, K)` (first alternative fixed to zero).

## Interpreting the output

The inference table shows one row per alternative (`alt`) and covariate (`k`): the estimated coefficient, its standard error, z-score, and p-value.

Example (immigration attitudes):
- `alt1` = “less immigration”, `alt2` = “stay the same”, `alt3` = “more immigration”.
- `k0` = non-EU migrant share, `k1` = unemployment rate.
- A row `alt=1, k=0, coef=-0.26` means higher non-EU share is associated with lower likelihood of choosing “less immigration”. Each row reads the same way for every attitude option and predictor.

## Matrix inputs (y_single/y_dual)

The choices-first flow is easiest to use. If you already have matrices:
- **y_single**: Binary matrix `(N, J)` where `y_single[i, j] = 1` if agent `i` chose alternative `j`.
- **y_dual**: Binary tensor `(N, J, J)` where `y_dual[i, s, t] = 1` if agent `i` chose the unordered pair `{s, t}` with `s < t`. Sparse CSR (shape `N × J²`, row-major flattening) and tuple index inputs `(rows, s, t)` are also supported.
- Convert from choices with `parse_choices(choices, J)`.

Matrix workflow example:
```python
from multe import MultichoiceLogit, parse_choices, simulate_choices, simulate_data

X, choices, _ = simulate_choices(N=1000, J=4, K=3, seed=42)
y_single, y_dual = parse_choices(choices, J=4)
model = MultichoiceLogit(num_alternatives=4, num_covariates=3)
model.fit_matrix(X, y_single=y_single, y_dual=y_dual)
```

Matrix-focused utilities:
- `parse_choices(choices, J)` – Convert a list of choices (ints or `(s, t)` tuples) into `y_single, y_dual`.
- `simulate_data(N, J, K, true_beta=None, mix_ratio=0.5, seed=42, rng=None, dtype=np.float64)` – Generate synthetic data following the RUM framework and return `(X, y_single, y_dual, true_beta)`.
- `fit_matrix(X, y_single, y_dual, **kwargs)` – Fit via MLE when you already have matrices.
- `gradient(flat_beta, X, y_single, y_dual)` – Public analytical gradient; accepts dense, sparse, or tuple dual inputs.
- `compute_standard_errors(X, y_single, y_dual, flat_beta=None, epsilon=None)` – Numerical Hessian standard errors (uses fitted params by default).
- `log_likelihood_contributions(X, y_single, y_dual, flat_beta=None)` – Per-observation log-likelihoods.

## Acknowledgements

Many thanks to [Alan Manning](https://www.alan-manning.com/) for his guidance and support with this project.
