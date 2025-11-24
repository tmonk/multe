# Multe: Multichoice Logit Estimation

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/tmonk/multe/actions/workflows/tests.yml/badge.svg)](https://github.com/tmonk/multe/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/multe)](https://pypi.org/project/multe/)

A Python library for estimating discrete choice models where agents can select either single alternatives or unordered pairs of alternatives.

This implements the model as described in Ophem, H.V., Stam, P. and Praag, B.V., 1999. [Multichoice Logit: Modeling Incomplete Preference Rankings of Classical Concerts](https://www.tandfonline.com/doi/abs/10.1198/07350019919290156). Journal of Business & Economic Statistics, 17(1), pp.117-128.

Built by [Thomas Monk](https://tdmonk.com), London School of Economics.

## Citation

If you use this package, please cite it as:

```
@misc{monk2025multe,
  author = {Thomas Monk},
  title = {Multe: Multichoice Logit Estimation},
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
from multe import MultichoiceLogit, simulate_data

# Generate synthetic data
X, y_single, y_dual, true_beta = simulate_data(N=1000, J=4, K=3, seed=42)

# Fit model
model = MultichoiceLogit(num_alternatives=4, num_covariates=3)
model.fit(X, y_single, y_dual)

# Access fitted coefficients
print(model.coef_)  # Shape: (J-1, K) = (3, 3)
```

See `examples/simple_fit_example.py` for a complete example, or `examples/basic_example.py` for advanced usage.

## Model

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

## Data Format

- **X**: Covariates (N, K)
- **y_single**: Binary matrix (N, J) where `y_single[i,j]=1` if agent i chose alternative j
- **y_dual**: Binary tensor (N, J, J) where `y_dual[i,s,t]=1` if agent i chose pair {s,t} with s<t

Each agent must have exactly one choice (one entry in either y_single or y_dual).

## Performance

Fully vectorized implementation for fast estimation and simulation:

**Estimation:**
- ~10ms per 1000 observations (optimization)
- Example: N=10,000, J=5, K=4 completes in ~0.1s

**Recommended optimizer:** L-BFGS-B

Run benchmarks: `python examples/benchmark.py`

## API Reference

### MultichoiceLogit(num_alternatives, num_covariates)
Model class with methods:
- **`fit(X, y_single, y_dual, method='L-BFGS-B')`** - Fit model using MLE (recommended)
  - Returns `self` with fitted `coef_` attribute
  - Stores optimization details in `optimization_result_`
- `neg_log_likelihood(flat_beta, X, y_single, y_dual)` - Negative log-likelihood
- `gradient(flat_beta, X, y_single, y_dual)` - Analytical gradient
- `compute_standard_errors(flat_beta, X, y_single, y_dual)` - Standard errors

### simulate_data(N, J, K, true_beta=None, mix_ratio=0.5, seed=42)
Generate synthetic data following the RUM framework.

Returns: `X, y_single, y_dual, true_beta`

## Acknowledgements

Many thanks to [Alan Manning](https://www.alan-manning.com/) for his guidance and support with this project.
