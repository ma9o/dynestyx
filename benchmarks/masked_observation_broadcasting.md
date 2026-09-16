# Masked observation broadcasting PoC

Experiment on `feat/public-model-operations`, initially based on PR #359 head
`dd7a56e91ef8cd45cdf806d7e044c835d70e4a75`, then merged with upstream `main`
at `1c5d89f`. The measured observation source is unchanged by that merge;
its SHA-256 is recorded in each result file. This PoC adds no production
dependency requirements.

The correlated Gaussian path benefits from Cholesky factorization and triangular
solves in every batching layout tested. Efficient broadcasting does not require
JAX 0.11 or our own direct call to `solve_triangular`: NumPyro's public
`MultivariateNormal.log_prob` already groups residuals sharing a factor into
multiple right-hand sides. An outer `vmap` also preserves this sharing, with
similar performance. These measurements support broad observation semantics
without establishing native batching as a requirement for particle inference.

The [implementation](../dynestyx/observation_missingness.py) retains Dynestyx's
mask semantics and delegates the Gaussian calculation to NumPyro. With component
mask `m`, it constructs `C = diag(m) R diag(m) + diag(1-m)` and residual
`r = where(m, y - mean, 0)`, then scores `MultivariateNormal(0, C).log_prob(r)` and removes
the missing coordinates' normalization constants. This is the exact observed
Gaussian marginal. It calls no private NumPyro APIs and contains no version branches.

For scalar and independent component distributions, the helper uses their masked
densities; no matrix solve is needed. Other vector families retain the existing
restriction to fully observed or fully missing rows.

## Broadcasting contract

For vector event shape `(D,)`, leading value, distribution, and mask dimensions
broadcast together. The helper returns individual scores without summing the
leading axes. All-pairs scoring requires an explicit singleton axis.

| Values | Distribution batch | Result | Meaning |
| --- | --- | --- | --- |
| `(D,)` | `()` | `()` | Single observation and distribution |
| `(D,)` | `(N,)` | `(N,)` | One observation across particles/parameter draws |
| `(M, D)` | `()` | `(M,)` | Many values under one fixed distribution |
| `(N, D)` | `(N,)` | `(N,)` | Aligned values and distributions |
| `(M, 1, D)` | `(N,)` | `(M, N)` | All observation/distribution pairs |

Masks can be shared `(D,)` or have compatible leading dimensions. Their trailing
event dimension must match. Scalar distributions use the same leading-axis rule
without a trailing event axis; a length-one array therefore represents a batch
of one, while a scalar represents one unbatched value.

## Measurements

Apple M1 Max CPU, float32, NumPyro 0.21.0, Python 3.12.14. Inputs are runtime
arguments. Each method is compiled and checked before timing, warmed up five
times, and measured for 100 synchronous calls in randomized, interleaved order.
Numbers below are median milliseconds, excluding compilation but including
dispatch. The checked-in [JAX 0.11.1 results](results/masked_observation_broadcasting-0.11.1.json)
and [JAX 0.8.2 results](results/masked_observation_broadcasting-0.8.2.json) include
quartiles and compiler temporary-memory estimates. Their source paths have been
normalized to repository-relative paths; measurements are unmodified. Small
timing differences should not be interpreted as firm rankings.

Five implementations in the [benchmark](masked_observation_broadcasting.py):

- **PoC:** Dynestyx's public helper using NumPyro's public Gaussian density.
- **Direct:** the PR's direct triangular solve, with generalized mask indexing.
- **Expanded:** explicitly broadcast the Cholesky factor to every residual before
  calling the triangular solver.
- **Outer vmap:** vectorize a scalar Gaussian scorer while leaving shared inputs
  unmapped (`jnp.vectorize`, which uses `vmap`).
- **Dense:** use a general linear solver on the same expanded Cholesky factor.
  This isolates the cost of discarding triangular structure; it is deliberately
  not presented as the best alternative Gaussian implementation.

JAX 0.11.1, observation dimension 32:

| Workload | PoC | Direct | Expanded | Outer vmap | Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| One y, 4,096 means, shared R/mask | 0.295 | 0.292 | 1.061 | 0.289 | 10.999 |
| 4,096 ys, one distribution, shared mask | 0.301 | 0.296 | 1.060 | 0.287 | 11.011 |
| 512 aligned pairs, varying R/masks | 0.854 | 0.854 | 0.855 | 0.855 | 2.313 |
| 32 × 128 pairs, shared R, masks vary by y | 0.461 | 0.462 | 1.074 | 0.460 | 11.236 |
| 32 × 128 pairs, R varies by distribution, masks by y | 4.568 | 4.295 | 4.255 | 4.294 | 14.478 |

JAX 0.8.2, observation dimension 32:

| Workload | PoC | Direct | Expanded | Outer vmap | Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| One y, 4,096 means, shared R/mask | 0.230 | Error | 0.908 | 0.213 | 10.807 |
| 4,096 ys, one distribution, shared mask | 0.233 | Error | 0.907 | 0.219 | 10.815 |
| 512 aligned pairs, varying R/masks | 0.586 | 0.589 | 0.583 | 0.587 | 2.016 |
| 32 × 128 pairs, shared R, masks vary by y | 0.426 | Error | 0.920 | 0.419 | 11.108 |
| 32 × 128 pairs, R varies by distribution, masks by y | 3.859 | 3.707 | 3.710 | 3.680 | 13.787 |

The direct call fails on JAX 0.8.2 when matrix and residual batch dimensions differ;
the PoC, outer vmap, and explicit expansion all succeed. Dimension 8 was also
measured, with the same success/failure pattern and qualitative performance result.
The PoC was about 6% slower than outer vmap in the largest distinct-covariance
JAX 0.11.1 case, so this is not evidence that the wrapper always wins on speed.

The older-JAX environment uses Equinox 0.13.2 and an explicit experimental
`blackjax==1.3` override. The PR currently requires `blackjax>=1.6`, whose current
releases require newer JAX. This establishes numerical-kernel compatibility with
0.8.2, not compatibility of the full unchanged dependency set. The JAX 0.11.1
environment uses Equinox 0.13.8 and BlackJax 1.6.2. Compare methods within each
environment; these runs do not isolate the effect of upgrading JAX.

## Why sharing matters

In the first workload at dimension 32, optimized HLO shows:

- **PoC and outer vmap:** one 32 × 32 Cholesky factor and a triangular solve
  against 4,096 right-hand sides.
- **Expanded:** the same single Cholesky factorization, followed by 4,096
  copies of the factor for separate triangular systems.
- **Dense:** an additional LU factorization of 4,096 expanded triangular
  matrices, followed by general-solver substitutions.

On JAX 0.11.1, PoC temporary memory is about 1 MiB versus 16.5 MiB for explicit
expansion and 33 MiB for the dense comparator. Expansion does not repeat the
Cholesky here; its penalty comes from the expanded matrices and solve layout.
Changing the mask can change the factor even when R is shared. The implementation
preserves sharing expressed through singleton axes; it does not deduplicate
equal mask values or covariance values at runtime.

These are forward-density CPU microbenchmarks, not end-to-end inference or GPU
benchmarks. Gradient correctness was tested, but gradient runtime was not timed.
Compilation measurements in JSON can include cache hits and are not a compilation
speed comparison.

## Verification and reproduction

The [new tests](../tests/missingness/test_masked_observation_broadcasting.py) compare
against separately selected, smaller observed marginals. They cover seven batch
layouts across Gaussian, independent LogNormal, scalar LogNormal, and full/missing
Student-t rows; NaNs; fully missing rows; invalid shapes/families; and gradients
with respect to means, covariance, and values for shared and varying covariances.

- JAX 0.11.1: 126 tests passed after merging upstream `main`, including existing
  prepared-scorer, hierarchical, discrete, and ODE missingness checks, plus model
  construction, discretization, and drift linearization. Three MCMC smoke tests
  were excluded. This validation used a fresh environment with the merged
  package's dependencies (including cd-dynamax 0.4.3).
- JAX 0.8.2: all 44 public-helper and broadcasting tests passed.
- Repository Ruff lint/format and ty checks passed, as did the separate benchmark
  lint/type checks.

Create isolated environments from the repository root. The older-JAX command
uses an explicit [experimental override](jax082-overrides.txt) for BlackJax;
it does not alter the package's declared dependency requirements.

```sh
# JAX 0.11.1
uv venv --python 3.12 .output/venvs/jax0111
uv pip install --python .output/venvs/jax0111/bin/python -e . --group dev \
  'jax==0.11.1' 'jaxlib==0.11.1' 'numpyro==0.21.0' 'equinox==0.13.8' \
  'blackjax==1.6.2' 'numpy==2.5.2'
UV_PROJECT_ENVIRONMENT=.output/venvs/jax0111 JAX_PLATFORMS=cpu \
  uv run --no-sync python -m benchmarks.masked_observation_broadcasting

UV_PROJECT_ENVIRONMENT=.output/venvs/jax0111 JAX_PLATFORMS=cpu \
  uv run --no-sync pytest -q -k 'not mcmc' \
  tests/missingness/test_masked_observation_log_prob.py \
  tests/missingness/test_masked_observation_broadcasting.py \
  tests/missingness/test_observation_log_prob.py \
  tests/missingness/test_discrete_simulator.py \
  tests/missingness/test_ode_simulator.py tests/missingness/test_hierarchical.py \
  tests/test_discretizers.py tests/test_drift_linearization.py tests/test_models_core.py

# JAX 0.8.2 (isolated experimental environment)
uv venv --python 3.12 .output/venvs/jax082
uv pip install --python .output/venvs/jax082/bin/python -e . --group dev \
  --overrides benchmarks/jax082-overrides.txt \
  'jax==0.8.2' 'jaxlib==0.8.2' 'numpyro==0.21.0' 'equinox==0.13.2' \
  'numpy==2.5.3'
UV_PROJECT_ENVIRONMENT=.output/venvs/jax082 JAX_PLATFORMS=cpu \
  uv run --no-sync python -m benchmarks.masked_observation_broadcasting

UV_PROJECT_ENVIRONMENT=.output/venvs/jax082 JAX_PLATFORMS=cpu \
  uv run --no-sync pytest -q tests/missingness/test_masked_observation_log_prob.py \
  tests/missingness/test_masked_observation_broadcasting.py
```

Use module execution (`python -m ...`) from the repository root. The benchmark
asserts the imported source path. New JSON results and optimized HLO are saved under
`.output/masked-observation-broadcasting/{0.11.1,0.8.2}/` (gitignored).
