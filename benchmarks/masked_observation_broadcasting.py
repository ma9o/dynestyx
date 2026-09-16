"""CPU PoC: compare masked Gaussian solve strategies on broadcast inputs.

Run from the repository root with `python -m benchmarks.masked_observation_broadcasting`.
Results and optimized HLO are written under
.output/masked-observation-broadcasting/<jax>/.
No model fitting, simulations, or GPU workloads are run.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist

import dynestyx as dsx

CASES = (
    "one_y_many_distributions",
    "many_y_one_distribution",
    "aligned_distinct_covariances",
    "all_pairs_shared_covariance",
    "all_pairs_distinct_covariances",
)


def _terms(mean, covariance, y, mask):
    observation_dist = dist.MultivariateNormal(mean, covariance)
    covariance = observation_dist.covariance_matrix
    residual = jnp.where(mask, y - observation_dist.loc, 0)
    mask_f = mask.astype(mean.dtype)
    masked_covariance = (
        covariance * mask_f[..., :, None] * mask_f[..., None, :]
        + jnp.eye(mean.shape[-1]) * (1 - mask_f)[..., None, :]
    )
    chol = jnp.linalg.cholesky(masked_covariance)
    logdet = 2 * jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
    normalization = logdet + mask_f.sum(-1) * jnp.log(2 * jnp.pi)
    return chol, residual, normalization


def _triangular_native(mean, covariance, y, mask):
    chol, residual, normalization = _terms(mean, covariance, y, mask)
    whitened = jsp.linalg.solve_triangular(chol, residual[..., None], lower=True)[
        ..., 0
    ]
    return -0.5 * ((whitened**2).sum(-1) + normalization)


def _triangular_broadcast(mean, covariance, y, mask):
    chol, residual, normalization = _terms(mean, covariance, y, mask)
    batch_shape = jnp.broadcast_shapes(chol.shape[:-2], residual.shape[:-1])
    chol = jnp.broadcast_to(chol, batch_shape + chol.shape[-2:])
    residual = jnp.broadcast_to(residual, batch_shape + residual.shape[-1:])
    whitened = jsp.linalg.solve_triangular(chol, residual[..., None], lower=True)[
        ..., 0
    ]
    return -0.5 * ((whitened**2).sum(-1) + normalization)


def _dense_solve(mean, covariance, y, mask):
    chol, residual, normalization = _terms(mean, covariance, y, mask)
    batch_shape = jnp.broadcast_shapes(chol.shape[:-2], residual.shape[:-1])
    chol = jnp.broadcast_to(chol, batch_shape + chol.shape[-2:])
    residual = jnp.broadcast_to(residual, batch_shape + residual.shape[-1:])
    # Deliberately use the general solver on the same Cholesky factor to
    # isolate the benefit of exploiting its triangular structure.
    whitened = jnp.linalg.solve(chol, residual[..., None])[..., 0]
    return -0.5 * ((whitened**2).sum(-1) + normalization)


def _public_numpyro(mean, covariance, y, mask):
    return dsx.masked_observation_log_prob(
        dist.MultivariateNormal(mean, covariance), y=y, obs_mask=mask
    )


# jnp.vectorize uses vmap, keeping singleton/broadcast arguments unmapped.
_outer_vmap = jnp.vectorize(_triangular_native, signature="(d),(d,d),(d),(d)->()")
METHODS = {
    "public_numpyro": _public_numpyro,
    "triangular_native": _triangular_native,
    "triangular_broadcast": _triangular_broadcast,
    "outer_vmap": _outer_vmap,
    "dense_solve": _dense_solve,
}


def _inputs(case, dimension, particles):
    rng = np.random.default_rng(219)
    factor = rng.normal(size=(dimension, dimension)).astype(np.float32) / np.sqrt(
        dimension
    )
    covariance = factor @ factor.T + np.eye(dimension, dtype=np.float32)
    shared_mask = np.arange(dimension) % 3 != 1
    if case == "one_y_many_distributions":
        mean = rng.normal(size=(particles, dimension))
        y = rng.normal(size=(dimension,))
        mask = shared_mask
    elif case == "many_y_one_distribution":
        mean = rng.normal(size=(dimension,))
        y = rng.normal(size=(particles, dimension))
        mask = shared_mask
    elif case == "aligned_distinct_covariances":
        size = max(16, particles // 8)
        mean = rng.normal(size=(size, dimension))
        y = rng.normal(size=(size, dimension))
        covariance = covariance * np.linspace(0.7, 1.3, size)[:, None, None]
        mask = (np.arange(size)[:, None] + np.arange(dimension)) % 3 != 1
    else:
        n_distributions = max(8, particles // 32)
        mean = rng.normal(size=(n_distributions, dimension))
        y = rng.normal(size=(32, 1, dimension))
        mask = (np.arange(32)[:, None, None] + np.arange(dimension)) % 3 != 1
        if case == "all_pairs_distinct_covariances":
            covariance = (
                covariance * np.linspace(0.7, 1.3, n_distributions)[:, None, None]
            )
    y = np.where(mask, y, np.nan)
    return tuple(
        jnp.asarray(value, dtype=jnp.float32) for value in (mean, covariance, y)
    ) + (jnp.asarray(mask),)


def _compile_method(function, args, expected, hlo_path):
    start = time.perf_counter()
    compiled = jax.jit(function).lower(*args).compile()
    compile_seconds = time.perf_counter() - start
    result = compiled(*args)
    result.block_until_ready()
    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-5)
    hlo_path.write_text(compiled.as_text())
    for _ in range(5):
        compiled(*args).block_until_ready()
    memory = compiled.memory_analysis()
    assert memory is not None
    return compiled, {
        "compile_seconds": compile_seconds,
        "temporary_bytes": memory.temp_size_in_bytes,
        "result_shape": list(result.shape),
    }


def _time_methods(compiled_methods, args, repeats):
    """Interleave methods to avoid consistently favoring later warmed-up runs."""
    rng = np.random.default_rng(829)
    timings = {name: [] for name in compiled_methods}
    for _ in range(repeats):
        for name in rng.permutation(list(compiled_methods)):
            start = time.perf_counter_ns()
            compiled_methods[name](*args).block_until_ready()
            timings[name].append((time.perf_counter_ns() - start) / 1e3)
    return {
        name: {
            "median_us": statistics.median(samples),
            "p25_us": float(np.percentile(samples, 25)),
            "p75_us": float(np.percentile(samples, 75)),
        }
        for name, samples in timings.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=100)
    options = parser.parse_args()
    output = Path(".output/masked-observation-broadcasting") / jax.__version__
    output.mkdir(parents=True, exist_ok=True)
    source = Path(dsx.__file__).parent / "observation_missingness.py"
    assert source.resolve() == Path("dynestyx/observation_missingness.py").resolve()
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "numpyro", "equinox", "blackjax", "numpy")
            },
            "devices": [str(device) for device in jax.devices()],
            "x64": jax.config.jax_enable_x64,
            "source": str(source.relative_to(Path.cwd())),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "options": vars(options),
        "timing": "randomized interleaving; synchronous calls; excludes compilation",
        "results": [],
    }
    print(json.dumps(report["environment"]), flush=True)
    for dimension in options.dimensions:
        for case in CASES:
            args = _inputs(case, dimension, options.particles)
            expected = jax.jit(_outer_vmap)(*args)
            expected.block_until_ready()
            compiled_methods = {}
            rows = []
            for name, method in METHODS.items():
                row = {"case": case, "dimension": dimension, "method": name}
                try:
                    compiled, metrics = _compile_method(
                        method,
                        args,
                        expected,
                        output / f"{case}-{dimension}-{name}.hlo.txt",
                    )
                    compiled_methods[name] = compiled
                    row.update(metrics)
                except (TypeError, ValueError) as error:
                    # A failed strategy is a benchmark result, not a runtime fallback.
                    row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
            timings = _time_methods(compiled_methods, args, options.repeats)
            for row in rows:
                row.update(timings.get(row["method"], {}))
                report["results"].append(row)
                print(json.dumps(row), flush=True)
                (output / "results.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
