"""Broadcast observation values, distributions, and component masks together."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import dynestyx as dsx

CASES = {
    "single": ((), (), ()),
    "distribution_batch": ((3,), (), ()),
    "value_batch": ((), (4,), (4,)),
    "aligned": ((4,), (4,), (4,)),
    "all_pairs": ((3,), (4, 1), (4, 1)),
    "nested": ((2, 1), (4, 1, 3), (4, 1, 3)),
    "distribution_masks": ((3,), (), (3,)),
}
COVARIANCE = jnp.array([[0.5, 0.1, 0.08], [0.1, 0.4, 0.05], [0.08, 0.05, 0.3]])
MASKS = jnp.array(
    [
        [True, False, True],
        [True, True, True],
        [False, False, False],
        [False, True, True],
    ]
)


def _inputs(case, shared_covariance=True):
    distribution_batch, value_batch, mask_batch = CASES[case]
    means = jnp.linspace(-0.3, 0.6, math.prod(distribution_batch) * 3).reshape(
        distribution_batch + (3,)
    )
    values = jnp.linspace(0.2, 0.9, math.prod(value_batch) * 3).reshape(
        value_batch + (3,)
    )
    mask = MASKS[jnp.arange(math.prod(mask_batch)) % len(MASKS)].reshape(
        mask_batch + (3,)
    )
    covariance = COVARIANCE
    if not shared_covariance:
        scales = jnp.linspace(0.7, 1.3, math.prod(distribution_batch))
        covariance = covariance * scales.reshape(distribution_batch + (1, 1))
    return means, covariance, values, mask


def _distribution(family, mean, covariance):
    if family == "gaussian":
        return dist.MultivariateNormal(mean, covariance)
    if family == "independent":
        return dist.LogNormal(mean, 0.5).to_event(1)
    if family == "student":
        return dist.MultivariateStudentT(5.0, mean, jnp.linalg.cholesky(covariance))
    if family == "scalar":
        return dist.LogNormal(mean, 0.5)
    raise AssertionError(family)


def _reference(family, means, covariance, values, mask):
    """Select each observed subvector before evaluating its smaller marginal."""
    event_ndim = 0 if family == "scalar" else 1
    batch_shape = jnp.broadcast_shapes(
        means.shape[: means.ndim - event_ndim],
        covariance.shape[:-2],
        values.shape[: values.ndim - event_ndim],
        mask.shape[: mask.ndim - event_ndim],
    )
    event_shape = () if family == "scalar" else (3,)
    means = jnp.broadcast_to(means, batch_shape + event_shape)
    values = jnp.broadcast_to(values, batch_shape + event_shape)
    mask = np.broadcast_to(np.asarray(mask), batch_shape + event_shape)
    covariance = jnp.broadcast_to(covariance, batch_shape + (3, 3))
    scores = []
    for index in np.ndindex(batch_shape):
        if not mask[index].any():
            scores.append(jnp.array(0.0))
        elif family == "scalar":
            scores.append(dist.LogNormal(means[index], 0.5).log_prob(values[index]))
        elif family == "student":
            scores.append(
                _distribution(family, means[index], covariance[index]).log_prob(
                    values[index]
                )
            )
        else:
            observed = np.flatnonzero(mask[index])
            marginal_mean = means[index][observed]
            marginal_covariance = covariance[index][observed[:, None], observed]
            scores.append(
                _distribution(family, marginal_mean, marginal_covariance).log_prob(
                    values[index][observed]
                )
            )
    return jnp.stack(scores).reshape(batch_shape)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("family", ["gaussian", "independent", "student", "scalar"])
def test_broadcast_scores_match_observed_marginals(case, family):
    means, covariance, values, mask = _inputs(case)
    if family == "student":
        mask = jnp.broadcast_to(jnp.all(mask, axis=-1, keepdims=True), mask.shape)
    if family == "scalar":
        means, values, mask = means[..., 0], values[..., 0], mask[..., 0]
    observation_dist = _distribution(family, means, covariance)
    expected = _reference(family, means, covariance, values, mask)
    actual = jax.jit(dsx.masked_observation_log_prob)(
        observation_dist, y=jnp.where(mask, values, jnp.nan), obs_mask=mask
    )
    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, atol=2e-6)


@pytest.mark.parametrize("case", ["distribution_batch", "all_pairs", "nested"])
@pytest.mark.parametrize("shared_covariance", [False, True])
def test_gaussian_broadcast_gradients_match_selected_marginals(case, shared_covariance):
    means, covariance, values, mask = _inputs(case, shared_covariance)

    def score(mean, cov, y):
        return dsx.masked_observation_log_prob(
            dist.MultivariateNormal(mean, cov),
            y=jnp.where(mask, y, jnp.nan),
            obs_mask=mask,
        ).sum()

    def reference(mean, cov, y):
        return _reference("gaussian", mean, cov, y, mask).sum()

    actual = jax.jit(jax.value_and_grad(score, argnums=(0, 1, 2)))(
        means, covariance, values
    )
    expected = jax.value_and_grad(reference, argnums=(0, 1, 2))(
        means, covariance, values
    )
    for result, target in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert jnp.all(jnp.isfinite(result))
        assert jnp.allclose(result, target, atol=1e-5, rtol=1e-5)


def test_unsupported_partial_row_is_rejected_among_batched_complete_rows():
    means, covariance, values, mask = _inputs("all_pairs")
    observation_dist = _distribution("student", means, covariance)
    with pytest.raises(ValueError, match="Partial missingness"):
        dsx.masked_observation_log_prob(observation_dist, y=values, obs_mask=mask)


def test_incompatible_leading_dimensions_are_rejected():
    observation_dist = dist.MultivariateNormal(jnp.zeros((3, 2)), jnp.eye(2))
    with pytest.raises(ValueError, match="broadcast"):
        dsx.masked_observation_log_prob(
            observation_dist, y=jnp.zeros((4, 2)), obs_mask=jnp.ones(2, dtype=bool)
        )
