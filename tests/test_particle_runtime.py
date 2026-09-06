"""Independent mathematical checks for the simulated merged library stack."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from cuthbert.mcmc.conditional import build_conditional_dsmc, build_conditional_smc
from cuthbert.mcmc.marginal_particle_gibbs import build_marginal_particle_gibbs_kernel
from cuthbert.mcmc.state import TrajectoryMCMCState

import dynestyx as dsx
from dynestyx.inference.configs.discretizer import EulerMaruyamaConfig
from dynestyx.inference.particle_runtime import (
    ParticleRuntime,
    ParticleSchedule,
    prepare_parameterization,
)

jax.config.update("jax_enable_x64", True)


class Context(NamedTuple):
    theta: jax.Array
    times: jax.Array


def linear_target():
    def prior():
        numpyro.sample("theta", dist.Normal(0.0, 1.0))

    parameters = prepare_parameterization(
        prior, jax.random.key(1), initial_values={"theta": 0.0}
    )
    times = jnp.array([0.0, 0.4, 1.1])
    observations = jnp.array([[0.2], [-0.3], [0.4]])

    def model(ctx):
        continuous = dsx.DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                jnp.array([ctx.theta]), covariance_matrix=jnp.array([[0.5]])
            ),
            state_evolution=dsx.ContinuousTimeStateEvolution(
                drift=lambda x, u, t: ctx.theta - 0.3 * x,
                diffusion=dsx.FullDiffusion(jnp.array([[0.3]])),
            ),
            observation_model=lambda x, u, t: dist.MultivariateNormal(
                x, covariance_matrix=jnp.array([[0.2]])
            ),
            t0=ctx.times[0],
        )
        return dsx.discretize_dynamics(
            continuous, EulerMaruyamaConfig(covariance_jitter=1e-8)
        )

    return ParticleRuntime(
        parameters,
        lambda z, ts: Context(z[0], ts),
        model,
        lambda ctx: ParticleSchedule(ctx.times),
        observations,
        times,
    )


def analytic_joint(target):
    """Build the Gaussian precision independently from the generative equations."""
    times = np.asarray(target.times)
    n = times.size
    residual_map = np.eye(n + 1)
    residual_map[1, 0] = -1.0
    variances = [1.0, 0.5]
    for i, dt in enumerate(np.diff(times), start=2):
        residual_map[i, 0] = -dt
        residual_map[i, i - 1] = -(1 - 0.3 * dt)
        variances.append(0.09 * dt + 1e-8)
    precision = residual_map.T @ np.diag(1 / np.array(variances)) @ residual_map
    h = np.eye(n + 1)[1:]
    precision += h.T @ h / 0.2
    covariance = np.linalg.inv(precision)
    mean = covariance @ h.T @ np.asarray(target.observations[:, 0]) / 0.2
    return mean, covariance


def assert_stationary(samples, mean, covariance):
    whitened = np.linalg.solve(
        np.linalg.cholesky(covariance), (np.asarray(samples) - mean).T
    ).T
    n = samples.shape[0]
    assert np.max(np.abs(whitened.mean(axis=0))) < 6 / np.sqrt(n)
    assert np.max(np.abs(whitened.T @ whitened / n - np.eye(mean.size))) < 6 * np.sqrt(
        2 / n
    )


def test_parameterization_replays_conditional_priors_and_jacobians():
    def prior():
        scale = numpyro.sample("scale", dist.LogNormal(0.0, 0.5))
        numpyro.sample("theta", dist.Normal(0.0, scale))

    params = prepare_parameterization(
        prior, jax.random.key(2), initial_values={"scale": 1.0, "theta": 0.0}
    )
    position = jnp.array([np.log(2.0), 0.7])
    expected = (
        dist.LogNormal(0.0, 0.5).log_prob(2.0)
        + np.log(2.0)
        + dist.Normal(0.0, 2.0).log_prob(0.7)
    )
    np.testing.assert_allclose(params.log_prior(position), expected, atol=1e-12)
    np.testing.assert_allclose(params.constrain(position)["scale"], 2.0)


def test_path_density_gradient_and_parameter_changes():
    target = linear_target()
    mean, covariance = analytic_joint(target)
    precision = np.linalg.inv(covariance)

    def density(point):
        return target.log_posterior(
            point[:1], point[1:, None], target.observations, target.times
        )

    point = jnp.array([0.4, -0.1, 0.3, 0.2])
    np.testing.assert_allclose(
        jax.grad(density)(point), -precision @ (point - mean), atol=1e-9
    )
    traces = []

    @jax.jit
    def compiled(point):
        traces.append(1)
        return density(point)

    for offset in (0.0, 0.2, -0.3):
        np.testing.assert_allclose(
            compiled(point + offset), density(point + offset), atol=1e-12
        )
    assert len(traces) == 1


@pytest.mark.parametrize("algorithm", ["sequential", "dsmc"])
def test_conditional_kernels_preserve_gaussian_posterior(algorithm):
    target = linear_target()
    mean, covariance = analytic_joint(target)
    theta = 0.25
    conditional_mean = mean[1:] + covariance[1:, 0] / covariance[0, 0] * (
        theta - mean[0]
    )
    conditional_cov = (
        covariance[1:, 1:]
        - np.outer(covariance[1:, 0], covariance[0, 1:]) / covariance[0, 0]
    )
    n = 4096
    refs = jax.random.multivariate_normal(
        jax.random.key(3),
        jnp.asarray(conditional_mean),
        jnp.asarray(conditional_cov),
        shape=(n,),
    )[:, :, None]
    builder = (
        build_conditional_smc if algorithm == "sequential" else build_conditional_dsmc
    )
    kernel = builder(target, jnp.array([theta]), num_particles=8)
    step = jax.jit(jax.vmap(lambda key, path: kernel.step(key, path).path))
    samples = step(jax.random.split(jax.random.key(4), n), refs)
    assert_stationary(samples[:, :, 0], conditional_mean, conditional_cov)
    assert np.mean(np.asarray(samples != refs)) > 0.05


@pytest.mark.parametrize("proposal", ["random_walk", "pseudo_langevin"])
def test_joint_kernel_preserves_analytic_posterior(proposal):
    target = linear_target()
    mean, covariance = analytic_joint(target)
    n = 4096
    points = jax.random.multivariate_normal(
        jax.random.key(5), jnp.asarray(mean), jnp.asarray(covariance), shape=(n,)
    )

    def state(point):
        position, path = point[:1], point[1:, None]
        ctx = target.context(position, target.times)
        complete, trajectory = target.log_posterior_from_context(
            position, ctx, path, target.observations
        )
        return TrajectoryMCMCState(
            position,
            ctx,
            path,
            trajectory,
            complete,
            jnp.full((3,), 0.2),
            jnp.asarray(0.08),
            None,
            None,
        )

    states = jax.vmap(state)(points)
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_particles=8,
        num_parameter_particles=2,
        param_step_size=0.08,
        parameter_proposal=proposal,
    )

    @jax.jit
    def advance(states):
        def one(states, key):
            states, _ = jax.vmap(kernel.step_fn)(states, jax.random.split(key, n))
            return states, None

        return jax.lax.scan(one, states, jax.random.split(jax.random.key(6), 5))[0]

    states = advance(states)
    samples = jnp.concatenate(
        [states.position, states.latent_trajectory[:, :, 0]], axis=1
    )
    assert_stationary(samples, mean, covariance)


def test_masked_factors_match_public_path_scoring():
    target = linear_target()
    ctx = target.context(jnp.array([0.2]), target.times)
    operators = target.operators(ctx)
    path = jnp.array([[0.1], [0.2], [-0.1]])
    observations = target.observations.at[1, 0].set(jnp.nan)
    expected = operators.trajectory_log_prob(
        state_path=path,
        state_path_times=target.times,
        obs_times=target.times,
        obs_values=observations,
    )
    actual = target.path_log_prob(ctx, path, observations)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_irregular_em_uses_left_time_and_identical_sampling_density_covariance():
    evolution = dsx.StochasticContinuousTimeStateEvolution(
        drift=lambda x, u, t: t + u - x**3,
        diffusion=dsx.FullDiffusion(jnp.array([[0.02]])),
    )
    numerical = dsx.discretize_state_evolution(
        evolution, EulerMaruyamaConfig(covariance_jitter=0.01)
    )
    previous, control, t0, t1 = jnp.array([0.4]), jnp.array([0.7]), 0.2, 0.23
    law = numerical(previous, control, t0, t1)
    mean = previous + (t1 - t0) * (t0 + control - previous**3)
    covariance = jnp.array([[0.0004 * (t1 - t0) + 0.01]])
    np.testing.assert_allclose(law.mean, mean, atol=1e-12)
    np.testing.assert_allclose(law.covariance_matrix, covariance, atol=1e-12)
    values = law.sample(jax.random.key(10), (16384,))
    assert_stationary(values, np.asarray(mean), np.asarray(covariance))
    reference = dist.MultivariateNormal(mean, covariance_matrix=covariance)
    np.testing.assert_allclose(
        law.log_prob(jnp.array([0.6])), reference.log_prob(jnp.array([0.6])), atol=1e-12
    )


def test_dsmc_uses_non_gaussian_initial_density():
    target = linear_target()
    weights, locations, variance, noise = (
        jnp.array([0.3, 0.7]),
        jnp.array([-1.1, 1.3]),
        0.15,
        0.3,
    )
    observation = 0.25
    initial = dist.MixtureSameFamily(
        dist.Categorical(probs=weights),
        dist.MultivariateNormal(
            locations[:, None], covariance_matrix=jnp.eye(1) * variance
        ),
    )

    def model(ctx):
        return dsx.DynamicalModel(
            initial_condition=initial,
            state_evolution=lambda x, u, t_now, t_next: dist.Independent(
                dist.Normal(x, 1.0), 1
            ),
            observation_model=lambda x, u, t: dist.Independent(
                dist.Normal(x, jnp.sqrt(noise)), 1
            ),
        )

    from dataclasses import replace

    target = replace(
        target,
        model=model,
        times=jnp.array([0.0]),
        observations=jnp.array([[observation]]),
    )
    posterior_variance = 1 / (1 / variance + 1 / noise)
    posterior_locations = posterior_variance * (
        locations / variance + observation / noise
    )
    log_weights = jnp.log(weights) + dist.Normal(
        locations, jnp.sqrt(variance + noise)
    ).log_prob(observation)
    posterior = dist.MixtureSameFamily(
        dist.Categorical(logits=log_weights),
        dist.Normal(posterior_locations, jnp.sqrt(posterior_variance)),
    )
    refs = posterior.sample(jax.random.key(11), (8192,))[:, None, None]
    kernel = build_conditional_dsmc(
        target, jnp.array([0.0]), num_particles=8, delta=0.7
    )
    result = jax.jit(jax.vmap(lambda key, path: kernel.step(key, path).path))(
        jax.random.split(jax.random.key(12), refs.shape[0]), refs
    )
    assert_stationary(
        result[:, 0],
        np.asarray(posterior.mean)[None],
        np.asarray(posterior.variance).reshape(1, 1),
    )
    expected_positive = jnp.sum(
        jax.nn.softmax(log_weights)
        * (1 - dist.Normal(posterior_locations, jnp.sqrt(posterior_variance)).cdf(0.0))
    )
    assert abs(float(jnp.mean(result > 0)) - float(expected_positive)) < 0.03


def test_density_kernel_rejects_sample_only_interpretation():
    from dataclasses import replace

    import diffrax

    from dynestyx.discretizers import DiffraxSampleConfig
    from dynestyx.inference.configs.simulator import SDESimulatorConfig

    target = linear_target()
    config = DiffraxSampleConfig(
        SDESimulatorConfig(
            source="diffrax", solver=diffrax.Euler(), dt0=0.01, max_steps=200
        )
    )

    def sample_model(ctx):
        continuous = dsx.DynamicalModel(
            initial_condition=dist.Independent(dist.Normal(jnp.zeros(1), 1.0), 1),
            state_evolution=dsx.ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -(x**3), diffusion=dsx.FullDiffusion(jnp.eye(1))
            ),
            observation_model=lambda x, u, t: dist.Independent(dist.Normal(x, 1.0), 1),
        )
        return dsx.discretize_dynamics(continuous, config)

    target = replace(target, model=sample_model)
    with pytest.raises(NotImplementedError, match="sampling only"):
        build_conditional_dsmc(target, target.initial_position)
    kernel = build_conditional_smc(target, target.initial_position, num_particles=4)
    path = kernel.step(jax.random.key(13), jnp.zeros((3, 1))).path
    assert jnp.all(jnp.isfinite(path))


def test_nonlinear_poisson_dsmc_preserves_grid_reference():
    """An independent grid smoother checks nonlinear, non-Gaussian invariance."""
    from dataclasses import replace

    from scipy.special import gammaln, logsumexp

    target = linear_target()
    observations = jnp.array([[1.0], [0.0], [2.0]])

    def model(ctx):
        continuous = dsx.DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                jnp.zeros(1), covariance_matrix=jnp.array([[0.6]])
            ),
            state_evolution=dsx.ContinuousTimeStateEvolution(
                drift=lambda x, u, t: ctx.theta - 0.3 * x - 0.2 * x**3,
                diffusion=dsx.FullDiffusion(jnp.array([[0.6]])),
            ),
            observation_model=lambda x, u, t: dist.Poisson(jnp.exp(x)),
        )
        return dsx.discretize_dynamics(
            continuous, EulerMaruyamaConfig(covariance_jitter=1e-8)
        )

    target = replace(target, model=model, observations=observations)
    theta = 0.15
    grid = np.linspace(-4.0, 4.0, 401)
    y = np.asarray(observations[:, 0])
    emissions = y[:, None] * grid - np.exp(grid) - gammaln(y[:, None] + 1)
    initial = -0.5 * (grid**2 / 0.6 + np.log(2 * np.pi * 0.6))
    transitions = []
    for dt in np.diff(target.times):
        means = grid + dt * (theta - 0.3 * grid - 0.2 * grid**3)
        variance = 0.36 * dt + 1e-8
        transitions.append(
            -0.5
            * (
                (grid[None, :] - means[:, None]) ** 2 / variance
                + np.log(2 * np.pi * variance)
            )
        )
    alpha = [initial + emissions[0]]
    for i, transition in enumerate(transitions):
        alpha.append(
            logsumexp(alpha[-1][:, None] + transition, axis=0) + emissions[i + 1]
        )
    beta = [np.zeros_like(grid)]
    for i in (1, 0):
        beta.insert(
            0, logsumexp(transitions[i] + (emissions[i + 1] + beta[0])[None, :], axis=1)
        )
    marginals = [np.exp(a + b - logsumexp(a + b)) for a, b in zip(alpha, beta)]
    expected_mean = np.array([p @ grid for p in marginals])
    expected_var = np.array(
        [p @ (grid - mean) ** 2 for p, mean in zip(marginals, expected_mean)]
    )
    # Bound truncation error separately from the seeded Monte Carlo tolerance.
    assert max(p[0] + p[-1] for p in marginals) < 1e-8
    n = 4096
    rng = np.random.default_rng(20)
    indices = rng.choice(len(grid), size=n, p=np.exp(alpha[-1] - logsumexp(alpha[-1])))
    ref = [grid[indices]]
    for i in (1, 0):
        logp = alpha[i][:, None] + transitions[i][:, indices]
        cumulative = np.cumsum(np.exp(logp - logsumexp(logp, axis=0)), axis=0)
        indices = np.sum(cumulative < rng.random(n), axis=0)
        ref.insert(0, grid[indices])
    ref = jnp.asarray(np.stack(ref, axis=1)[..., None])
    kernel = build_conditional_dsmc(
        target, jnp.array([theta]), num_particles=8, delta=0.4
    )
    samples = jax.jit(jax.vmap(lambda key, path: kernel.step(key, path).path))(
        jax.random.split(jax.random.key(21), n), ref
    )[:, :, 0]
    values = np.asarray(samples)
    assert (
        np.max(np.abs((values.mean(0) - expected_mean) / np.sqrt(expected_var)))
        < 6 / np.sqrt(n) + 0.005
    )
    assert np.max(np.abs(values.var(0) / expected_var - 1)) < 6 * np.sqrt(2 / n) + 0.005
    for i, p in enumerate(marginals):
        assert abs(np.mean(values[:, i] <= 0) - p[grid <= 0].sum()) < 0.05
