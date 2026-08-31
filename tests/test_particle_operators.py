"""Tests for backend-neutral particle operators."""

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest

import dynestyx as dsx
from dynestyx.discretizers import DiffraxSampleConfig
from dynestyx.inference.configs.simulator import SDESimulatorConfig


def _controlled_discrete_model() -> dsx.DynamicalModel:
    return dsx.LTI_discrete(
        A=jnp.array([[0.7, 0.1], [-0.2, 0.8]]),
        Q=jnp.array([[0.3, 0.05], [0.05, 0.2]]),
        B=jnp.array([[0.4], [-0.1]]),
        b=jnp.array([0.2, -0.3]),
        H=jnp.array([[1.0, -0.5]]),
        R=jnp.array([[0.4]]),
        D=jnp.array([[0.25]]),
        d=jnp.array([0.1]),
        initial_mean=jnp.array([0.1, -0.2]),
        initial_cov=jnp.array([[0.8, 0.1], [0.1, 0.6]]),
    )


def _diffrax_config() -> DiffraxSampleConfig:
    return DiffraxSampleConfig(
        SDESimulatorConfig(
            source="diffrax",
            solver=dfx.Euler(),
            dt0=0.01,
            max_steps=100,
        )
    )


def test_particle_operators_match_model_sampling_and_scores():
    dynamics = _controlled_discrete_model()
    operators = dsx.build_particle_operators(dynamics)
    initial_key = jr.PRNGKey(2)
    transition_key = jr.PRNGKey(3)
    previous_state = jnp.array([0.3, -0.4])
    state = jnp.array([-0.2, 0.7])
    control = jnp.array([0.5])
    observation = jnp.array([0.8])
    previous_time = jnp.array(0.2)
    time = jnp.array(0.9)
    transition = dynamics.state_evolution(
        previous_state,
        control,
        previous_time,
        time,
    )

    assert jnp.array_equal(
        operators.initial_sample(initial_key),
        dynamics.initial_condition.sample(initial_key),
    )
    assert jnp.allclose(
        operators.initial_log_prob(state),
        dynamics.initial_condition.log_prob(state),
    )
    assert jnp.array_equal(
        operators.transition_sample(
            transition_key,
            previous_state,
            previous_control=control,
            previous_time=previous_time,
            time=time,
        ),
        transition.sample(transition_key),
    )
    assert jnp.allclose(
        operators.transition_log_prob(
            previous_state,
            state,
            previous_control=control,
            previous_time=previous_time,
            time=time,
        ),
        transition.log_prob(state),
    )
    assert jnp.allclose(
        operators.incremental_log_potential(
            state,
            observation=observation,
            control=control,
            time=time,
        ),
        dynamics.observation_model(state, control, time).log_prob(observation),
    )


def test_transition_sample_accepts_black_box_transition_result():
    class _BlackBoxTransition:
        def __init__(self, location):
            self.location = location

        def sample(self, key):
            return self.location + 0.05 * jr.normal(key, self.location.shape)

        def shape(self):
            return self.location.shape

    def _state_evolution(x, u, t_now, t_next):
        location = jnp.tanh(x) + u * (t_next - t_now)
        return _BlackBoxTransition(location)

    dynamics = dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.array([1.0]), 0.1 * jnp.eye(1)),
        state_evolution=_state_evolution,
        observation_model=dsx.LinearGaussianObservation(H=jnp.eye(1), R=jnp.eye(1)),
        control_dim=1,
    )
    operators = dsx.build_particle_operators(dynamics)
    key = jr.PRNGKey(4)
    previous_state = jnp.array([0.3])
    control = jnp.array([0.2])
    previous_time = jnp.array(0.1)
    time = jnp.array(0.6)

    actual = operators.transition_sample(
        key,
        previous_state,
        previous_control=control,
        previous_time=previous_time,
        time=time,
    )
    expected_location = jnp.tanh(previous_state) + control * (time - previous_time)
    expected = expected_location + 0.05 * jr.normal(key, expected_location.shape)
    assert jnp.array_equal(actual, expected)


def test_pairwise_transition_log_prob_preserves_batch_axes_and_pair_order():
    coefficients = jnp.array([0.5, 1.5])
    variances = jnp.array([0.2, 0.4])
    with dsx.plate("members", 2):
        dynamics = dsx.LTI_discrete(
            A=coefficients[:, None, None],
            Q=variances[:, None, None],
            H=jnp.ones((2, 1, 1)),
            R=jnp.ones((2, 1, 1)),
            initial_mean=jnp.zeros((2, 1)),
            initial_cov=jnp.tile(jnp.eye(1), (2, 1, 1)),
        )
    operators = dsx.build_particle_operators(dynamics)
    previous_states = jnp.array([[0.0], [1.0]])
    states = jnp.array([[-1.0], [0.5], [2.0]])

    actual = operators.pairwise_transition_log_prob(
        previous_states,
        states,
        previous_control=None,
        previous_time=0.0,
        time=1.0,
    )

    means = coefficients[:, None, None] * previous_states[None, :, 0, None]
    errors = states[None, None, :, 0] - means
    expected = -0.5 * (
        jnp.log(2.0 * jnp.pi * variances[:, None, None])
        + errors**2 / variances[:, None, None]
    )
    assert actual.shape == (2, 2, 3)
    assert jnp.allclose(actual, expected)


def test_trajectory_log_prob_matches_manual_decomposition():
    state_times = jnp.array([0.0, 1.0, 2.0])
    state_path = jnp.array([0.2, -0.1, 0.4])
    control_values = jnp.array([0.5, -0.25, 0.75])
    observation_times = jnp.array([0.0, 2.0])
    observation_values = jnp.array([0.3, -0.2])
    dynamics = dsx.DynamicalModel(
        control_dim=1,
        initial_condition=dist.Normal(0.0, 1.1),
        state_evolution=lambda x, u, t_now, t_next: dist.Normal(
            0.7 * x + 0.5 * u,
            0.3,
        ),
        observation_model=lambda x, u, t: dist.Normal(x - 0.25 * u, 0.4),
    )
    operators = dsx.build_particle_operators(dynamics)

    actual = operators.trajectory_log_prob(
        state_path=state_path,
        state_path_times=state_times,
        obs_times=observation_times,
        obs_values=observation_values,
        ctrl_times=state_times,
        ctrl_values=control_values,
    )

    expected = dynamics.initial_condition.log_prob(state_path[0])
    expected = expected + dynamics.state_evolution(
        state_path[0], control_values[0], state_times[0], state_times[1]
    ).log_prob(state_path[1])
    expected = expected + dynamics.state_evolution(
        state_path[1], control_values[1], state_times[1], state_times[2]
    ).log_prob(state_path[2])
    expected = expected + dynamics.observation_model(
        state_path[0], control_values[0], observation_times[0]
    ).log_prob(observation_values[0])
    expected = expected + dynamics.observation_model(
        state_path[2], control_values[2], observation_times[1]
    ).log_prob(observation_values[1])
    assert jnp.allclose(actual, expected)


def test_build_particle_operators_rejects_continuous_model():
    dynamics = dsx.LTI_continuous(
        A=jnp.array([[-0.4]]),
        L=jnp.array([[0.2]]),
        H=jnp.eye(1),
        R=jnp.eye(1),
    )

    with pytest.raises(TypeError, match="requires a discrete-time DynamicalModel"):
        dsx.build_particle_operators(dynamics)


def test_transition_density_reports_sample_only_capability():
    continuous = dsx.LTI_continuous(
        A=jnp.array([[-0.4]]),
        L=jnp.array([[0.2]]),
        H=jnp.eye(1),
        R=jnp.eye(1),
    )
    dynamics = dsx.discretize_dynamics(continuous, _diffrax_config())
    operators = dsx.build_particle_operators(dynamics)

    with pytest.raises(
        NotImplementedError,
        match="DiffraxSampleConfig provides sampling only",
    ):
        operators.transition_log_prob(
            jnp.array([0.1]),
            jnp.array([0.2]),
            previous_control=None,
            previous_time=0.0,
            time=0.1,
        )
