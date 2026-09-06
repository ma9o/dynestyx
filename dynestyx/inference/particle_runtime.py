"""Parameter-aware particle targets composed from public model interpretations.

Models and schedules are functions of a consumer-defined array pytree. They are
evaluated inside the traced computation, so parameter values remain dynamic.
Numerical interpretation is selected by the model factory, before this layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from dynestyx.inference.particle_operators import build_particle_operators
from dynestyx.models import DynamicalModel


class ParticleSchedule(NamedTuple):
    """Time points and controls; transition controls index the destination step."""

    times: Any
    observation_controls: Any = None
    transition_controls: Any = None


@dataclass(frozen=True)
class Parameterization:
    """NumPyro owns transforms, conditional priors, factors, and Jacobians."""

    initial_position: Any
    unravel: Callable
    constrain: Callable
    log_prior: Callable


def prepare_parameterization(
    prior_model, key, *, model_args=(), model_kwargs=None, initial_values=None
):
    """Prepare a NumPyro prior program without freezing conditional distributions."""
    info, potential, postprocess, _ = initialize_model(
        key,
        prior_model,
        model_args=model_args,
        model_kwargs={} if model_kwargs is None else model_kwargs,
        init_strategy=init_to_value(values=initial_values),
    )
    initial, unravel = ravel_pytree(info.z)
    return Parameterization(
        initial,
        unravel,
        lambda z: postprocess(unravel(z)),
        lambda z: -potential(unravel(z)),
    )


@dataclass(frozen=True, eq=False)
class ParticleRuntime:
    """Pure state/parameter target implementing Cuthbert's kernel contract.

    The initial law is at schedule.times[0]. A numerical step is made only between
    successive states. Each transition's sample, density, and mean share the same
    distribution. Observation distributions may implement their own missingness
    semantics; otherwise pass ``marginalize_missing=True`` for supported families.
    """

    parameters: Parameterization
    context: Callable
    model: Callable[[Any], DynamicalModel]
    schedule: Callable[[Any], ParticleSchedule]
    observations: Any
    times: Any
    marginalize_missing: bool = True

    @property
    def initial_position(self):
        return self.parameters.initial_position

    def log_prior(self, position):
        return self.parameters.log_prior(position)

    def operators(self, context):
        return build_particle_operators(self.model(context))

    def initial_moments(self, context):
        distribution = self.model(context).initial_condition
        return jnp.asarray(distribution.mean), jnp.asarray(
            distribution.covariance_matrix
        )

    def initial_log_prob(self, context, state):
        return jnp.sum(self.operators(context).initial_log_prob(state))

    def initial_sample(self, key, context, num_particles):
        return jax.vmap(self.operators(context).initial_sample)(
            jax.random.split(key, num_particles)
        )

    def _interval(self, context, index):
        schedule = self.schedule(context)
        return dict(
            previous_control=(
                None
                if schedule.transition_controls is None
                else schedule.transition_controls[index]
            ),
            previous_time=schedule.times[jnp.maximum(index - 1, 0)],
            time=schedule.times[index],
        )

    def transition_log_prob(self, context, previous, current, index):
        return jnp.sum(
            self.operators(context).transition_log_prob(
                previous, current, **self._interval(context, index)
            )
        )

    def aligned_transition_log_prob(self, context, previous, current, index):
        return jax.vmap(lambda a, b: self.transition_log_prob(context, a, b, index))(
            previous, current
        )

    def pairwise_transition_log_prob(self, context, previous, current, index):
        values = self.operators(context).pairwise_transition_log_prob(
            previous, current, **self._interval(context, index)
        )
        return jnp.sum(values, axis=tuple(range(values.ndim - 2)))

    def transition_sample(self, key, context, previous, index):
        operators = self.operators(context)
        interval = self._interval(context, index)
        keys = jax.random.split(key, previous.shape[0])
        return jax.vmap(
            lambda k, state: operators.transition_sample(k, state, **interval)
        )(keys, previous)

    def initial_path(self, context):
        dynamics = self.model(context)
        initial = jnp.asarray(dynamics.initial_condition.mean)

        def step(previous, index):
            interval = self._interval(context, index)
            current = dynamics.state_evolution(
                previous,
                interval["previous_control"],
                interval["previous_time"],
                interval["time"],
            ).mean
            return current, current

        _, tail = jax.lax.scan(
            step, initial, jnp.arange(1, self.schedule(context).times.size)
        )
        return jnp.concatenate([initial[None], tail])

    def observation_increment(self, context, state, index, observations):
        schedule = self.schedule(context)
        observation = observations[index]
        mask = ~jnp.isnan(observation) if self.marginalize_missing else None
        return jnp.sum(
            self.operators(context).observation_log_prob(
                state,
                observation=observation,
                control=(
                    None
                    if schedule.observation_controls is None
                    else schedule.observation_controls[index]
                ),
                time=schedule.times[index],
                observation_mask=mask,
            )
        )

    def observation_log_probs(self, context, path, observations):
        return jax.vmap(
            lambda state, i: self.observation_increment(context, state, i, observations)
        )(path, jnp.arange(path.shape[0]))

    def path_log_prob(self, context, path, observations):
        transitions = jax.vmap(
            lambda a, b, i: self.transition_log_prob(context, a, b, i)
        )(path[:-1], path[1:], jnp.arange(1, path.shape[0]))
        return (
            self.initial_log_prob(context, path[0])
            + jnp.sum(transitions)
            + jnp.sum(self.observation_log_probs(context, path, observations))
        )

    def log_posterior_from_context(self, position, context, path, observations):
        path_density = self.path_log_prob(context, path, observations)
        return self.log_prior(position) + path_density, path_density

    def log_posterior(self, position, path, observations, times):
        return self.log_posterior_from_context(
            position, self.context(position, times), path, observations
        )[0]
