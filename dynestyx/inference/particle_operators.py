"""Backend-neutral particle-inference operations compiled from a Dynestyx model."""

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PRNGKeyArray, Real
from numpyro.distributions import Distribution

from dynestyx.inference.state_paths.score import compute_state_path_log_prob
from dynestyx.models import DynamicalModel
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    MissingObservationStrategy,
)

type State = Real[Array, " state_dim"] | Real[Array, ""]
type Control = Real[Array, " control_dim"] | Real[Array, ""] | None
type Time = float | int | Real[Array, ""]


class ParticleOperators(eqx.Module):
    """Particle sampling and scoring operations for one discrete Dynestyx model.

    Algorithm choices such as particle count, resampling, and conditioning on a
    reference path deliberately remain with the consuming inference backend.
    Transition-density methods require the model's transition distribution to
    implement ``log_prob``; sampling-only transitions can still use
    :meth:`transition_sample`.
    """

    dynamics: DynamicalModel

    def initial_sample(self, key: PRNGKeyArray) -> State:
        """Sample one state from the model's initial distribution."""
        return self.dynamics.initial_condition.sample(key)

    def initial_log_prob(self, state: State) -> Real[Array, ""]:
        """Evaluate the initial-state log density as a scalar score."""
        return jnp.asarray(self.dynamics.initial_condition.log_prob(state)).sum()

    def _transition_distribution(
        self,
        previous_state: State,
        *,
        previous_control: Control,
        previous_time: Time,
        time: Time,
    ) -> Distribution:
        transition = cast(DiscreteStateTransition, self.dynamics.state_evolution)
        return transition(
            previous_state,
            previous_control,
            previous_time,
            time,
        )

    def transition_sample(
        self,
        key: PRNGKeyArray,
        previous_state: State,
        *,
        previous_control: Control,
        previous_time: Time,
        time: Time,
    ) -> State:
        """Sample one state from the discrete transition distribution."""
        return self._transition_distribution(
            previous_state,
            previous_control=previous_control,
            previous_time=previous_time,
            time=time,
        ).sample(key)

    def transition_log_prob(
        self,
        previous_state: State,
        state: State,
        *,
        previous_control: Control,
        previous_time: Time,
        time: Time,
    ) -> Real[Array, ""]:
        """Evaluate one discrete transition log density as a scalar score."""
        transition = self._transition_distribution(
            previous_state,
            previous_control=previous_control,
            previous_time=previous_time,
            time=time,
        )
        return jnp.asarray(transition.log_prob(state)).sum()

    def pairwise_transition_log_prob(
        self,
        previous_states: Real[Array, "n_previous state_dim"]
        | Real[Array, " n_previous"],
        states: Real[Array, "n_current state_dim"] | Real[Array, " n_current"],
        *,
        previous_control: Control,
        previous_time: Time,
        time: Time,
    ) -> Real[Array, "n_previous n_current"]:
        """Score every previous/current pair, returning ``(n_previous, n_current)``."""

        def _from_previous(previous_state: State):
            return jax.vmap(
                lambda state: self.transition_log_prob(
                    previous_state,
                    state,
                    previous_control=previous_control,
                    previous_time=previous_time,
                    time=time,
                )
            )(states)

        return jax.vmap(_from_previous)(previous_states)

    def incremental_log_potential(
        self,
        state: State,
        *,
        observation: Real[Array, " observation_dim"] | Real[Array, ""],
        control: Control,
        time: Time,
    ) -> Real[Array, ""]:
        """Evaluate the current observation contribution as a scalar score."""
        observation_dist = self.dynamics.observation_model(state, control, time)
        return jnp.asarray(observation_dist.log_prob(observation)).sum()

    def trajectory_log_prob(
        self,
        *,
        state_path: Real[Array, "state_path_time state_dim"]
        | Real[Array, " state_path_time"],
        state_path_times: Real[Array, " state_path_time"],
        obs_times: Real[Array, " obs_time"] | None = None,
        obs_values: Real[Array, "obs_time observation_dim"]
        | Real[Array, " obs_time"]
        | None = None,
        obs_values_filled: Real[Array, "obs_time observation_dim"]
        | Real[Array, " obs_time"]
        | None = None,
        obs_mask: Bool[Array, "obs_time observation_dim"]
        | Bool[Array, " obs_time"]
        | None = None,
        missing_observation_strategy: MissingObservationStrategy = "auto",
        missing_obs_values: Real[Array, " n_missing_obs"]
        | Real[Array, " obs_time"]
        | Real[Array, "obs_time observation_dim"]
        | Real[Array, ""]
        | None = None,
        missing_obs_metadata: MissingObservationMetadata | None = None,
        ctrl_times: Real[Array, " ctrl_time"] | None = None,
        ctrl_values: Real[Array, "ctrl_time control_dim"]
        | Real[Array, " ctrl_time"]
        | None = None,
        chunk_size: int | None = 0,
        observations_are_exact_constraints: bool = False,
    ) -> Real[Array, "*log_prob_batch"]:
        """Evaluate the model's joint state-path and observation log density."""
        return compute_state_path_log_prob(
            self.dynamics,
            state_path=state_path,
            state_path_times=state_path_times,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_values_filled=obs_values_filled,
            obs_mask=obs_mask,
            missing_observation_strategy=missing_observation_strategy,
            missing_obs_values=missing_obs_values,
            missing_obs_metadata=missing_obs_metadata,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            chunk_size=chunk_size,
            observations_are_exact_constraints=observations_are_exact_constraints,
        )


def compile_particle_operators(dynamics: DynamicalModel) -> ParticleOperators:
    """Compile a discrete Dynestyx model into backend-neutral particle operations.

    Raw continuous-time models require a consumer-selected discretization before
    they have a sampleable transition distribution and an evaluable transition
    density on a discrete inference lattice.
    """
    if dynamics.continuous_time:
        raise TypeError(
            "compile_particle_operators requires a discrete-time DynamicalModel. "
            "Discretize continuous-time dynamics on the intended inference "
            "lattice before compiling the particle operators."
        )
    return ParticleOperators(dynamics=dynamics)


__all__ = ["ParticleOperators", "compile_particle_operators"]
