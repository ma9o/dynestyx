"""Backend-neutral particle-inference operations built from a Dynestyx model."""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Real

from dynestyx.inference.state_paths.score import compute_state_path_log_prob
from dynestyx.models import DynamicalModel
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    MissingObservationStrategy,
    _canonicalize_observation_distribution,
    _distribution_mode,
    masked_observation_log_prob,
)


class ParticleOperators(eqx.Module):
    """Particle sampling and scoring operations for one discrete Dynestyx model.

    Algorithm choices such as particle count, resampling, and conditioning on a
    reference path deliberately remain with the consuming inference backend.
    Transition-density methods require the model's transition result to
    implement ``log_prob``; sampling-only transition results can still use
    `transition_sample`.

    Score methods retain any batch axes returned by the underlying NumPyro
    distributions. Consuming algorithms are responsible for reducing those
    axes when they require scalar particle weights.

    Attributes:
        dynamics: Discrete-time model that supplies the initial, transition,
            and observation distributions.
    """

    dynamics: DynamicalModel

    def initial_sample(
        self, rng_key: PRNGKeyArray
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        """Sample one state from the initial distribution.

        Args:
            rng_key: JAX PRNG key used for sampling.

        Returns:
            Array: Sampled initial state.
        """
        return self.dynamics.initial_condition.sample(rng_key)

    def initial_log_prob(
        self, state: Real[Array, " state_dim"] | Real[Array, ""]
    ) -> Real[Array, "*log_prob_batch"]:
        """Evaluate the initial-state log density.

        Args:
            state: State at the model's initial time.

        Returns:
            Array: Initial-state log density, retaining distribution batch axes.
        """
        return jnp.asarray(self.dynamics.initial_condition.log_prob(state))

    def _transition_distribution(
        self,
        previous_state: Real[Array, " state_dim"] | Real[Array, ""],
        *,
        previous_control: Real[Array, " control_dim"] | Real[Array, ""] | None,
        previous_time: float | int | Real[Array, ""],
        time: float | int | Real[Array, ""],
    ) -> Any:
        transition = cast(DiscreteStateTransition, self.dynamics.state_evolution)
        return transition(
            previous_state,
            previous_control,
            previous_time,
            time,
        )

    def transition_sample(
        self,
        rng_key: PRNGKeyArray,
        previous_state: Real[Array, " state_dim"] | Real[Array, ""],
        *,
        previous_control: Real[Array, " control_dim"] | Real[Array, ""] | None,
        previous_time: float | int | Real[Array, ""],
        time: float | int | Real[Array, ""],
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        """Sample one state from a discrete transition distribution.

        Args:
            rng_key: JAX PRNG key used for sampling.
            previous_state: State at `previous_time`.
            previous_control: Control applied at `previous_time`, or `None` for
                an uncontrolled model.
            previous_time: Time associated with `previous_state`.
            time: Time of the state to sample.

        Returns:
            Array: State sampled at `time`.
        """
        return self._transition_distribution(
            previous_state,
            previous_control=previous_control,
            previous_time=previous_time,
            time=time,
        ).sample(rng_key)

    def transition_log_prob(
        self,
        previous_state: Real[Array, " state_dim"] | Real[Array, ""],
        state: Real[Array, " state_dim"] | Real[Array, ""],
        *,
        previous_control: Real[Array, " control_dim"] | Real[Array, ""] | None,
        previous_time: float | int | Real[Array, ""],
        time: float | int | Real[Array, ""],
    ) -> Real[Array, "*log_prob_batch"]:
        """Evaluate one discrete transition log density.

        Args:
            previous_state: State at `previous_time`.
            state: State at `time`.
            previous_control: Control applied at `previous_time`, or `None` for
                an uncontrolled model.
            previous_time: Time associated with `previous_state`.
            time: Time associated with `state`.

        Returns:
            Array: Transition log density, retaining distribution batch axes.
        """
        transition = self._transition_distribution(
            previous_state,
            previous_control=previous_control,
            previous_time=previous_time,
            time=time,
        )
        return jnp.asarray(transition.log_prob(state))

    def pairwise_transition_log_prob(
        self,
        previous_states: Real[Array, "n_previous state_dim"]
        | Real[Array, " n_previous"],
        states: Real[Array, "n_current state_dim"] | Real[Array, " n_current"],
        *,
        previous_control: Real[Array, " control_dim"] | Real[Array, ""] | None,
        previous_time: float | int | Real[Array, ""],
        time: float | int | Real[Array, ""],
    ) -> Real[Array, "*log_prob_batch n_previous n_current"]:
        """Evaluate the transition log density for every state pair.

        The returned full pair matrix contains
        `n_previous * n_current` entries after any distribution batch axes.

        Args:
            previous_states: Candidate states at `previous_time`.
            states: Candidate states at `time`.
            previous_control: Control applied at `previous_time`, or `None` for
                an uncontrolled model.
            previous_time: Time associated with `previous_states`.
            time: Time associated with `states`.

        Returns:
            Array: Pairwise transition log densities. Distribution batch axes
                lead the `n_previous` and `n_current` axes.
        """

        def _from_previous(
            previous_state: Real[Array, " state_dim"] | Real[Array, ""],
        ):
            return jax.vmap(
                lambda state: self.transition_log_prob(
                    previous_state,
                    state,
                    previous_control=previous_control,
                    previous_time=previous_time,
                    time=time,
                )
            )(states)

        pairwise_log_prob = jax.vmap(_from_previous)(previous_states)
        return jnp.moveaxis(pairwise_log_prob, (0, 1), (-2, -1))

    def observation_log_prob(
        self,
        state: Real[Array, " state_dim"] | Real[Array, ""],
        *,
        observation: Real[Array, " observation_dim"] | Real[Array, ""],
        control: Real[Array, " control_dim"] | Real[Array, ""] | None,
        time: float | int | Real[Array, ""],
        observation_mask: Array | None = None,
    ) -> Real[Array, "*log_prob_batch"]:
        """Evaluate an emission factor, before algorithm-specific proposal corrections.

        Args:
            state: State associated with the observation.
            observation: Observed value at `time`.
            control: Control at `time`, or `None` for an uncontrolled model.
            time: Time associated with `state` and `observation`.

        Returns:
            Array: Observation log density, retaining distribution batch axes.
        """
        observation_dist = self.dynamics.observation_model(state, control, time)
        if observation_mask is None:
            return jnp.asarray(observation_dist.log_prob(observation))
        values = jnp.atleast_1d(jnp.asarray(observation))
        mask = jnp.atleast_1d(observation_mask)
        observation_dist = _canonicalize_observation_distribution(
            observation_dist, observation_dim=values.shape[-1]
        )
        mode = _distribution_mode(observation_dist, has_partial_missing=True)
        return masked_observation_log_prob(
            observation_dist,
            y=jnp.where(mask, values, 0),
            obs_mask=mask,
            row_has_any_observed=jnp.any(mask),
            observation_dim=values.shape[-1],
            has_partial_missing=True,
            expected_mode=mode,
            expected_event_shape=tuple(observation_dist.event_shape),
        )

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
    ) -> Real[Array, "*log_prob_batch"]:
        """Evaluate the joint state-path and observation log density.

        If either `obs_times` or `obs_values` is `None`, observation terms are
        omitted.

        Args:
            state_path: Complete state values, with a leading time axis aligned
                with `state_path_times`.
            state_path_times: Times associated with `state_path`.
            obs_times: Times associated with `obs_values`. Every observation
                time must occur exactly in `state_path_times`.
            obs_values: Observation values, including any missing entries.
            missing_observation_strategy: Method used to handle missing
                observations.
            missing_obs_values: Values used to complete missing observations
                when augmentation is active.
            missing_obs_metadata: Positions, times, and component indices for
                `missing_obs_values`.
            ctrl_times: Times associated with `ctrl_values`. Required control
                times must occur exactly in this array.
            ctrl_values: Control values, or `None` for an uncontrolled model.
            chunk_size: Batch size passed to `jax.lax.map` while scoring terms.
                The default, `0`, evaluates all terms with one `jax.vmap`;
                `None` maps one term at a time.

        Returns:
            Array: Joint log density, retaining distribution batch axes.

        Raises:
            ValueError: If path, observation, control, or missing-observation
                inputs are inconsistent.
            eqx.EquinoxRuntimeError: If a required observation or control time
                is absent from its source time array.
            NotImplementedError: If the selected missing-observation strategy
                is unsupported by the observation distribution.
        """
        return compute_state_path_log_prob(
            self.dynamics,
            state_path=state_path,
            state_path_times=state_path_times,
            obs_times=obs_times,
            obs_values=obs_values,
            missing_observation_strategy=missing_observation_strategy,
            missing_obs_values=missing_obs_values,
            missing_obs_metadata=missing_obs_metadata,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            chunk_size=chunk_size,
        )


def build_particle_operators(dynamics: DynamicalModel) -> ParticleOperators:
    """Build backend-neutral particle operations from a discrete model.

    Raw continuous-time models require a consumer-selected discretization before
    they have a sampleable transition distribution and an evaluable transition
    density on a discrete inference lattice.

    Args:
        dynamics: Discrete-time Dynestyx model to adapt.

    Returns:
        ParticleOperators: Sampling and scoring operations backed by `dynamics`.

    Raises:
        TypeError: If `dynamics` has continuous-time state evolution.
    """
    if dynamics.continuous_time:
        raise TypeError(
            "build_particle_operators requires a discrete-time DynamicalModel. "
            "Discretize continuous-time dynamics on the intended inference "
            "lattice before building the particle operators."
        )
    return ParticleOperators(dynamics=dynamics)


__all__ = ["ParticleOperators", "build_particle_operators"]
