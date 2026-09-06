"""Chain adaptation and summaries for structured particle inference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from blackjax.adaptation.step_size import DualAveragingAdaptationState
from cuthbert.mcmc.state import TrajectoryMCMCState


def _adapt_scale(
    scale: jnp.ndarray,
    *,
    accepted: jnp.ndarray,
    target_accept: float,
    adaptation_rate: float,
    min_scale: float = 1e-6,
    max_scale: float = 1e3,
) -> jnp.ndarray:
    """Simple exponential step-size adaptation on per-step binary accept."""
    dtype = scale.dtype
    accepted_arr = jnp.asarray(accepted, dtype=dtype)
    target_accept_arr = jnp.asarray(target_accept, dtype=dtype)
    adaptation_rate_arr = jnp.asarray(adaptation_rate, dtype=dtype)
    min_scale_arr = jnp.asarray(min_scale, dtype=dtype)
    max_scale_arr = jnp.asarray(max_scale, dtype=dtype)
    factor = jnp.exp(adaptation_rate_arr * (accepted_arr - target_accept_arr))
    return jnp.clip(scale * factor, min_scale_arr, max_scale_arr)


def _latent_summary_from_chain_moments(
    chain_means: jnp.ndarray,
    chain_stds: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    pooled_mean = jnp.mean(chain_means, axis=0)
    pooled_second_moment = jnp.mean(
        chain_stds * chain_stds + chain_means * chain_means, axis=0
    )
    pooled_var = jnp.maximum(pooled_second_moment - pooled_mean * pooled_mean, 0.0)
    return {
        "chain_mean": chain_means,
        "chain_std": chain_stds,
        "mean": pooled_mean,
        "std": jnp.sqrt(pooled_var),
    }


def _clip_dual_averaging_state(
    da_state: DualAveragingAdaptationState,
    *,
    min_scale: float | None,
    max_scale: float | None,
) -> DualAveragingAdaptationState:
    if min_scale is None and max_scale is None:
        return da_state

    log_min = (
        None
        if min_scale is None
        else jnp.log(jnp.asarray(min_scale, dtype=da_state.mu.dtype))
    )
    log_max = (
        None
        if max_scale is None
        else jnp.log(jnp.asarray(max_scale, dtype=da_state.mu.dtype))
    )

    def _clip_log_value(value: jnp.ndarray) -> jnp.ndarray:
        clipped = value
        if log_min is not None:
            clipped = jnp.maximum(clipped, log_min)
        if log_max is not None:
            clipped = jnp.minimum(clipped, log_max)
        return clipped

    return DualAveragingAdaptationState(
        log_step_size=_clip_log_value(da_state.log_step_size),
        log_step_size_avg=_clip_log_value(da_state.log_step_size_avg),
        step=da_state.step,
        avg_error=da_state.avg_error,
        mu=_clip_log_value(da_state.mu),
    )


def _stack_chain_states(states: list[TrajectoryMCMCState]) -> TrajectoryMCMCState:
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=0), *states)


def _stack_sample_history(
    history: list[jnp.ndarray],
    *,
    num_chains: int,
    trailing_shape: tuple[int, ...],
    dtype,
) -> jnp.ndarray:
    if not history:
        return jnp.zeros((num_chains, 0, *trailing_shape), dtype=dtype)
    stacked = jnp.stack(history, axis=0)
    return jnp.swapaxes(stacked, 0, 1)
