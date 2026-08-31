"""Configuration-driven discretization of continuous-time models."""

from typing import Any

from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Real

from dynestyx.discretization.diffrax_sample import _DiffraxSampleStateEvolution
from dynestyx.discretization.exact_affine import _ExactAffineStateEvolution
from dynestyx.discretization.gaussian import (
    _ConfiguredGaussianStateEvolution,
    _local_linearization_parameters,
)
from dynestyx.discretization.ode_flow import _ODEFlowStateEvolution
from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.configs.discretizer import (
    BaseDiscretizerConfig,
    DiffraxSampleConfig,
    DiscretizerConfig,
    EulerMaruyamaConfig,
    ExactAffineConfig,
    LocalLinearizationConfig,
    MeanTrajectoryLinearizationConfig,
    ODEFlowConfig,
)
from dynestyx.models import (
    AffineDrift,
    DeterministicContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
    DynamicalModel,
    LinearGaussianParams,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.models.core import StateEvolutionLike


def _automatic_discretizer_config(
    cte: DeterministicContinuousTimeStateEvolution
    | StochasticContinuousTimeStateEvolution,
) -> DiscretizerConfig:
    if isinstance(cte, DeterministicContinuousTimeStateEvolution):
        return ODEFlowConfig()
    if (
        isinstance(cte.drift, AffineDrift)
        and cte.potential is None
        and not callable(cte.diffusion.coefficient)
    ):
        return ExactAffineConfig()
    return EulerMaruyamaConfig()


def _discretize_state_evolution(
    cte: StateEvolutionLike,
    config: BaseDiscretizerConfig | None = None,
) -> DiscreteTimeStateEvolution:
    """Build the private discrete transition selected by a config."""
    if not isinstance(
        cte,
        (
            DeterministicContinuousTimeStateEvolution,
            StochasticContinuousTimeStateEvolution,
        ),
    ):
        raise TypeError(
            "Discretizer configs require a continuous-time state "
            f"evolution; got {type(cte).__name__}."
        )
    resolved = _automatic_discretizer_config(cte) if config is None else config
    if isinstance(cte, DeterministicContinuousTimeStateEvolution):
        if isinstance(resolved, ODEFlowConfig):
            return _ODEFlowStateEvolution(cte, resolved)
        raise TypeError(
            f"{type(resolved).__name__} requires a stochastic continuous-time "
            "state evolution; got DeterministicContinuousTimeStateEvolution."
        )
    if isinstance(resolved, ODEFlowConfig):
        raise TypeError(
            "ODEFlowConfig requires a deterministic continuous-time state "
            "evolution; got StochasticContinuousTimeStateEvolution."
        )
    if isinstance(resolved, ExactAffineConfig):
        return _ExactAffineStateEvolution(
            cte,
            covariance_jitter=resolved.covariance_jitter,
        )
    if isinstance(resolved, DiffraxSampleConfig):
        return _DiffraxSampleStateEvolution(cte, resolved)
    if isinstance(
        resolved,
        (
            EulerMaruyamaConfig,
            LocalLinearizationConfig,
            MeanTrajectoryLinearizationConfig,
        ),
    ):
        if isinstance(resolved, LocalLinearizationConfig) and callable(
            cte.diffusion.coefficient
        ):
            raise TypeError(
                "LocalLinearizationConfig requires structurally constant "
                "additive diffusion."
            )
        return _ConfiguredGaussianStateEvolution(cte, resolved)
    raise TypeError(
        "discretizer_config must be a concrete BaseDiscretizerConfig; "
        f"got {type(resolved).__name__}."
    )


def discretize_dynamics(
    dynamics: DynamicalModel,
    discretizer_config: BaseDiscretizerConfig | None = None,
) -> DynamicalModel:
    """Build a discrete-time model from continuous-time dynamics.

    This is the pure model-level counterpart to the `Discretizer` effect
    handler. It preserves the initial condition, observation model, control
    metadata, and declared initial time while replacing the continuous state
    evolution with the transition selected by `discretizer_config`.

    When no config is provided, deterministic ODEs use their numerical flow,
    affine SDEs with constant diffusion and no potential use an exact Gaussian
    transition, and other SDEs use Euler--Maruyama.

    Args:
        dynamics: Continuous-time model to discretize.
        discretizer_config: Explicit discretization config, or `None` for
            automatic routing.

    Returns:
        DynamicalModel: A discrete-time model with the selected interval
            transition.

    Raises:
        TypeError: If `dynamics` is already discrete-time or the config is not
            compatible with its continuous state evolution.
    """
    if not dynamics.continuous_time:
        raise TypeError(
            "discretize_dynamics requires a continuous-time DynamicalModel; "
            "got a discrete-time model."
        )
    return DynamicalModel(
        initial_condition=dynamics.initial_condition,
        state_evolution=_discretize_state_evolution(
            dynamics.state_evolution,
            discretizer_config,
        ),
        observation_model=dynamics.observation_model,
        control_model=dynamics.control_model,
        control_dim=dynamics.control_dim,
        t0=dynamics.t0,
    )


def linearized_transition_parameters(
    dynamics: DynamicalModel,
    discretizer_config: LocalLinearizationConfig,
    *,
    linearization_state: Real[Array, " state_dim"] | Real[Array, ""],
    previous_control: Real[Array, " control_dim"] | Real[Array, ""] | None,
    previous_time: float | int | Real[Array, ""],
    time: float | int | Real[Array, ""],
) -> LinearGaussianParams:
    """Discretize one local affine approximation of a nonlinear SDE.

    The continuous drift is linearized with respect to state at
    `linearization_state`, `previous_control`, and `previous_time`. The control
    is held fixed over the interval and is therefore absorbed into `bias`; the
    returned `B` is `None`. The state matrix, bias, and additive diffusion are
    then discretized over `[previous_time, time]` with the same Van Loan method
    used by `LocalLinearizationConfig`.

    Choosing a sequence of linearization states and running a Gaussian
    inference algorithm remain consumer responsibilities. This function only
    supplies transition-side `LinearGaussianParams`; observation linearization
    and Gaussian recursion remain with the consumer.

    Args:
        dynamics: Continuous-time stochastic Dynestyx model to interpret.
        discretizer_config: Local-linearization numerical configuration.
        linearization_state: State about which to linearize the drift.
        previous_control: Control held fixed over the interval, or `None` for
            an uncontrolled model.
        previous_time: Left endpoint of the transition interval.
        time: Right endpoint of the transition interval.

    Returns:
        LinearGaussianParams: Local `(A, B, bias, cov)` parameters, with
            `B=None` because the supplied control is frozen into `bias`.

    Raises:
        TypeError: If the model is not a stochastic continuous-time model, the
            config is not `LocalLinearizationConfig`, or the diffusion is not
            structurally constant and additive.
    """
    if not isinstance(discretizer_config, LocalLinearizationConfig):
        raise TypeError(
            "linearized_transition_parameters requires "
            "LocalLinearizationConfig; "
            f"got {type(discretizer_config).__name__}."
        )
    cte = dynamics.state_evolution
    if not isinstance(cte, StochasticContinuousTimeStateEvolution):
        raise TypeError(
            "linearized_transition_parameters requires a stochastic "
            "continuous-time DynamicalModel."
        )
    if callable(cte.diffusion.coefficient):
        raise TypeError(
            "LocalLinearizationConfig requires structurally constant "
            "additive diffusion."
        )
    return _local_linearization_parameters(
        cte,
        linearization_state,
        previous_control,
        previous_time,
        time,
        covariance_jitter=discretizer_config.covariance_jitter,
    )


class Discretizer(ObjectInterpretation, HandlesSelf):
    r"""Performs discretization of a continuous-time state evolution, converting it to a discrete-time state evolution.

    A `Discretizer` interpretation should be used inside an inference or simulation context. The outside inference/simulation
    context may then use the resulting `DiscreteTimeStateEvolution`:

    ```python
    from dynestyx.discretizers import (
        Discretizer,
        EnKFConfig,
        Filter,
        MeanTrajectoryLinearizationConfig,
    )
    with Filter(EnKFConfig()):
        with Discretizer(MeanTrajectoryLinearizationConfig()):
            model(...)
    ```

    When no config is provided, ODE models use their numerical flow, SDE models
    with an affine drift use an exact Gaussian discretization, and other SDE
    models use Euler--Maruyama.

    Attributes:
        discretizer_config: Explicit discretization config, or `None` for
            automatic routing.
    """

    def __init__(
        self,
        discretizer_config: BaseDiscretizerConfig | None = None,
    ):
        super().__init__()
        if discretizer_config is not None and not isinstance(
            discretizer_config, BaseDiscretizerConfig
        ):
            raise TypeError(
                "discretizer_config must be a BaseDiscretizerConfig or None, "
                f"got {type(discretizer_config).__name__}."
            )
        self.discretizer_config = discretizer_config

    @implements(_condition_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> Any:
        if isinstance(
            dynamics.state_evolution,
            (
                DeterministicContinuousTimeStateEvolution,
                StochasticContinuousTimeStateEvolution,
            ),
        ):
            dynamics = discretize_dynamics(
                dynamics,
                self.discretizer_config,
            )
        return fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )


__all__ = [
    "BaseDiscretizerConfig",
    "DiffraxSampleConfig",
    "Discretizer",
    "DiscretizerConfig",
    "EulerMaruyamaConfig",
    "ExactAffineConfig",
    "LocalLinearizationConfig",
    "MeanTrajectoryLinearizationConfig",
    "ODEFlowConfig",
    "discretize_dynamics",
    "linearized_transition_parameters",
]
