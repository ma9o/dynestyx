# Discretization

A continuous-time `DynamicalModel` can be converted into a discrete-time model
without entering an effect-handler context:

```python
import dynestyx as dsx
from dynestyx.discretizers import EulerMaruyamaConfig

discrete_dynamics = dsx.discretize_dynamics(
    continuous_dynamics,
    EulerMaruyamaConfig(),
)
```

The returned model preserves the initial condition, observation model, control
metadata, and initial time. Its state evolution is the interval transition
selected by the discretizer configuration. This pure form is suitable for
building reusable algorithm-facing objects such as
`dsx.build_particle_operators(discrete_dynamics)`.

## Effect-handler form

A `Discretizer` maps a `ContinuousTimeStateEvolution` to a `DiscreteTimeStateEvolution` by discretizing the corresponding ODE or SDE; the resulting model is compatible with discrete-time inference techniques in `dynestyx` when the selected transition interface supplies what the inference method requires. The discretizer context should be placed *inside* the corresponding inference context:

```python
import dynestyx as dsx
from dynestyx.discretizers import (
    Discretizer,
    MeanTrajectoryLinearizationConfig,
)
from dynestyx.inference.filters import EnKFConfig, Filter

with Filter(EnKFConfig(n_particles=100)):
    with Discretizer(MeanTrajectoryLinearizationConfig()):
        result = model(obs_times=obs_times, obs_values=obs_values)
```

The config (in the above, `MeanTrajectoryLinearizationConfig`) changes the corresponding method for discretizing the continuous-time dynamics. See [Discretizer configurations](../inference/configs/discretizer_configs.md) for more information about each.

## Automatic routing

When no configuration is supplied, `Discretizer()` chooses automatically:

- a deterministic ODE is integrated with `ODEFlowConfig()`, producing a Delta transition at the numerical flow endpoint;
- an `AffineDrift` with constant diffusion and no potential is discretized exactly; and
- other SDE models use Euler--Maruyama discretization by default.

Pass `ODEFlowConfig(simulator_config=ODESimulatorConfig(...), jitter_scale=...)` to customize ODE integration; all Diffrax settings are taken from the nested `ODESimulatorConfig`.

Both `discretize_dynamics()` and `Discretizer()` use this routing. The handler
delegates its model conversion to the same pure function.

## Local affine-Gaussian parameters

Some inference algorithms need the affine parameters of the approximation,
rather than only the Gaussian transition distribution produced by
`LocalLinearizationConfig`. Those parameters can be requested for one interval
without selecting a filter or smoother:

```python
import dynestyx as dsx
from dynestyx.discretizers import LocalLinearizationConfig

params = dsx.linearized_transition_parameters(
    continuous_dynamics,
    LocalLinearizationConfig(covariance_jitter=1e-8),
    linearization_state=reference_state,
    previous_control=control,
    previous_time=t_now,
    time=t_next,
)

A = params.A
bias = params.bias
Q = params.cov
```

The result is the existing `LinearGaussianParams` value. It describes the
same one-interval approximation used by `LocalLinearizationConfig`: the drift
is linearized at `linearization_state`, while time and control are frozen at
the left endpoint. Consequently, the supplied control is included in `bias`
and `params.B` is `None`.

The function is JAX-transformable, so a consumer can `vmap` it along a
reference trajectory. Choosing that trajectory, iterating an IEKS, or running
a Kalman filter remains the responsibility of the inference library. These are
transition-side inputs only: observation linearization and Gaussian recursion
remain consumer responsibilities. For a globally affine model,
`ExactAffineConfig` instead produces a
`LinearGaussianStateEvolution` whose `params_at(t_now, t_next)` method returns
exact interval parameters without a linearization state.

::: dynestyx.discretizers
    options:
      members:
        - discretize_dynamics
        - linearized_transition_parameters
        - Discretizer
      show_root_heading: false
      show_root_toc_entry: false
