# Particle Operators

`ParticleOperators` adapts a discrete-time `DynamicalModel` to the sampling and
scoring operations used by particle-inference algorithms. It leaves algorithm
choices such as particle count, resampling, conditioning, and diagnostics with
the consuming backend.

## Usage

```python
import jax.random as jr

import dynestyx as dsx
from dynestyx.discretizers import EulerMaruyamaConfig

discrete_dynamics = dsx.discretize_dynamics(
    continuous_dynamics,
    EulerMaruyamaConfig(),
)
operators = dsx.build_particle_operators(discrete_dynamics)
initial_state = operators.initial_sample(jr.key(0))
initial_log_prob = operators.initial_log_prob(initial_state)
```

The transition operations expose both sampling and density evaluation. Pairwise
transition scoring is suitable for algorithms such as backward sampling that
compare every previous particle with every current particle. Path scoring uses
the same missing-observation and batch semantics as `dsx.log_prob`.

Because `pairwise_transition_log_prob` returns every previous/current pair, its
output storage and default nested-vectorization work are
`O(n_previous * n_current)`. Consumers should account for that quadratic cost
when choosing particle counts.

Score methods preserve any batch axes returned by the underlying NumPyro
distribution. An algorithm that requires a scalar log weight must choose how to
reduce those axes; the built-in Cuthbert adapter sums them.

`build_particle_operators` accepts a discrete-time model. Use
`discretize_dynamics` first when starting from continuous dynamics. Keeping
these steps separate makes the numerical approximation explicit and lets the
same discrete model serve inference algorithms other than particle methods.

The selected transition determines which particle operations are available.
Sampling requires only a transition result with a `sample` method, including
Dynestyx's existing black-box transition objects. Density operations
additionally require `log_prob`. For example, `EulerMaruyamaConfig` supplies a
Gaussian interval transition with both capabilities, whereas
`DiffraxSampleConfig` supplies sampling without a transition density.

::: dynestyx.inference.particle_operators
    options:
      members:
        - ParticleOperators
        - build_particle_operators
