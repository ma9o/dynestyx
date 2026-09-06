# Joint parameter and trajectory inference

`ParticleRuntime` composes a **discrete** Dynestyx model with a NumPyro prior program,
observations, and an aligned time schedule. Its model factory runs inside the JAX
trace: changing parameter values changes the distributions without changing model
topology. NumPyro supplies constraints, conditional-prior replay, factors, and
Jacobian terms through `prepare_parameterization`.

```python
from dynestyx.inference.particle_runtime import (
    ParticleRuntime, ParticleSchedule, prepare_parameterization,
)
from cuthbert.mcmc.conditional import build_conditional_smc, build_conditional_dsmc
from cuthbert.mcmc.marginal_particle_gibbs import build_marginal_particle_gibbs_kernel
from dynestyx.inference.particle_mcmc import run_marginal_particle_gibbs
```

The consumer supplies `context(position, times)`, an array PyTree, and
`model(context)` and `schedule(context)` factories. The schedule contains the
state/observation times, observation controls, and transition controls. Transition
controls index the **destination**: entry `i` is held over `[times[i-1], times[i]]`.
The initial law is at `times[0]`; the runtime introduces no transition before it.
Observations have the same time axis as the states; insert missing rows when an
inference lattice contains unobserved states.

The model factory chooses numerical interpretation explicitly. The runtime does
not select a backend or linearize a model. `discretize_dynamics(model)` uses
Dynestyx's existing default; an explicit config chooses another supported
interpretation; a consumer-defined discrete state evolution supplies its own
transition distribution. Sampling and scoring use that same distribution.

`build_conditional_smc(target, position)` and
`build_conditional_dsmc(target, position)` return the same full-pass
`step(key, reference_path)` interface. The former is bootstrap conditional SMC;
the latter is conditional parallel-in-time dSMC. The dSMC kernel requires
transition densities and derivatives. Sample-only Diffrax transitions support
bootstrap conditional SMC, but are rejected when constructing the density-based
kernel. These operations do not supply an analytic endpoint density for a
multi-step numerical SDE solver.

The joint kernel implements the parameter-ensemble construction in
[Corenflos (2025)](https://arxiv.org/abs/2505.04611). Its gradient oracle uses a
**fixed pilot path**, so the oracle is a function of the parameter alone. The
pilot may be supplied by the consumer; its approximation changes efficiency,
while the label correction defines the intended target. It must not be silently
replaced by the evolving trajectory in the chain.

`run_marginal_particle_gibbs` supplies the outer loop, warmup-only adaptation,
chain summaries, optional trajectory retention, and exact observation factors on
joint draws. Those factors can support PSIS leave-one-observation-row-out
interpolation. They are not a claim of leave-future-out forecasting accuracy.

This runtime currently handles dense continuous state vectors and point
observations on one shared schedule. General state PyTrees, interval observations,
ragged batches, and automatic model-class dispatch are outside this interface.
The existing `LatentPathBuilder` and whole-path scoring remain available; the
runtime reuses NumPyro transformations and the model's distribution operations.

::: dynestyx.inference.particle_runtime
