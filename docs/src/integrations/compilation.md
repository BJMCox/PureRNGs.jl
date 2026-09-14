# Differentiation and compilation

Enzyme differentiates Julia code. Reactant compiles array programs.
Their RNG integrations serve different purposes.

## Enzyme

Loading Enzyme activates rules for primitive floating-point uniform, normal, and exponential fills, including continuation fills.

Treat the immutable generator as constant.
An overwriting random fill clears the destination's incoming shadow.
It does not differentiate the seed or repeat the primal draw during the reverse pass.

These rules cover CPU and CUDA use. They do not promise general differentiation through sampling, distribution objects, or mutable `StatefulRNG` effects.

Use `Enzyme.Const(rng)` for the generator when differentiating a supported fill.
Differentiate the computation consuming the samples, not the random key.

```julia
using PureRNGs, Random, Enzyme

function sample_sum!(rng, destination, scale)
    randn!(rng, destination; threaded=false)
    return scale * sum(destination)
end

rng = Philox4x32(123456)
values = zeros(16)
shadow = ones(16)
derivatives = only(autodiff(
    Reverse, sample_sum!, Active,
    Const(rng), Duplicated(values, shadow), Active(2.0),
))
@assert derivatives[3] ≈ sum(values)
@assert all(iszero, shadow)
```

## Reactant: pass state as runtime data

Reactant support requires version 0.2.280 or later in the 0.2 series.
Convert the generator once before compilation:

```julia
using PureRNGs, Reactant

Reactant.set_default_backend("cpu")

function step(rng)
    return rand_next(rng, Float64)
end

rng = Reactant.to_rarray(Philox4x32(123456))
compiled_step = Reactant.@compile sync=true step(rng)

value, rng = compiled_step(rng)
value, rng = compiled_step(rng)

other = Reactant.to_rarray(Philox4x32(654321))
value, other = compiled_step(other)
```

The converted carrier holds runtime key and position data.
The same executable accepts changed keys and positions with matching generator type, backend, shape, and sharding.

An ordinary unconverted immutable struct can become a compile-time constant.
Do not use that path for frequently changing RNG state.

Select the Reactant backend before conversion. Use `"gpu"` for a supported GPU environment.
This is not an MLDataDevices `ReactantDevice()` binding.

On a GPU, Reactant reserves three quarters of the device memory for its process by default.
Compiled draws need a few hundred megabytes. Set `XLA_REACTANT_GPU_PREALLOCATE=false` before loading Reactant when other processes share the device.

## Compiled scope and limits

Carriers support scalar primitive, range, and supported distribution draws, scalar continuations, addressed draws, static key derivation, and array draws with static sizes: uniform, normal, exponential, integer range, and unweighted `randsample`.
Destination fills into traced arrays replace the destination's value.

Scalar, continuation, and addressed Reactant forms also cover `LogNormal`,
`Weibull`, `Rayleigh`, and `Laplace`, using native compiled math. Categorical is
not a carrier operation. These additions do not extend differentiation support.

Weighted `randsample` is not a carrier API.

A carrier holds no output block, so each chained scalar draw evaluates the core again and compiles to its own kernels.
For more than a few draws in one compiled function, draw an array once and index it.
Split counts using `Val` and purpose IDs must be static.
Changing static inputs can require recompilation.

!!! warning "Capacity is the caller's responsibility"
    Compiled carriers omit exhaustion checks. Keep every compiled draw within the generator's per-key capacity.
    Ordinary eager generators retain their checks.

Primitive integer and uniform streams retain their exact value contract.
Compiled normal and exponential transforms may differ from eager GPU values without an eager-relative ULP bound.

Compiled distribution arithmetic can also differ from eager mappings.
Repeated execution of the same executable with the same input state remains reproducible.

Do not use cross-backend equality of transformed values as a substitute for comparing the underlying stream and state.
