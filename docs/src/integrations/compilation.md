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
These forms trace but are outside the release conformance gate.

Scalar, continuation, and addressed Reactant forms cover every fixed
distribution listed in [Distributions](@ref), using native compiled math.
`Categorical` is not a carrier operation. These additions do not extend
differentiation support.

Weighted `randsample` is not a carrier API.

A carrier holds no output block, so each chained scalar draw evaluates the core again and compiles to its own kernels.
For more than a few draws in one compiled function, draw an array once and index it.
Split counts using `Val` and purpose IDs must be static.
Changing static inputs can require recompilation.

!!! warning "Capacity is the caller's responsibility"
    Compiled carriers omit exhaustion checks. Keep every compiled draw within the generator's per-key capacity.
    Ordinary eager generators retain their checks.
    On the two-word 32-bit generators the counter wraps, so a compiled draw past capacity repeats the start of the stream.

Primitive integer and uniform streams retain their exact value contract.
Compiled normal and exponential transforms may differ from eager GPU values without an eager-relative ULP bound.

Compiled distribution arithmetic can also differ from eager mappings.
Repeated execution of the same executable with the same input state remains reproducible.

Do not use cross-backend equality of transformed values as a substitute for comparing the underlying stream and state.

## TandemRNG

Loading TandemRNG enables its optional PureRNGs bridge:

```julia
import PureRNGs, TandemRNG

rng = TandemRNG.Tandem8x32(42)
values, rng = PureRNGs.rand_next(rng, Float64, 17, 3)
children = PureRNGs.splitrng(rng, Val(3))
stateful = PureRNGs.StatefulRNG(rng)
```

The bridge supports uniform scalar, addressed, allocating, and destination draws,
`rngkey`, `rngposition`, `splitrng`, and `subrng`. Its values and advancement follow
Tandem's natural alignment law. Tandem remains its own generator type.
It follows the same MLDataDevices residence rules as the built-in generators:
bind with `CUDADevice()` before GPU allocation, preserve binding through draws and
derivations, and reject a destination on another backend even when empty.
Backend tokens do not pin a physical GPU. Unsupported operations do not fall back to CPU.
`StatefulRNG(rng)` returns `TandemRNG.Stateful` for Random consumers;
`parent(stateful)` returns its current immutable generator.

With Reactant loaded, convert through `Reactant.to_rarray(rng)` before compilation.
Uniform draws, fills, static `splitrng(..., Val(N))`, and static `subrng` retain the same
contract. Compiled carriers omit exhaustion checks. Keep each call within Tandem's
2^64-bit stream, with the final position below 2^64.
Distribution and continuation methods are outside the Tandem bridge's scope.

The bridge precompile workload covers K1/K32/K64, all uniform draw types, vector and
matrix fills, addressed draws, derivations, and the stateful constructor.
Tandem's Reactant extension caches Julia tracing methods; the first compiled call
still builds a backend executable.
