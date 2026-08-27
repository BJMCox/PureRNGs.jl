# PureRNGs.jl

PureRNGs.jl provides counter-based random generators whose state is an
immutable value. A draw can return the advanced generator with its result, so
random state stays explicit and easy to split across independent work.

```jldoctest
julia> using PureRNGs

julia> rng = Philox4x32(123);

julia> next_rng, value = rand_next(rng, Float64);

julia> 0.0 <= value < 1.0
true

julia> rng == next_rng
false
```

Use `rand_next`, `randn_next`, and `randexp_next` when later draws must continue
from the returned state. Use `rand`, `randn`, and `randexp` when only the value
matters. Both forms leave the input generator unchanged.

Derive generators for independent roles with [`subrng`](@ref), or derive an
ordered group with [`splitrng`](@ref). Use [`StatefulRNG`](@ref) only when an API
requires `Random.AbstractRNG`.

Start with [Immutable workflows](tutorials/immutable-workflows.md), then read
[Splitting and devices](tutorials/splitting-and-devices.md) and
[Sampling](tutorials/sampling.md).

See the [API reference](@ref) for the exported interface.
