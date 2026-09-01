# Stateful interoperability

Use `StatefulRNG` only when an API requires `Random.AbstractRNG`. It wraps one
immutable RNG in mutable host state and advances that state after each draw.

```julia
using PureRNGs
using Random

function legacy_simulation(rng::Random.AbstractRNG)
    return (rand(rng), randn(rng), rand(rng, 1:6))
end

root = Philox4x32(123)
bridge = StatefulRNG(root)
result = legacy_simulation(bridge)

@assert result isa Tuple{Float64, Float64, Int}
@assert root == Philox4x32(123)
```

The wrapper supports scalar `Bool`, `UInt32`, `Int32`, `UInt64`, `Int64`,
`Float32`, and `Float64` draws, integer ranges, and `Float32` and `Float64`
normal and exponential draws. Untyped uniform, normal, and exponential calls
use Julia's `Float64` default.

```julia
bridge = StatefulRNG(Philox4x32(5))
signed32 = rand(bridge, Int32)
signed64 = rand(bridge, Int64)
exponential32 = randexp(bridge, Float32)
exponentials64 = randexp(bridge, Float64, 128)

@assert signed32 isa Int32
@assert signed64 isa Int64
@assert exponential32 isa Float32
@assert exponentials64 isa Vector{Float64}
```

## Fill host arrays

The package owns `rand!`, `randn!`, and `randexp!` for matching concrete
`Array` destinations. It also owns `rand!` for `BitArray` and integer-range
`Array` fills.

```julia
bridge = StatefulRNG(Threefry4x32(9))
signed = Vector{Int64}(undef, 256)
normals = Vector{Float64}(undef, 256)
exponentials = Vector{Float32}(undef, 256)

rand!(bridge, signed)
randn!(bridge, normals)
randexp!(bridge, exponentials)

@assert eltype(signed) === Int64
@assert length(normals) == 256
@assert all(value -> value >= 0, exponentials)
```

Each one-argument owned fill preflights its full reservation before mutation
and equals chained scalar continuation draws. Counter exhaustion therefore
leaves its destination and the bridge unchanged.

The integer-range form `rand!(bridge, destination, range)` consumes chained
scalar range draws. Counter exhaustion writes only the maximal valid prefix
and leaves the bridge after the last successful draw.

Other destination types use foreign `Random` fill methods. Those methods may
use another scalar consumption order. If exhaustion occurs mid-fill, the
destination may be partly written and the bridge stays valid at the position
after the last successful scalar draw.

## Inspect and replay state

`parent(bridge)` returns the exact CPU-bound immutable generator currently held
by the bridge. Use it to checkpoint or resume with the immutable API.

```julia
bridge = StatefulRNG(Philox4x32(77))
checkpoint = parent(bridge)
next_checkpoint, expected = randexp_next(checkpoint, Float32)

@assert randexp(bridge, Float32) == expected
@assert parent(bridge) == next_checkpoint

replay = StatefulRNG(checkpoint)
@assert randexp(replay, Float32) == expected
```

`copy(bridge)` returns an independent wrapper at the same position. Reseeding
replaces the held generator with a fresh zero-position CPU generator of the
same family.

```julia
bridge = StatefulRNG(Philox4x32(77))
replay = copy(bridge)
@assert rand(bridge, Int64) == rand(replay, Int64)

Random.seed!(bridge, 88)
@assert rand(bridge, UInt32) == rand(StatefulRNG(Philox4x32(88)), UInt32)
```

Construction always rebinds the source RNG to CPU. `StatefulRNG` is a host-only
bridge, even when its source uses a GPU backend. It preserves the key and
position, and every bridge result stays on the host. Keep device workflows
immutable and use `rand_next`, `rand_next!`, `randn_next`, `randn_next!`,
`randexp_next`, or `randexp_next!` directly.
