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

The wrapper supports scalar `Bool`, `UInt32`, `UInt64`, `Float32`, and
`Float64` draws, `Float32` and `Float64` normal draws, and integer ranges. It
also supports package-owned array and bit-array fills.

```julia
bridge = StatefulRNG(Threefry4x32(9))
uniforms = Vector{Float32}(undef, 256)
normals = Vector{Float64}(undef, 256)

rand!(bridge, uniforms)
randn!(bridge, normals)

@assert all(0 <= value < 1 for value in uniforms)
@assert length(normals) == 256
```

Copy a wrapper to replay from its current position. Reseed it to replace its
held RNG with a fresh CPU RNG of the same family.

```julia
bridge = StatefulRNG(Philox4x32(77))
replay = copy(bridge)
@assert rand(bridge, UInt64) == rand(replay, UInt64)

Random.seed!(bridge, 88)
@assert rand(bridge, UInt32) == rand(StatefulRNG(Philox4x32(88)), UInt32)
```

Construction always rebinds the source RNG to CPU. `StatefulRNG` is a host-only
bridge, even when its source uses a GPU backend. Keep device workflows
immutable and use `rand_next`, `rand_next!`, `randn_next`, or `randn_next!`
directly.
