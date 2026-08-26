# Immutable workflows

PureRNGs makes stream state explicit. Each continuing draw returns the
next RNG first and the generated value second.

```julia
using PureRNGs
using Random

root = Philox4x32(1234)
rng, uniform = rand_next(root, Float64)
rng, integers = rand_next(rng, UInt32, 4)
rng, normal = randn_next(rng, Float32)

@assert root == Philox4x32(1234)
@assert rand(root, Float64) == uniform
@assert length(integers) == 4
@assert normal isa Float32
```

Use `rand_next` and `randn_next` when later work must continue the stream. Use
`rand` and `randn` when only the value matters. A plain draw does not change the
immutable RNG, so calling it again with the same arguments replays the value.

## Fill an existing array

The continuing in-place methods return the next RNG and the same destination.

```julia
root = Philox4x32(2026)
buffer = Vector{Float32}(undef, 1024)
rng, returned = rand_next!(root, buffer)

@assert returned === buffer
@assert buffer == last(rand_next(root, Float32, length(buffer)))
```

The default path selects an optimized fill automatically. The optional
`threaded=false` keyword forces the serial CPU fill when measuring or debugging
a specific path. Normal use does not need this knob.

## Address independent work

`randat` and `randnat` use one-based logical draw indices. They do not change
the RNG and do not depend on call order.

```julia
root = Threefry4x32(7)
addressed = [randat(root, UInt64, index) for index in 1:8]
cursor, sequential = rand_next(root, UInt64, 8)

@assert addressed == sequential
normal_rng, _ = randn_next(root, Float64)
_, second_normal = randn_next(normal_rng, Float64)
@assert randnat(root, Float64, 2) == second_normal
```

Use continuation for a sequential algorithm. Use addressed draws for work that
may run in another order. Derive separate role or chunk keys before independent
jobs. Read [Splitting keys](../guides/splitting.md) for that contract.

Read [Reproducibility](../guides/reproducibility.md) before relying on bitwise
normal results across architectures.
