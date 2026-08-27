# Immutable workflows

PureRNGs makes stream state explicit. Each continuing draw returns the
next RNG first and the generated value second.

```julia
using PureRNGs
using Random

root = Philox4x32(1234)
rng, uniform = rand_next(root, Float64)
rng, signed = rand_next(rng, Int32)
rng, normal = randn_next(rng, Float32)
rng, exponential = randexp_next(rng, Float64)

@assert root == Philox4x32(1234)
@assert rand(root, Float64) == uniform
@assert signed isa Int32
@assert normal isa Float32
@assert exponential isa Float64
```

Use `rand_next`, `randn_next`, and `randexp_next` when later work must continue
the stream. Use `rand`, `randn`, and `randexp` when only the value matters. A
plain draw does not change the immutable RNG, so calling it again with the same
arguments replays the value.

An `Int32` draw consumes 32 bits, and an `Int64` draw consumes 64 bits. Its
value reinterprets the bits of the matching unsigned draw. Exponential
`Float32` and `Float64` draws consume 24 and 53 bits. Mixed continuation calls
advance by exactly those widths without padding.

## Fill an existing array

The continuing in-place methods return the next RNG and the same destination.

```julia
root = Philox4x32(2026)
rng, values = randexp_next(root, Float32, 1024)
buffer = similar(values)
rng, returned = randexp_next!(rng, buffer)

@assert returned === buffer
@assert values isa Vector{Float32}
@assert all(value -> value >= 0, buffer)
```

The default path selects an optimized fill automatically. The optional
`threaded=false` keyword forces the serial CPU fill when measuring or debugging
a specific path. Normal use does not need this knob.

## Address independent work

`randat`, `randnat`, and `randexpat` use one-based logical draw indices. They do
not change the RNG and do not depend on call order.

```julia
root = Threefry4x32(7)
addressed = [randat(root, UInt64, index) for index in 1:8]
cursor, sequential = rand_next(root, UInt64, 8)

@assert addressed == sequential
normal_rng, _ = randn_next(root, Float64)
_, second_normal = randn_next(normal_rng, Float64)
@assert randnat(root, Float64, 2) == second_normal

exp_rng, _ = randexp_next(root, Float32)
_, second_exp = randexp_next(exp_rng, Float32)
@assert randexpat(root, Float32, 2) == second_exp
```

Use continuation for a sequential algorithm. Use addressed draws for work that
may run in another order. Derive separate role or chunk keys before independent
jobs. Read [Splitting keys](../guides/splitting.md) for that contract.

Read [Reproducibility](../guides/reproducibility.md) before relying on bitwise
normal results across architectures.
