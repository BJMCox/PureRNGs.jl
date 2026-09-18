# Docstrings for the `Random` names this package extends. They are bound to
# signatures rather than written above the definitions, because the methods come
# from `@eval` loops over the supported element types.

@doc """
    rand(rng::AbstractPureRNG, T) -> value
    rand(rng::AbstractPureRNG, T, dims...) -> Array{T}
    rand(rng::AbstractPureRNG, range) -> value
    rand(rng::AbstractPureRNG, range, dims...) -> Array

Draw uniform values at the held position of `rng` without advancing it. Two
calls on the same generator return the same values. Use [`rand_next`](@ref) to
continue the stream.

`T` is required: `rand(rng)` throws. Supported result types are `Bool`,
`UInt32`, `Int32`, `UInt64`, `Int64`, `Float32`, and `Float64`. Integer draws
cover the whole type and floating-point draws lie in `[0, 1)`. A `range`
argument draws from an integer range with signed or unsigned element type
through 64 bits. The array forms allocate on the generator's device.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> rand(rng, UInt32)
0x23b42aea

julia> rand(rng, UInt32)
0x23b42aea

julia> value, next_rng = rand_next(rng, UInt32);

julia> rand(next_rng, UInt32)
0x467098dd
```
""" Random.rand(::AbstractPureRNG, ::Any...)

@doc """
    rand!(rng::AbstractPureRNG, dest; threaded = true) -> dest
    rand!(rng::AbstractPureRNG, dest, range; threaded = true) -> dest

Fill `dest` with uniform values from the held position of `rng` without
advancing it. Two calls on the same generator write the same values. Use
[`rand_next!`](@ref) to continue the stream.

The destination element type must be `Bool`, `UInt32`, `Int32`, `UInt64`,
`Int64`, `Float32`, or `Float64`, or with `range` the integer element type of
that range. The destination device must match the generator device.

`threaded = false` requests the serial CPU fill path. The keyword picks how the
work is scheduled and never changes the values written.
""" Random.rand!(::AbstractPureRNG, ::AbstractArray, ::Any...)

@doc """
    randn(rng::AbstractPureRNG, T) -> value
    randn(rng::AbstractPureRNG, T, dims...) -> Array{T}

Draw standard normal values at the held position of `rng` without advancing it.
Two calls on the same generator return the same values. Use
[`randn_next`](@ref) to continue the stream.

`T` is required and is `Float32` or `Float64`: `randn(rng)` throws. The array
forms allocate on the generator's device.
""" Random.randn(::AbstractPureRNG, ::Any...)

@doc """
    randn!(rng::AbstractPureRNG, dest; threaded = true) -> dest

Fill a `Float32` or `Float64` destination with standard normal values from the
held position of `rng` without advancing it. Two calls on the same generator
write the same values. Use [`randn_next!`](@ref) to continue the stream.

The destination device must match the generator device. `threaded = false`
requests the serial CPU fill path and never changes the values written.
""" Random.randn!(::AbstractPureRNG, ::AbstractArray, ::Any...)

@doc """
    randexp(rng::AbstractPureRNG, T) -> value
    randexp(rng::AbstractPureRNG, T, dims...) -> Array{T}

Draw standard exponential values at the held position of `rng` without
advancing it. Two calls on the same generator return the same values. Use
[`randexp_next`](@ref) to continue the stream.

`T` is required and is `Float32` or `Float64`: `randexp(rng)` throws. The array
forms allocate on the generator's device.
""" Random.randexp(::AbstractPureRNG, ::Any...)

@doc """
    randexp!(rng::AbstractPureRNG, dest; threaded = true) -> dest

Fill a `Float32` or `Float64` destination with standard exponential values from
the held position of `rng` without advancing it. Two calls on the same
generator write the same values. Use [`randexp_next!`](@ref) to continue the
stream.

The destination device must match the generator device. `threaded = false`
requests the serial CPU fill path and never changes the values written.
""" Random.randexp!(::AbstractPureRNG, ::AbstractArray, ::Any...)
