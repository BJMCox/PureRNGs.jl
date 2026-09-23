const _ScalarUniform32Generators =
    Union{Philox2x32,Philox4x32,Threefry2x32,Threefry4x32,ChaCha}
const _ScalarUniform64Generators = Union{Philox2x64,Philox4x64,Threefry2x64,Threefry4x64}
const _ScalarUniformGenerators =
    Union{_ScalarUniform32Generators,_ScalarUniform64Generators}
const _UniformInteger32 = Union{Int32,UInt32}
const _UniformInteger64 = Union{Int64,UInt64}
const _UniformInteger = Union{_UniformInteger32,_UniformInteger64}
const _UniformFloat = Union{Float32,Float64}
const _UniformResult = Union{Bool,_UniformInteger,_UniformFloat}

@inline _draw_bits(::Type{Bool}) = UInt16(1)
@inline _draw_bits(::Type{Float32}) = UInt16(24)
@inline _draw_bits(::Type{UInt32}) = UInt16(32)
@inline _draw_bits(::Type{Int32}) = UInt16(32)
@inline _draw_bits(::Type{Float64}) = UInt16(53)
@inline _draw_bits(::Type{UInt64}) = UInt16(64)
@inline _draw_bits(::Type{Int64}) = UInt16(64)

@inline _position_block(position::_Position64) = position.block
@inline _position_block(position::_Position128) = (position.lo, position.hi)

@inline function _draw_raw(rng::_ScalarUniformGenerators, position, ::Val{W}) where {W}
    return _extract_bits_unchecked(rng, _position_block(position), position.bit, Val(W))
end

@inline _draw_raw(rng::_ScalarUniformGenerators, width) =
    _draw_raw(rng, rng.position, width)

@inline _from_bits(::Type{Bool}, value::UInt64) = isone(value)
@inline _from_bits(::Type{UInt32}, value::UInt64) = value % UInt32
@inline _from_bits(::Type{Int32}, value::UInt64) = reinterpret(Int32, value % UInt32)
@inline _from_bits(::Type{UInt64}, value::UInt64) = value
@inline _from_bits(::Type{Int64}, value::UInt64) = reinterpret(Int64, value)
@inline _from_bits(::Type{Float32}, value::UInt64) =
    Float32(value % UInt32) * Float32(0x1p-24)
@inline _from_bits(::Type{Float64}, value::UInt64) = Float64(value) * 0x1p-53

@inline _draw_unchecked(
    rng::_ScalarUniformGenerators,
    ::Type{T},
) where {T<:_UniformResult} = _from_bits(T, _draw_raw(rng, Val(Int(_draw_bits(T)))))

@inline _draw_unchecked(
    rng::_ScalarUniformGenerators,
    position,
    ::Type{T},
) where {T<:_UniformResult} =
    _from_bits(T, _draw_raw(rng, position, Val(Int(_draw_bits(T)))))

@noinline function _untyped_draw_error(held_form::String, next_form::String)
    throw(
        ArgumentError(
            "untyped immutable draws are forbidden; use $held_form to draw at the " *
            "held position, or $next_form to advance",
        ),
    )
end

# Nine guard methods keep Base's untyped fallbacks unreachable. The dims
# spellings would otherwise reach `Random.Sampler` through the collection path.
Random.rand(::AbstractPureRNG) = _untyped_draw_error("rand(rng, T)", "rand_next(rng, T)")
Random.rand(::AbstractPureRNG, ::Integer, ::Integer...) =
    _untyped_draw_error("rand(rng, T, dims...)", "rand_next(rng, dims...)")
Random.rand(::AbstractPureRNG, ::Dims) =
    _untyped_draw_error("rand(rng, T, dims...)", "rand_next(rng, dims...)")

@inline function _rand_next_scalar(rng::_ScalarUniformGenerators, ::Type{T}) where {T}
    next_rng = _reserve_scalar(rng, _draw_bits(T))
    return _from_bits(T, _chain_bits(rng, next_rng, Val(Int(_draw_bits(T))))), next_rng
end

@inline _rand_scalar(rng::_ScalarUniformGenerators, ::Type{T}) where {T} =
    first(_rand_next_scalar(rng, T))

@inline rand_next(rng::_ScalarUniformGenerators) = rand_next(rng, Float64)

@inline Random.rand(rng::_ScalarUniformGenerators, ::Type{T}) where {T<:_UniformResult} =
    _rand_scalar(rng, T)
@inline rand_next(rng::_ScalarUniformGenerators, ::Type{T}) where {T<:_UniformResult} =
    _rand_next_scalar(rng, T)

@doc """
    rand_next(rng[, T]) -> (value, next_rng)
    rand_next(rng, range) -> (value, next_rng)
    rand_next(rng[, T], dims...) -> (values, next_rng)
    rand_next(rng, range, dims...) -> (values, next_rng)

Draw from `rng` and return the advanced immutable generator with the result.
Omitting `T` selects `Float64`. Supported scalar types are `Bool`, `UInt32`,
`Int32`, `UInt64`, `Int64`, `Float32`, and `Float64`. Integer ranges support
signed and unsigned integer element types through 64 bits. Dimensions may
also be one tuple, as in `Random`.

The allocating forms create an array on the generator's device. The input
generator never changes.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> value, next_rng = rand_next(rng, UInt32);

julia> value
0x23b42aea

julia> first(rand_next(next_rng, UInt32))
0x467098dd

julia> first(rand_next(rng, UInt32, 3))
3-element Vector{UInt32}:
 0x23b42aea
 0x467098dd
 0xc25ecc0b
```
""" rand_next
