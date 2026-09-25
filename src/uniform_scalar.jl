const _ScalarUniform32Generators =
    Union{Philox2x32,Philox4x32,Threefry2x32,Threefry4x32,ChaCha}
const _ScalarUniform64Generators = Union{Philox2x64,Philox4x64,Threefry2x64,Threefry4x64}
const _ScalarUniformGenerators =
    Union{_ScalarUniform32Generators,_ScalarUniform64Generators}
const _UniformInteger32 = Union{Int32,UInt32}
const _UniformInteger64 = Union{Int64,UInt64}
const _UniformInteger = Union{_UniformInteger32,_UniformInteger64}
const _NarrowInteger = Union{Int8,UInt8,Int16,UInt16}
const _WideInteger = Union{Int128,UInt128}
const _UniformFloat = Union{Float32,Float64}
# Floating-point types with normal and exponential draws. Float16 values are the
# Float32 transform on the same bits, rounded once.
const _TransformFloat = Union{Float16,_UniformFloat}
const _ComplexResult = Complex{<:_TransformFloat}
const _UniformResult =
    Union{Bool,_NarrowInteger,_UniformInteger,_WideInteger,_TransformFloat,_ComplexResult}

# Every draw reads its width in MSB-first stream bits: integers use their full
# width, and a floating-point value in [0, 1) uses its significand width plus one.
@inline _draw_bits(::Type{Bool}) = UInt16(1)
@inline _draw_bits(::Type{<:Union{Int8,UInt8}}) = UInt16(8)
@inline _draw_bits(::Type{<:Union{Int16,UInt16}}) = UInt16(16)
@inline _draw_bits(::Type{Float16}) = UInt16(11)
@inline _draw_bits(::Type{Float32}) = UInt16(24)
@inline _draw_bits(::Type{UInt32}) = UInt16(32)
@inline _draw_bits(::Type{Int32}) = UInt16(32)
@inline _draw_bits(::Type{Float64}) = UInt16(53)
@inline _draw_bits(::Type{UInt64}) = UInt16(64)
@inline _draw_bits(::Type{Int64}) = UInt16(64)
@inline _draw_bits(::Type{<:_WideInteger}) = UInt16(128)
# A complex value is its real draw followed by its imaginary draw.
@inline _draw_bits(::Type{Complex{T}}) where {T} = UInt16(2) * _draw_bits(T)

@inline _position_block(position::_Position64) = position.block
@inline _position_block(position::_Position128) = (position.lo, position.hi)

@inline function _draw_raw(rng::_ScalarUniformGenerators, position, ::Val{W}) where {W}
    return _extract_bits_unchecked(rng, _position_block(position), position.bit, Val(W))
end

@inline _draw_raw(rng::_ScalarUniformGenerators, width) =
    _draw_raw(rng, rng.position, width)

@inline _from_bits(::Type{Bool}, value::UInt64) = isone(value)
@inline _from_bits(::Type{UInt8}, value::UInt64) = value % UInt8
@inline _from_bits(::Type{Int8}, value::UInt64) = reinterpret(Int8, value % UInt8)
@inline _from_bits(::Type{UInt16}, value::UInt64) = value % UInt16
@inline _from_bits(::Type{Int16}, value::UInt64) = reinterpret(Int16, value % UInt16)
# Both steps are exact: an 11-bit integer and a power-of-two scale fit Float32.
@inline _from_bits(::Type{Float16}, value::UInt64) =
    Float16(Float32(value % UInt16) * Float32(0x1p-11))
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
) where {T<:_UniformResult} = _draw_unchecked(rng, rng.position, T)

@inline _draw_unchecked(
    rng::_ScalarUniformGenerators,
    position,
    ::Type{T},
) where {T<:_UniformResult} =
    _from_bits(T, _draw_raw(rng, position, Val(Int(_draw_bits(T)))))

@inline _wide_from_words(::Type{UInt128}, hi::UInt64, lo::UInt64) =
    (UInt128(hi) << 64) | UInt128(lo)
@inline _wide_from_words(::Type{Int128}, hi::UInt64, lo::UInt64) =
    reinterpret(Int128, _wide_from_words(UInt128, hi, lo))

# A 128-bit draw is the 64-bit draw at its position followed by the next one.
@inline function _draw_unchecked(
    rng::_ScalarUniformGenerators,
    position,
    ::Type{T},
) where {T<:_WideInteger}
    lo, hi = _extract_bits128_unchecked(rng, _position_block(position), position.bit)
    return _wide_from_words(T, hi, lo)
end

@inline function _draw_unchecked(
    rng::_ScalarUniformGenerators,
    position,
    ::Type{Complex{T}},
) where {T<:_TransformFloat}
    width = UInt64(_draw_bits(T))
    imaginary = _advance_position_unchecked(position, width, UInt64(0), _block_shift(rng))
    return Complex{T}(_draw_unchecked(rng, position, T), _draw_unchecked(rng, imaginary, T))
end

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
# The collection picks take tuples, so this guard names the same generator union
# they do and stays more specific for a tuple of `Int`, which is a shape.
Random.rand(::_ScalarUniformGenerators, ::Dims) =
    _untyped_draw_error("rand(rng, T, dims...)", "rand_next(rng, dims...)")

@inline function _rand_next_scalar(rng::_ScalarUniformGenerators, ::Type{T}) where {T}
    next_rng = _reserve_scalar(rng, _draw_bits(T))
    return _from_bits(T, _chain_bits(rng, next_rng, Val(Int(_draw_bits(T))))), next_rng
end

# Draws wider than one word reserve their whole span first, so a draw either
# completes or throws, then read it at the held position.
@inline function _rand_next_scalar(
    rng::_ScalarUniformGenerators,
    ::Type{T},
) where {T<:Union{_WideInteger,_ComplexResult}}
    next_rng = _reserve_scalar(rng, _draw_bits(T))
    return _draw_unchecked(rng, rng.position, T), next_rng
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
    rand_next(rng, collection) -> (value, next_rng)
    rand_next(rng[, T], dims...) -> (values, next_rng)
    rand_next(rng, collection, dims...) -> (values, next_rng)

Draw from `rng` and return the advanced immutable generator with the result.
Omitting `T` selects `Float64`. Supported scalar types are `Bool`, the 8- to 128-bit signed and unsigned integers,
`Float16`, `Float32`, `Float64`, `Complex` values of those three, and `Char`, which is
uniform over the Unicode scalar values as in `Random`.
Integer ranges support signed and unsigned integer element types through 128
bits. Dimensions may also be one tuple, as in `Random`. The 128-bit types run on
the CPU only.

A collection is an integer range, any other array or range, a tuple, a string, a
dict, or a set. A pick from it consumes and returns what one
[`randsample_next`](@ref) draw does. A tuple of `Int` is a shape, not a
collection. Strings, dicts, and sets reach the drawn position by iteration.

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
