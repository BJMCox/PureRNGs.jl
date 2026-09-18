const _AddressIndex64 = Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}

@noinline function _invalid_address_index()
    throw(ArgumentError("addressed draw index must be positive"))
end

@inline function _addressed_rng_device(
    rng::_ScalarUniformGenerators,
    width::UInt16,
    i::_AddressIndex64,
)
    i < 1 && _invalid_address_index()
    index = UInt64(i)
    end_lo, end_hi = _bit_span(index, width)
    _, ok = _try_advance(rng, end_lo, end_hi)
    ok || _stream_exhausted(rng, end_lo, end_hi)
    start_lo, start_hi = _bit_span(index - UInt64(1), width)
    position = _advance_position_unchecked(rng, start_lo, start_hi)
    position == rng.position && return rng
    return _rebuild(rng, position, rng.device)
end

@inline _addressed_rng(rng::_Position64Generators, width::UInt16, i::_AddressIndex64) =
    _addressed_rng_device(rng, width, i)
@inline _addressed_rng(rng::_Position128Generators, width::UInt16, i::_AddressIndex64) =
    _addressed_rng_device(rng, width, i)

@noinline function _addressed_rng(rng::_Position64Generators, width::UInt16, i::Integer)
    i < 1 && _invalid_address_index()
    position = rng.position
    block_bits = BigInt(_block_bits(rng))
    offset = (BigInt(i) - 1) * BigInt(width)
    span = offset + BigInt(width)
    _, valid = _try_advance(rng, UInt64(0), UInt64(0))
    valid || _stream_exhausted(rng, span)
    current = BigInt(position.block) * block_bits + BigInt(position.bit)
    capacity = (BigInt(_max_block(rng)) + 1) * block_bits
    current + span <= capacity || _stream_exhausted(rng, span)
    block, bit = divrem(current + offset, block_bits)
    return _rebuild(rng, _Position64(UInt64(block), UInt16(bit)), rng.device)
end

@noinline function _addressed_rng(rng::_Position128Generators, width::UInt16, i::Integer)
    i < 1 && _invalid_address_index()
    position = rng.position
    block_bits = BigInt(_block_bits(rng))
    offset = (BigInt(i) - 1) * BigInt(width)
    span = offset + BigInt(width)
    _, valid = _try_advance(rng, UInt64(0), UInt64(0))
    valid || _stream_exhausted(rng, span)
    block = (BigInt(position.hi) << 64) + BigInt(position.lo)
    current = block * block_bits + BigInt(position.bit)
    capacity = (BigInt(1) << 128) * block_bits
    current + span <= capacity || _stream_exhausted(rng, span)
    start_block, bit = divrem(current + offset, block_bits)
    mask = BigInt(typemax(UInt64))
    lo = UInt64(start_block & mask)
    hi = UInt64(start_block >> 64)
    return _rebuild(rng, _Position128(lo, hi, UInt16(bit)), rng.device)
end

# The draws at consecutive addresses are the fill that starts at the first one.
@inline function _addressed_array(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    indices::AbstractUnitRange{<:Integer},
    width::UInt16,
    fill_next,
) where {T}
    isempty(indices) && return _allocate_draw_array(rng.device, T, (0,))
    return first(fill_next(_addressed_rng(rng, width, first(indices)), T, length(indices)))
end

@inline randat(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    i::Integer,
) where {T<:_UniformResult} = _draw_unchecked(_addressed_rng(rng, _draw_bits(T), i), T)
@inline randat(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    indices::AbstractUnitRange{<:Integer},
) where {T<:_UniformResult} = _addressed_array(rng, T, indices, _draw_bits(T), rand_next)

@doc """
    randat(rng, T, i)
    randat(rng, T, i:j)
    randat(rng, range, i)

Return the `i`th uniform draw at or after the current position of `rng`, where
`i` is one-based, or the vector of draws `i` through `j`. Supported result
types are `Bool`, `UInt32`, `Int32`, `UInt64`, `Int64`, `Float32`, and
`Float64`. Passing an integer range in place of `T` returns the `i`th draw
from that range.

Addressed draws do not advance or change `rng`. They throw when `i` is not
positive or the addressed draw exceeds the generator's counter capacity.

The range and distribution forms of `randat` take a single index.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> randat(rng, UInt32, 3)
0xc25ecc0b

julia> randat(rng, UInt32, 1:3)
3-element Vector{UInt32}:
 0x23b42aea
 0x467098dd
 0xc25ecc0b
```
""" randat
