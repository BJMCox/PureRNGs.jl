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
    ok || _stream_exhausted(typeof(rng), end_lo, end_hi)
    start_lo, start_hi = _bit_span(index - UInt64(1), width)
    position = _advance_position_unchecked(rng, start_lo, start_hi)
    position == rng.position && return rng
    return _rebuild(rng, position, rng.device)
end

@inline _addressed_rng(rng::AbstractPureRNG, width::UInt16, i::_AddressIndex64) =
    _addressed_rng_device(rng, width, i)

# An index too wide for UInt64 needs BigInt arithmetic, so this body stays out
# of the fast path. Both position widths are the same walk over stream bits:
# `rngposition` reads the current bit count and `_position_from_bits` writes it
# back, and both report the terminal state as the full capacity.
@noinline function _addressed_rng(rng::AbstractPureRNG, width::UInt16, i::Integer)
    i < 1 && _invalid_address_index()
    offset = (BigInt(i) - 1) * BigInt(width)
    span = offset + BigInt(width)
    _, valid = _try_advance(rng, UInt64(0), UInt64(0))
    valid || _stream_exhausted(typeof(rng), span)
    current = BigInt(rngposition(rng))
    capacity = _stream_capacity(typeof(rng))
    current + span <= capacity || _stream_exhausted(typeof(rng), span)
    return _rebuild(
        rng,
        _position_from_bits(typeof(rng), current + offset, capacity),
        rng.device,
    )
end

# The draws at consecutive addresses are the fill that starts at the first one.
@inline function _addressed_array(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    indices::AbstractUnitRange{<:Integer},
    width::UInt16,
    fill_next,
    threaded::Bool,
) where {T}
    _check_serviceability(rng, T)
    isempty(indices) && return _allocate_draw_array(rng.device, T, (0,))
    addressed = _addressed_rng(rng, width, first(indices))
    return first(fill_next(addressed, T, length(indices); threaded))
end

@inline rand_at(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    i::Integer,
) where {T<:_UniformResult} = _draw_unchecked(_addressed_rng(rng, _draw_bits(T), i), T)
@inline rand_at(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    indices::AbstractUnitRange{<:Integer};
    threaded::Bool = false,
) where {T<:_UniformResult} =
    _addressed_array(rng, T, indices, _draw_bits(T), rand_next, threaded)

@doc """
    rand_at(rng, T, i)
    rand_at(rng, T, i:j)
    rand_at(rng, range, i)

Return the `i`th uniform draw at or after the current position of `rng`, where
`i` is one-based, or the vector of draws `i` through `j`. Supported result
types are those of [`rand_next`](@ref). Passing an integer range in place of `T`
returns the `i`th draw from that range.

Addressed draws do not advance or change `rng`. They throw when `i` is not
positive or the addressed draw exceeds the generator's counter capacity.

The range and distribution forms of `rand_at` take a single index.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> rand_at(rng, UInt32, 3)
0xc25ecc0b

julia> rand_at(rng, UInt32, 1:3)
3-element Vector{UInt32}:
 0x23b42aea
 0x467098dd
 0xc25ecc0b
```
""" rand_at
