const _ScalarUniform32Family = Union{Philox2x32,Philox4x32,Threefry2x32,Threefry4x32}
const _ScalarUniform64Family = Union{Philox2x64,Philox4x64,Threefry2x64,Threefry4x64}
const _ScalarUniformFamily = Union{_ScalarUniform32Family,_ScalarUniform64Family}

@inline _draw_bits(::Type{Bool}) = UInt16(1)
@inline _draw_bits(::Type{Float32}) = UInt16(24)
@inline _draw_bits(::Type{UInt32}) = UInt16(32)
@inline _draw_bits(::Type{Float64}) = UInt16(53)
@inline _draw_bits(::Type{UInt64}) = UInt16(64)

@inline _position_block(position::_Position64) = position.block
@inline _position_block(position::_Position128) = (position.lo, position.hi)

@inline function _draw_raw(rng::_ScalarUniformFamily, position, ::Val{W}) where {W}
    return _extract_bits_unchecked(
        rng,
        FAMILY_BITS,
        _position_block(position),
        position.bit,
        Val(W),
    )
end

@inline _draw_raw(rng::_ScalarUniformFamily, width) = _draw_raw(rng, rng.position, width)

@inline _from_bits(::Type{Bool}, value::UInt64) = isone(value)
@inline _from_bits(::Type{UInt32}, value::UInt64) = value % UInt32
@inline _from_bits(::Type{UInt64}, value::UInt64) = value
@inline _from_bits(::Type{Float32}, value::UInt64) =
    Float32(value % UInt32) * Float32(0x1p-24)
@inline _from_bits(::Type{Float64}, value::UInt64) = Float64(value) * 0x1p-53

@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{Bool}) =
    _from_bits(Bool, _draw_raw(rng, Val(1)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{Float32}) =
    _from_bits(Float32, _draw_raw(rng, Val(24)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{UInt32}) =
    _from_bits(UInt32, _draw_raw(rng, Val(32)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{Float64}) =
    _from_bits(Float64, _draw_raw(rng, Val(53)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{UInt64}) =
    _from_bits(UInt64, _draw_raw(rng, Val(64)))

@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{Bool}) =
    _from_bits(Bool, _draw_raw(rng, position, Val(1)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{Float32}) =
    _from_bits(Float32, _draw_raw(rng, position, Val(24)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{UInt32}) =
    _from_bits(UInt32, _draw_raw(rng, position, Val(32)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{Float64}) =
    _from_bits(Float64, _draw_raw(rng, position, Val(53)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{UInt64}) =
    _from_bits(UInt64, _draw_raw(rng, position, Val(64)))

function Random.rand(::AbstractPureRNG)
    throw(ArgumentError("untyped immutable draws are forbidden; use rand(rng, T)"))
end

@inline function _rand_scalar(rng::_ScalarUniformFamily, ::Type{T}) where {T}
    _reserve(rng, UInt64(_draw_bits(T)), UInt64(0))
    return _draw_unchecked(rng, T)
end

@inline rand_next(rng::_ScalarUniformFamily) = rand_next(rng, Float64)

@inline function _rand_next_scalar(rng::_ScalarUniformFamily, ::Type{T}) where {T}
    next_rng = _reserve(rng, UInt64(_draw_bits(T)), UInt64(0))
    return next_rng, _draw_unchecked(rng, T)
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline Random.rand(rng::_ScalarUniformFamily, ::Type{$T}) = _rand_scalar(rng, $T)
        @inline rand_next(rng::_ScalarUniformFamily, ::Type{$T}) =
            _rand_next_scalar(rng, $T)
    end
end

@doc """
    rand_next(rng[, T]) -> (next_rng, value)
    rand_next(rng, range) -> (next_rng, value)
    rand_next(rng[, T], dims...) -> (next_rng, values)
    rand_next(rng, range, dims...) -> (next_rng, values)

Draw from `rng` and return the advanced immutable generator with the result.
Omitting `T` selects `Float64`. Supported scalar types are `Bool`, `UInt32`,
`UInt64`, `Float32`, and `Float64`. Integer ranges support signed and unsigned
integer element types through 64 bits.

The allocating forms create an array on the generator's device. The input
generator never changes.
""" rand_next

@noinline function _fill_device_mismatch()
    throw(ArgumentError("destination device differs from the generator device"))
end

@inline function _same_fill_device(generator_device::_BackendToken, destination)
    return MLDataDevices.get_device_type(generator_device) ===
           MLDataDevices.get_device_type(destination)
end

@inline function _check_fill_device(rng::_ScalarUniformFamily, destination)
    _same_fill_device(rng.device, destination) || _fill_device_mismatch()
    return rng.device
end

@inline _check_serviceability(rng, ::Type) = nothing
@inline _with_device(f, ::_BackendToken) = f()
