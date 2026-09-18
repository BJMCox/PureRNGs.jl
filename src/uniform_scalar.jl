const _ScalarUniform32Generators =
    Union{Philox2x32,Philox4x32,Threefry2x32,Threefry4x32,ChaCha}
const _ScalarUniform64Generators = Union{Philox2x64,Philox4x64,Threefry2x64,Threefry4x64}
const _ScalarUniformGenerators =
    Union{_ScalarUniform32Generators,_ScalarUniform64Generators}
const _UniformInteger32 = Union{Int32,UInt32}
const _UniformInteger64 = Union{Int64,UInt64}
const _UniformInteger = Union{_UniformInteger32,_UniformInteger64}

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

@inline _draw_unchecked(rng::_ScalarUniformGenerators, ::Type{Bool}) =
    _from_bits(Bool, _draw_raw(rng, Val(1)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, ::Type{Float32}) =
    _from_bits(Float32, _draw_raw(rng, Val(24)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, ::Type{UInt32}) =
    _from_bits(UInt32, _draw_raw(rng, Val(32)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, ::Type{Int32}) =
    _from_bits(Int32, _draw_raw(rng, Val(32)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, ::Type{Float64}) =
    _from_bits(Float64, _draw_raw(rng, Val(53)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, ::Type{UInt64}) =
    _from_bits(UInt64, _draw_raw(rng, Val(64)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, ::Type{Int64}) =
    _from_bits(Int64, _draw_raw(rng, Val(64)))

@inline _draw_unchecked(rng::_ScalarUniformGenerators, position, ::Type{Bool}) =
    _from_bits(Bool, _draw_raw(rng, position, Val(1)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, position, ::Type{Float32}) =
    _from_bits(Float32, _draw_raw(rng, position, Val(24)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, position, ::Type{UInt32}) =
    _from_bits(UInt32, _draw_raw(rng, position, Val(32)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, position, ::Type{Int32}) =
    _from_bits(Int32, _draw_raw(rng, position, Val(32)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, position, ::Type{Float64}) =
    _from_bits(Float64, _draw_raw(rng, position, Val(53)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, position, ::Type{UInt64}) =
    _from_bits(UInt64, _draw_raw(rng, position, Val(64)))
@inline _draw_unchecked(rng::_ScalarUniformGenerators, position, ::Type{Int64}) =
    _from_bits(Int64, _draw_raw(rng, position, Val(64)))

@noinline function _untyped_draw_error(held_form::String, next_form::String)
    throw(
        ArgumentError(
            "untyped immutable draws are forbidden; use $held_form to draw at the " *
            "held position, or $next_form to advance",
        ),
    )
end

# [R23] Nine guard methods keep Base's untyped fallbacks unreachable. The dims
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

for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
    @eval begin
        @inline Random.rand(rng::_ScalarUniformGenerators, ::Type{$T}) =
            _rand_scalar(rng, $T)
        @inline rand_next(rng::_ScalarUniformGenerators, ::Type{$T}) =
            _rand_next_scalar(rng, $T)
    end
end

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
""" rand_next

@noinline function _fill_device_mismatch()
    throw(ArgumentError("destination device differs from the generator device"))
end

@inline function _same_fill_device(generator_device::_BackendToken, destination)
    return MLDataDevices.get_device_type(generator_device) ===
           MLDataDevices.get_device_type(destination)
end

@inline function _check_fill_device(rng::_ScalarUniformGenerators, destination)
    _same_fill_device(rng.device, destination) || _fill_device_mismatch()
    return nothing
end

@inline _check_serviceability(rng, ::Type) = nothing
@inline _check_serviceability(rng, range::AbstractRange) =
    _check_serviceability(rng, eltype(range))
