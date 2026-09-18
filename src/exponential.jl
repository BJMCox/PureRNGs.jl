
@inline _exponential_bits(::Type{Float32}) = UInt16(24)
@inline _exponential_bits(::Type{Float64}) = UInt16(53)

@inline function _exponential_lattice(::Type{Float32}, value::UInt64)
    u = Float32(value) * Float32(0x1p-24)
    return u, one(Float32) - u
end

@inline function _exponential_lattice(::Type{Float64}, value::UInt64)
    u = Float64(value) * Float64(0x1p-53)
    return u, one(Float64) - u
end

struct _NativeTransformOps end

function _transform_muladd end

@inline _transform_muladd(::_NativeTransformOps, a, b, c) = fma(a, b, c)

@inline function _exponential_reduced(ops, ::Type{Float32}, m, n)
    one_ = reinterpret(Float32, UInt32(0x3f800000))
    ln2_hi = reinterpret(Float32, UInt32(0x3f317200))
    ln2_lo = reinterpret(Float32, UInt32(0x35c00000))
    c3 = reinterpret(Float32, UInt32(0x3eaaaaab))
    c5 = reinterpret(Float32, UInt32(0x3e4ccccd))
    c7 = reinterpret(Float32, UInt32(0x3e124925))
    c9 = reinterpret(Float32, UInt32(0x3de38e39))
    c11 = reinterpret(Float32, UInt32(0x3dba2e8c))

    t = (m - one_) / (m + one_)
    z = t * t
    p = _transform_muladd(ops, z, c11, c9)
    p = _transform_muladd(ops, z, p, c7)
    p = _transform_muladd(ops, z, p, c5)
    p = _transform_muladd(ops, z, p, c3)
    tz = t * z
    log_m = _transform_muladd(ops, tz, p, t)
    log_m += log_m
    return _transform_muladd(ops, n, ln2_hi, _transform_muladd(ops, n, ln2_lo, -log_m))
end

@inline function _exponential_reduced(ops, ::Type{Float64}, m, n)
    one_ = reinterpret(Float64, UInt64(0x3ff0000000000000))
    ln2_hi = reinterpret(Float64, UInt64(0x3fe62e42fef00000))
    ln2_lo = reinterpret(Float64, UInt64(0x3dd473de00000000))
    c3 = reinterpret(Float64, UInt64(0x3fd5555555555555))
    c5 = reinterpret(Float64, UInt64(0x3fc999999999999a))
    c7 = reinterpret(Float64, UInt64(0x3fc2492492492492))
    c9 = reinterpret(Float64, UInt64(0x3fbc71c71c71c71c))
    c11 = reinterpret(Float64, UInt64(0x3fb745d1745d1746))
    c13 = reinterpret(Float64, UInt64(0x3fb3b13b13b13b14))
    c15 = reinterpret(Float64, UInt64(0x3fb1111111111111))
    c17 = reinterpret(Float64, UInt64(0x3fae1e1e1e1e1e1e))
    c19 = reinterpret(Float64, UInt64(0x3faaf286bca1af28))
    c21 = reinterpret(Float64, UInt64(0x3fa8618618618618))
    c23 = reinterpret(Float64, UInt64(0x3fa642c8590b2164))

    t = (m - one_) / (m + one_)
    z = t * t
    p = _transform_muladd(ops, z, c23, c21)
    p = _transform_muladd(ops, z, p, c19)
    p = _transform_muladd(ops, z, p, c17)
    p = _transform_muladd(ops, z, p, c15)
    p = _transform_muladd(ops, z, p, c13)
    p = _transform_muladd(ops, z, p, c11)
    p = _transform_muladd(ops, z, p, c9)
    p = _transform_muladd(ops, z, p, c7)
    p = _transform_muladd(ops, z, p, c5)
    p = _transform_muladd(ops, z, p, c3)
    tz = t * z
    log_m = _transform_muladd(ops, tz, p, t)
    log_m += log_m
    return _transform_muladd(ops, n, ln2_hi, _transform_muladd(ops, n, ln2_lo, -log_m))
end

@inline function _exponential_transform(::_CPUBackend, ::Type{Float32}, v::Float32)
    sqrt2 = reinterpret(Float32, UInt32(0x3fb504f3))
    half = reinterpret(Float32, UInt32(0x3f000000))

    bits = reinterpret(UInt32, v)
    exponent = Int32((bits >> 23) & 0xff) - Int32(127)
    m = reinterpret(Float32, (bits & 0x007fffff) | 0x3f800000)
    upper = m > sqrt2
    m = ifelse(upper, m * half, m)
    exponent += ifelse(upper, Int32(1), Int32(0))
    n = -Float32(exponent)
    return _exponential_reduced(_NativeTransformOps(), Float32, m, n)
end

@inline function _exponential_transform(::_CPUBackend, ::Type{Float64}, v::Float64)
    sqrt2 = reinterpret(Float64, UInt64(0x3ff6a09e667f3bcd))
    half = reinterpret(Float64, UInt64(0x3fe0000000000000))

    bits = reinterpret(UInt64, v)
    exponent = Int64((bits >> 52) & 0x7ff) - Int64(1023)
    m = reinterpret(Float64, (bits & 0x000fffffffffffff) | 0x3ff0000000000000)
    upper = m > sqrt2
    m = ifelse(upper, m * half, m)
    exponent += ifelse(upper, Int64(1), Int64(0))
    n = -Float64(exponent)
    return _exponential_reduced(_NativeTransformOps(), Float64, m, n)
end

@inline _exponential_transform(::_CUDABackend, ::Type{T}, v::T) where {T} = -Base.log(v)
@inline _exponential_transform(::_AMDGPUBackend, ::Type{T}, v::T) where {T} = -Base.log(v)
@inline _exponential_transform(::_MetalBackend, ::Type{T}, v::T) where {T} = -Base.log(v)

@inline function _exponential_from_bits(device, ::Type{T}, value::UInt64) where {T}
    _, v = _exponential_lattice(T, value)
    return _exponential_transform(device, T, v)
end

@inline function _draw_exponential_unchecked(
    rng::_ScalarUniformGenerators,
    position,
    ::Type{T},
) where {T}
    block = _position_block(position)
    value = if T === Float32
        _extract_bits_unchecked(rng, block, position.bit, Val(24))
    else
        _extract_bits_unchecked(rng, block, position.bit, Val(53))
    end
    return _exponential_from_bits(rng.device, T, value)
end

@inline _draw_exponential_unchecked(rng::_ScalarUniformGenerators, ::Type{T}) where {T} =
    _draw_exponential_unchecked(rng, rng.position, T)

Random.randexp(::AbstractPureRNG) =
    _untyped_draw_error("randexp(rng, T)", "randexp_next(rng, T)")
Random.randexp(::AbstractPureRNG, ::Integer, ::Integer...) =
    _untyped_draw_error("randexp(rng, T, dims...)", "randexp_next(rng, dims...)")
Random.randexp(::AbstractPureRNG, ::Dims) =
    _untyped_draw_error("randexp(rng, T, dims...)", "randexp_next(rng, dims...)")

@inline function _randexp_next_scalar(rng::_ScalarUniformGenerators, ::Type{T}) where {T}
    next_rng = _reserve_scalar(rng, _exponential_bits(T))
    raw = _chain_bits(rng, next_rng, Val(Int(_exponential_bits(T))))
    return _exponential_from_bits(rng.device, T, raw), next_rng
end

@inline _randexp_scalar(rng::_ScalarUniformGenerators, ::Type{T}) where {T} =
    first(_randexp_next_scalar(rng, T))

@inline randexp_next(rng::_ScalarUniformGenerators) = randexp_next(rng, Float64)

for T in (Float32, Float64)
    @eval begin
        @inline Random.randexp(rng::_ScalarUniformGenerators, ::Type{$T}) =
            _randexp_scalar(rng, $T)
        @inline randexp_next(rng::_ScalarUniformGenerators, ::Type{$T}) =
            _randexp_next_scalar(rng, $T)
        @inline randexpat(rng::_ScalarUniformGenerators, ::Type{$T}, i::Integer) =
            _draw_exponential_unchecked(_addressed_rng(rng, _exponential_bits($T), i), $T)
        @inline randexpat(
            rng::_ScalarUniformGenerators,
            ::Type{$T},
            indices::AbstractUnitRange{<:Integer},
        ) = _addressed_array(rng, $T, indices, _exponential_bits($T), randexp_next)
    end
end

@doc """
    randexp_next(rng[, T]) -> (value, next_rng)
    randexp_next(rng[, T], dims...) -> (values, next_rng)

Draw standard exponential values from `rng` and return the advanced immutable
generator with the result. Omitting `T` selects `Float64`; `T` may be `Float32`
or `Float64`.

The allocating form creates an array on the generator's device. The input
generator never changes.
""" randexp_next

@doc """
    randexpat(rng, T, i)
    randexpat(rng, T, i:j)

Return the `i`th standard exponential draw at or after the current position of
`rng`, where `i` is one-based, or the vector of draws `i` through `j`. `T` is
`Float32` or `Float64`.

Addressed draws do not advance or change `rng`. They throw when `i` is not
positive or the addressed draw exceeds the generator's counter capacity.
""" randexpat

@inline _transformed_fill_plan(::_ExponentialCodec, backend, rng, T) = nothing

@inline _cooperative_value(codec::_ExponentialCodec, ::Type{T}, raw) where {T} =
    _exponential_from_bits(codec.backend, T, raw)
@inline _fill_width(::_ExponentialCodec, ::Type{T}) where {T} = _exponential_bits(T)

@inline function _randexp_next_fill!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    threaded::Bool,
) where {T}
    return _rand_transformed_next_fill!(
        rng,
        destination,
        threaded,
        _ExponentialCodec(rng.device),
    )
end

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randexp!(
            rng::_ScalarUniformGenerators,
            destination::AbstractArray{$T};
            threaded = true,
        )
            result, _ = _randexp_next_fill!(rng, destination, _check_threaded(threaded))
            return result
        end

        @inline function randexp_next!(
            rng::_ScalarUniformGenerators,
            destination::AbstractArray{$T};
            threaded = true,
        )
            return _randexp_next_fill!(rng, destination, _check_threaded(threaded))
        end
    end
end

@doc """
    randexp_next!(rng, destination; threaded=true) -> (destination, next_rng)

Fill a `Float32` or `Float64` destination with standard exponential values and
return the advanced immutable generator with the same destination. The
destination's device must match the generator.

Set `threaded=false` to request the serial CPU fill path. The keyword does not
change the generated stream. The input generator never changes.
""" randexp_next!
