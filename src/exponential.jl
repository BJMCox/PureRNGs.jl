
@inline _exponential_bits(::Type{Float32}) = UInt16(23)
@inline _exponential_bits(::Type{Float64}) = UInt16(52)

# Draws on the open midpoint lattice the normal transform uses: `u` and
# `v = 1 - u` are both exact in `T`, neither endpoint occurs, so the transform
# output is strictly positive with reach `(width + 1) * ln 2`.
@inline function _exponential_lattice(::Type{T}, value::UInt64) where {T<:_UniformFloat}
    u = _open_midpoint(T, value)
    return u, one(T) - u
end

struct _NativeTransformOps end

function _transform_muladd end
function _transform_product end

@inline _transform_muladd(::_NativeTransformOps, a, b, c) = fma(a, b, c)
# `half` is the midpoint of `b`'s range. A backend whose compiler contracts the
# product's rounding into the next operation uses it to build the dynamic sign
# `_rounded_product` needs; the native product ignores it.
@inline _transform_product(::_NativeTransformOps, a, b, half) = a * b

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
        _extract_bits_unchecked(rng, block, position.bit, Val(23))
    else
        _extract_bits_unchecked(rng, block, position.bit, Val(52))
    end
    return _exponential_from_bits(rng.device, T, value)
end

Random.randexp(::AbstractPureRNG) =
    _untyped_draw_error("randexp(rng, T)", "randexp_next(rng, T)")
Random.randexp(::AbstractPureRNG, ::Integer, ::Integer...) =
    _untyped_draw_error("randexp(rng, T, dims...)", "randexp_next(rng, dims...)")
Random.randexp(::AbstractPureRNG, ::Dims) =
    _untyped_draw_error("randexp(rng, T, dims...)", "randexp_next(rng, dims...)")

@inline randexp_next(rng::_ScalarUniformGenerators) = randexp_next(rng, Float64)

@inline Random.randexp(rng::_ScalarUniformGenerators, ::Type{T}) where {T<:_UniformFloat} =
    first(_draw_next(rng, _ExponentialCodec(rng.device), T))
@inline randexp_next(rng::_ScalarUniformGenerators, ::Type{T}) where {T<:_UniformFloat} =
    _draw_next(rng, _ExponentialCodec(rng.device), T)
@inline randexp_at(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    i::Integer,
) where {T<:_UniformFloat} = _draw_at(rng, _ExponentialCodec(rng.device), T, i)
@inline randexp_at(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    indices::AbstractUnitRange{<:Integer};
    threaded::Bool = false,
) where {T<:_UniformFloat} =
    _addressed_array(rng, T, indices, _exponential_bits(T), randexp_next, threaded)

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
    randexp_at(rng, T, i)
    randexp_at(rng, T, i:j)

Return the `i`th standard exponential draw at or after the current position of
`rng`, where `i` is one-based, or the vector of draws `i` through `j`. `T` is
`Float32` or `Float64`.

Addressed draws do not advance or change `rng`. They throw when `i` is not
positive or the addressed draw exceeds the generator's counter capacity.
""" randexp_at

@inline _cooperative_value(codec::_ExponentialCodec, ::Type{T}, raw) where {T} =
    _exponential_from_bits(codec.backend, T, raw)
@inline _fill_width(::_ExponentialCodec, ::Type{T}) where {T} = _exponential_bits(T)

@inline function Random.randexp!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_UniformFloat}
    result, _ = _rand_transformed_next_fill!(
        rng,
        destination,
        threaded,
        _ExponentialCodec(rng.device),
    )
    return result
end

@inline function randexp_next!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_UniformFloat}
    return _rand_transformed_next_fill!(
        rng,
        destination,
        threaded,
        _ExponentialCodec(rng.device),
    )
end

@doc """
    randexp_next!(rng, destination; threaded=false) -> (destination, next_rng)

Fill a `Float32` or `Float64` destination with standard exponential values and
return the advanced immutable generator with the same destination. The
destination's device must match the generator.

Fills run serially by default. Set `threaded=true` to split a CPU fill across
threads; the keyword never changes the generated stream. The input generator
never changes.
""" randexp_next!

@inline function randexp_next(
    rng::_ScalarUniformGenerators,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    return _rand_transformed_next_array(
        rng,
        Float64,
        (dim1, dims...),
        _ExponentialCodec(rng.device),
        threaded,
    )
end
@inline randexp_next(rng::_ScalarUniformGenerators, dims::Dims; threaded::Bool = false) =
    _rand_transformed_next_array(
        rng,
        Float64,
        dims,
        _ExponentialCodec(rng.device),
        threaded,
    )

@inline function Random.randexp(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) where {T<:_UniformFloat}
    destination, _ = _rand_transformed_next_array(
        rng,
        T,
        (dim1, dims...),
        _ExponentialCodec(rng.device),
        threaded,
    )
    return destination
end
@inline Random.randexp(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:_UniformFloat} = first(
    _rand_transformed_next_array(rng, T, dims, _ExponentialCodec(rng.device), threaded),
)
@inline randexp_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:_UniformFloat} =
    _rand_transformed_next_array(rng, T, dims, _ExponentialCodec(rng.device), threaded)

@inline function randexp_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) where {T<:_UniformFloat}
    return _rand_transformed_next_array(
        rng,
        T,
        (dim1, dims...),
        _ExponentialCodec(rng.device),
        threaded,
    )
end
