
const _AS241_A32 = (5.9109374720f1, 1.5929113202f2, 5.0434271938f1, 3.3871327179f0)
const _AS241_B32 = (6.7187563600f1, 7.8757757664f1, 1.7895169469f1, 1.0f0)
const _AS241_C32 = (1.7023821103f-1, 1.3067284816f0, 2.7568153900f0, 1.4234372777f0)
const _AS241_D32 = (1.2021132975f-1, 7.3700164250f-1, 1.0f0)

const _AS241_A64 = (
    2.5090809287301226727e3,
    3.3430575583588128105e4,
    6.7265770927008700853e4,
    4.5921953931549871457e4,
    1.3731693765509461125e4,
    1.9715909503065514427e3,
    1.3314166789178437745e2,
    3.3871328727963666080,
)
const _AS241_B64 = (
    5.2264952788528545610e3,
    2.8729085735721942674e4,
    3.9307895800092710610e4,
    2.1213794301586595867e4,
    5.3941960214247511077e3,
    6.8718700749205790830e2,
    4.2313330701600911252e1,
    1.0,
)
const _AS241_C64 = (
    7.74545014278341407640e-4,
    2.27238449892691845833e-2,
    2.41780725177450611770e-1,
    1.27045825245236838258,
    3.64784832476320460504,
    5.76949722146069140550,
    4.63033784615654529590,
    1.42343711074968357734,
)
const _AS241_D64 = (
    1.05075007164441684324e-9,
    5.47593808499534494600e-4,
    1.51986665636164571966e-2,
    1.48103976427480074590e-1,
    6.89767334985100004550e-1,
    1.67638483018380384940,
    2.05319162663775882187,
    1.0,
)
const _AS241_E64 = (
    2.01033439929228813265e-7,
    2.71155556874348757815e-5,
    1.24266094738807843860e-3,
    2.65321895265761230930e-2,
    2.96560571828504891230e-1,
    1.78482653991729133580,
    5.46378491116411436990,
    6.65790464350110377720,
)
const _AS241_F64 = (
    2.04426310338993978564e-15,
    1.42151175831644588870e-7,
    1.84631831751005468180e-5,
    7.86869131145613259100e-4,
    1.48753612908506148525e-2,
    1.36929880922735805310e-1,
    5.99832206555887937690e-1,
    1.0,
)

# Giles' erfinv (2010), published coefficients, in Horner order. It costs a
# logarithm on every draw, but its tail branch fires on 0.34 % of the lattice
# against AS241's 15 %, and it has no division. On an A100 that is 1.40x.
const _GILES_G1_32 = (
    2.81022636f-8,
    3.43273939f-7,
    -3.5233877f-6,
    -4.39150654f-6,
    0.00021858087f0,
    -0.00125372503f0,
    -0.00417768164f0,
    0.246640727f0,
    1.50140941f0,
)
const _GILES_G2_32 = (
    -0.000200214257f0,
    0.000100950558f0,
    0.00134934322f0,
    -0.00367342844f0,
    0.00573950773f0,
    -0.0076224613f0,
    0.00943887047f0,
    1.00167406f0,
    2.83297682f0,
)

const _GILES_G1_64 = (
    -3.6444120640178196996e-21,
    -1.685059138182016589e-19,
    1.2858480715256400167e-18,
    1.115787767802518096e-17,
    -1.333171662854620906e-16,
    2.0972767875968561637e-17,
    6.6376381343583238325e-15,
    -4.0545662729752068639e-14,
    -8.1519341976054721522e-14,
    2.6335093153082322977e-12,
    -1.2975133253453532498e-11,
    -5.4154120542946279317e-11,
    1.051212273321532285e-9,
    -4.1126339803469836976e-9,
    -2.9070369957882005086e-8,
    4.2347877827932403518e-7,
    -1.3654692000834678645e-6,
    -1.3882523362786468719e-5,
    0.0001867342080340571352,
    -0.00074070253416626697512,
    -0.0060336708714301490533,
    0.24015818242558961693,
    1.6536545626831027356,
)
const _GILES_G2_64 = (
    2.2137376921775787049e-9,
    9.0756561938885390979e-8,
    -2.7517406297064545428e-7,
    1.8239629214389227755e-8,
    1.5027403968909827627e-6,
    -4.013867526981545969e-6,
    2.9234449089955446044e-6,
    1.2475304481671778723e-5,
    -4.7318229009055733981e-5,
    6.8284851459573175448e-5,
    2.4031110387097893999e-5,
    -0.0003550375203628474796,
    0.00095328937973738049703,
    -0.0016882755560235047313,
    0.0024914420961078508066,
    -0.0037512085075692412107,
    0.005370914553590063617,
    1.0052589676941592334,
    3.0838856104922207635,
)
const _GILES_G3_64 = (
    -2.7109920616438573243e-11,
    -2.5556418169965252055e-10,
    1.5076572693500548083e-9,
    -3.7894654401267369937e-9,
    7.6157012080783393804e-9,
    -1.4960026627149240478e-8,
    2.9147953450901080826e-8,
    -6.7711997758452339498e-8,
    2.2900482228026654717e-7,
    -9.9298272942317002539e-7,
    4.5260625972231537039e-6,
    -1.9681778105531670567e-5,
    7.5995277030017761139e-5,
    -0.00021503011930044477347,
    -0.00013871931833623122026,
    1.0103004648645343977,
    4.8499064014085844221,
)

# The Float32 table stops at the near tail. Its far-tail pair is unreachable,
# for the reason `_as241_tail_chain` gives below.
@inline _as241_coefficients(::Type{Float32}) =
    (_AS241_A32, _AS241_B32, _AS241_C32, _AS241_D32)
@inline _as241_coefficients(::Type{Float64}) =
    (_AS241_A64, _AS241_B64, _AS241_C64, _AS241_D64, _AS241_E64, _AS241_F64)

@inline function _quantile_horner(x::T, coefficients::NTuple{N,T}) where {T,N}
    value = coefficients[1]
    for index = 2:N
        value = fma(value, x, coefficients[index])
    end
    return value
end

# [R28] `log` and `sqrt` are the execution site's native operations and are the
# only ones a fast-math substitution may touch. CUDA takes the substitution for
# `Float32` only: it costs 4.686 ulp against the 5.0 gate and buys 1.27x on an
# A100. `Float64` keeps the accurate pair because NVPTX lowers the fast
# `Float64` square root to `rsqrt.approx.f64`, which is accurate to 2^-23 and
# costs about 6e9 ulp, and because the fast pair buys nothing at that width.
# Both fast forms fall back to the plain operation off the device, so a host
# draw on a CUDA-token generator is unaffected.
@inline _normal_log(::_BackendToken, x) = log(x)
@inline _normal_sqrt(::_BackendToken, x) = sqrt(x)
@inline _normal_log(::_CUDABackend, x::Float32) = Base.FastMath.log_fast(x)
@inline _normal_sqrt(::_CUDABackend, x::Float32) = Base.FastMath.sqrt_fast(x)

@inline function _as241_central(q::T) where {T<:_UniformFloat}
    A, B = _as241_coefficients(T)
    r = T(0.180625) - q * q
    return q * (_quantile_horner(r, A) / _quantile_horner(r, B))
end

# [R28] the Float32 far tail `r > 5` is unreachable: the largest `r` the Float32
# midpoint lattice reaches is 4.08. LLVM cannot prove it and keeps a third
# division, which costs 5 % of the CPU fill and a division slow path on a GPU.
@inline function _as241_tail_chain(r::Float32)
    _, _, C, D = _as241_coefficients(Float32)
    s = r - 1.6f0
    return _quantile_horner(s, C) / _quantile_horner(s, D)
end

@inline function _as241_tail_chain(r::Float64)
    _, _, C, D, E, F = _as241_coefficients(Float64)
    if r <= 5.0
        s = r - 1.6
        return _quantile_horner(s, C) / _quantile_horner(s, D)
    end
    s = r - 5.0
    return _quantile_horner(s, E) / _quantile_horner(s, F)
end

@inline function _as241_tail(u::T, q::T) where {T<:_UniformFloat}
    z = _as241_tail_chain(sqrt(-log(q < zero(T) ? u : one(T) - u)))
    return q < zero(T) ? -z : z
end

@inline function _as241(u::T) where {T<:_UniformFloat}
    q = u - T(0.5)
    return abs(q) <= T(0.425) ? _as241_central(q) : _as241_tail(u, q)
end

# Defined on the [R28] midpoint lattice, where `T(2) * u - one(T)` is exact. Off
# the lattice that subtraction loses the low bits of `u` and the tail argument
# with them, so the ulp bounds hold for lattice inputs only.
#
# The branching form is kept deliberately. Selecting both chains instead costs
# 477 against 567 GiB/s on an A100: only 10 % of warps diverge, and an
# unconditional `sqrt` pulls in the domain-error path the branch avoids.
@inline function _giles_erfinv(device, u::Float32)
    sqrt2 = reinterpret(Float32, UInt32(0x3fb504f3))
    x = 2.0f0 * u - one(Float32)
    w = -_normal_log(device, (one(Float32) - x) * (one(Float32) + x))
    p = if w < 5.0f0
        _quantile_horner(w - 2.5f0, _GILES_G1_32)
    else
        _quantile_horner(_normal_sqrt(device, w) - 3.0f0, _GILES_G2_32)
    end
    return sqrt2 * (x * p)
end

@inline function _giles_erfinv(device, u::Float64)
    sqrt2 = reinterpret(Float64, UInt64(0x3ff6a09e667f3bcd))
    x = 2.0 * u - one(Float64)
    w = -_normal_log(device, (one(Float64) - x) * (one(Float64) + x))
    p = if w < 6.25
        _quantile_horner(w - 3.125, _GILES_G1_64)
    elseif w < 16.0
        _quantile_horner(_normal_sqrt(device, w) - 3.25, _GILES_G2_64)
    else
        _quantile_horner(_normal_sqrt(device, w) - 5.0, _GILES_G3_64)
    end
    return sqrt2 * (x * p)
end

# [R28] the backend token selects the transform. Both reach the same lattice
# ends and both stay inside the [R43] ulp gates; AS241 is the faster on a CPU
# and Giles the faster on a GPU, by a factor of about two in each direction.
@inline _normal_transform(::_CPUBackend, u::T) where {T<:_UniformFloat} = _as241(u)
@inline _normal_transform(device::_CUDABackend, u::T) where {T<:_UniformFloat} =
    _giles_erfinv(device, u)
@inline _normal_transform(device::_AMDGPUBackend, u::T) where {T<:_UniformFloat} =
    _giles_erfinv(device, u)
@inline _normal_transform(device::_MetalBackend, u::T) where {T<:_UniformFloat} =
    _giles_erfinv(device, u)

@inline function _open_midpoint(::Type{Float32}, value::UInt64)
    k32 = value % UInt32
    return Float32((k32 << UInt32(1)) | UInt32(1)) * Float32(0x1p-24)
end

@inline function _open_midpoint(::Type{Float64}, value::UInt64)
    k64 = value
    return Float64((k64 << UInt64(1)) | UInt64(1)) * Float64(0x1p-53)
end

@inline _normal_bits(::Type{Float32}) = UInt16(23)
@inline _normal_bits(::Type{Float64}) = UInt16(52)

function _midpoint_value end

@inline function _draw_normal_unchecked(
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
    return _normal_from_bits(rng.device, T, value)
end

Random.randn(::AbstractPureRNG) = _untyped_draw_error("randn(rng, T)", "randn_next(rng, T)")
Random.randn(::AbstractPureRNG, ::Integer, ::Integer...) =
    _untyped_draw_error("randn(rng, T, dims...)", "randn_next(rng, dims...)")
Random.randn(::AbstractPureRNG, ::Dims) =
    _untyped_draw_error("randn(rng, T, dims...)", "randn_next(rng, dims...)")

@inline randn_next(rng::_ScalarUniformGenerators) = randn_next(rng, Float64)

@inline Random.randn(rng::_ScalarUniformGenerators, ::Type{T}) where {T<:_UniformFloat} =
    first(_draw_next(rng, _NormalCodec(rng.device), T))
@inline randn_next(rng::_ScalarUniformGenerators, ::Type{T}) where {T<:_UniformFloat} =
    _draw_next(rng, _NormalCodec(rng.device), T)
@inline randn_at(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    i::Integer,
) where {T<:_UniformFloat} = _draw_at(rng, _NormalCodec(rng.device), T, i)
@inline randn_at(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    indices::AbstractUnitRange{<:Integer};
    threaded::Bool = false,
) where {T<:_UniformFloat} =
    _addressed_array(rng, T, indices, _normal_bits(T), randn_next, threaded)

@doc """
    randn_next(rng[, T]) -> (value, next_rng)
    randn_next(rng[, T], dims...) -> (values, next_rng)

Draw standard normal values from `rng` and return the advanced immutable
generator with the result. Omitting `T` selects `Float64`; `T` may be
`Float32` or `Float64`.

The allocating form creates an array on the generator's device. The input
generator never changes.
""" randn_next

@doc """
    randn_at(rng, T, i)
    randn_at(rng, T, i:j)

Return the `i`th standard normal draw at or after the current position of `rng`,
where `i` is one-based, or the vector of draws `i` through `j`. `T` is
`Float32` or `Float64`.

Addressed draws do not advance or change `rng`. They throw when `i` is not
positive or the addressed draw exceeds the generator's counter capacity.
""" randn_at

@inline _normal_from_bits(device, ::Type{T}, value::UInt64) where {T} =
    _normal_transform(device, _open_midpoint(T, value))
@inline _cooperative_value(codec::_NormalCodec, ::Type{T}, raw) where {T} =
    _normal_from_bits(codec.backend, T, raw)

@inline _fill_width(::_NormalCodec, ::Type{T}) where {T} = _normal_bits(T)

@inline function Random.randn!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_UniformFloat}
    result, _ =
        _rand_transformed_next_fill!(rng, destination, threaded, _NormalCodec(rng.device))
    return result
end

@inline function randn_next!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_UniformFloat}
    return _rand_transformed_next_fill!(
        rng,
        destination,
        threaded,
        _NormalCodec(rng.device),
    )
end

@doc """
    randn_next!(rng, destination; threaded=false) -> (destination, next_rng)

Fill a `Float32` or `Float64` destination with standard normal values and return
the advanced immutable generator with the same destination. The destination's
device must match the generator.

Fills run serially by default. Set `threaded=true` to split a CPU fill across
threads; the keyword never changes the generated stream. The input generator
never changes.
""" randn_next!

@inline function randn_next(
    rng::_ScalarUniformGenerators,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    return _rand_transformed_next_array(
        rng,
        Float64,
        (dim1, dims...),
        _NormalCodec(rng.device),
        threaded,
    )
end
@inline randn_next(rng::_ScalarUniformGenerators, dims::Dims; threaded::Bool = false) =
    _rand_transformed_next_array(rng, Float64, dims, _NormalCodec(rng.device), threaded)

@inline function Random.randn(
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
        _NormalCodec(rng.device),
        threaded,
    )
    return destination
end
@inline Random.randn(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:_UniformFloat} =
    first(_rand_transformed_next_array(rng, T, dims, _NormalCodec(rng.device), threaded))
@inline randn_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:_UniformFloat} =
    _rand_transformed_next_array(rng, T, dims, _NormalCodec(rng.device), threaded)

@inline function randn_next(
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
        _NormalCodec(rng.device),
        threaded,
    )
end
