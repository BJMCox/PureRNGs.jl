const FAMILY_NORMAL = UInt32(0x00000001)

const _AS241_A32 = (5.9109374720f1, 1.5929113202f2, 5.0434271938f1, 3.3871327179f0)
const _AS241_B32 = (6.7187563600f1, 7.8757757664f1, 1.7895169469f1, 1.0f0)
const _AS241_C32 = (1.7023821103f-1, 1.3067284816f0, 2.7568153900f0, 1.4234372777f0)
const _AS241_D32 = (1.2021132975f-1, 7.3700164250f-1, 1.0f0)
const _AS241_E32 = (1.7337203997f-2, 4.2868294337f-1, 3.0812263860f0, 6.6579051150f0)
const _AS241_F32 = (1.2258202635f-2, 2.4197894225f-1, 1.0f0)

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

@inline _as241_coefficients(::Type{Float32}) =
    (_AS241_A32, _AS241_B32, _AS241_C32, _AS241_D32, _AS241_E32, _AS241_F32)
@inline _as241_coefficients(::Type{Float64}) =
    (_AS241_A64, _AS241_B64, _AS241_C64, _AS241_D64, _AS241_E64, _AS241_F64)

@inline function _as241_horner(x::T, coefficients::NTuple{N,T}) where {T,N}
    value = coefficients[1]
    for index = 2:N
        value = fma(value, x, coefficients[index])
    end
    return value
end

@inline function _as241(u::T) where {T<:Union{Float32,Float64}}
    A, B, C, D, E, F = _as241_coefficients(T)
    q = u - T(0.5)
    if abs(q) <= T(0.425)
        r = T(0.180625) - q * q
        return q * (_as241_horner(r, A) / _as241_horner(r, B))
    end

    r = sqrt(-log(q < zero(T) ? u : one(T) - u))
    if r <= T(5)
        r = r - T(1.6)
        z = _as241_horner(r, C) / _as241_horner(r, D)
    else
        r = r - T(5)
        z = _as241_horner(r, E) / _as241_horner(r, F)
    end
    return q < zero(T) ? -z : z
end

@inline function _normal_midpoint(::Type{Float32}, value::UInt64)
    k32 = value % UInt32
    return Float32((k32 << UInt32(1)) | UInt32(1)) * Float32(0x1p-24)
end

@inline function _normal_midpoint(::Type{Float64}, value::UInt64)
    k64 = value
    return Float64((k64 << UInt64(1)) | UInt64(1)) * Float64(0x1p-53)
end

@inline _normal_bits(::Type{Float32}) = UInt16(23)
@inline _normal_bits(::Type{Float64}) = UInt16(52)

@inline function _draw_normal_unchecked(
    rng::_ScalarUniformFamily,
    position,
    ::Type{T},
) where {T}
    block = _position_block(position)
    value = if T === Float32
        _extract_bits_unchecked(rng, FAMILY_NORMAL, block, position.bit, Val(23))
    else
        _extract_bits_unchecked(rng, FAMILY_NORMAL, block, position.bit, Val(52))
    end
    return _as241(_normal_midpoint(T, value))
end

@inline _draw_normal_unchecked(rng::_ScalarUniformFamily, ::Type{T}) where {T} =
    _draw_normal_unchecked(rng, rng.position, T)

function Random.randn(::AbstractPureRNG)
    throw(ArgumentError("untyped immutable draws are forbidden; use randn(rng, T)"))
end

@inline function _randn_scalar(rng::_ScalarUniformFamily, ::Type{T}) where {T}
    _reserve(rng, UInt64(_normal_bits(T)), UInt64(0))
    return _draw_normal_unchecked(rng, T)
end

@inline randn_next(rng::_ScalarUniformFamily) = randn_next(rng, Float64)

@inline function _randn_next_scalar(rng::_ScalarUniformFamily, ::Type{T}) where {T}
    next_rng = _reserve(rng, UInt64(_normal_bits(T)), UInt64(0))
    return next_rng, _draw_normal_unchecked(rng, T)
end

for T in (Float32, Float64)
    @eval begin
        @inline Random.randn(rng::_ScalarUniformFamily, ::Type{$T}) = _randn_scalar(rng, $T)
        @inline randn_next(rng::_ScalarUniformFamily, ::Type{$T}) =
            _randn_next_scalar(rng, $T)
        @inline randnat(rng::_ScalarUniformFamily, ::Type{$T}, i::Integer) =
            _draw_normal_unchecked(_addressed_rng(rng, _normal_bits($T), i), $T)
    end
end

@inline function _fill_normal_unchecked!(
    rng::_ScalarUniformFamily,
    position,
    destination,
    ::Type{T},
    indices,
) where {T}
    width = UInt64(_normal_bits(T))
    shift = _block_shift(rng)
    remaining = length(indices)
    @inbounds for index in indices
        destination[index] = _draw_normal_unchecked(rng, position, T)
        remaining -= 1
        iszero(remaining) ||
            (position = _advance_position_unchecked(position, width, UInt64(0), shift))
    end
    return nothing
end

@inline _normal_from_bits(::Type{T}, value::UInt64) where {T} =
    _as241(_normal_midpoint(T, value))

@inline function _fill_normal_dense_cpu!(
    rng,
    position,
    destination::Array{T},
    ::Type{T},
    indices,
) where {T<:Union{Float32,Float64}}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, FAMILY_NORMAL, _position_block(position), position.bit)
    width = Val(_normal_bits(T))
    @inbounds for index in indices
        raw, cursor = _take_dense_bits_unchecked(rng, FAMILY_NORMAL, cursor, width)
        destination[index] = _normal_from_bits(T, raw)
    end
    return nothing
end

@inline _fill_normal_dense_cpu!(rng, position, destination, ::Type{T}, indices) where {T} =
    _fill_normal_unchecked!(rng, position, destination, T, indices)

KernelAbstractions.@kernel function _normal_fill_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    index = @index(Global, Linear)
    bits_lo, bits_hi = _bit_span(UInt64(index - 1), _normal_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = _draw_normal_unchecked(rng, position, T)
end

@inline _normal_fill_chunk_elements(::Type{T}) where {T} =
    Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_normal_bits(T)))

KernelAbstractions.@kernel function _normal_fill_dense_kernel!(
    rng,
    destination,
    ::Type{T},
    chunk_elements,
) where {T}
    workitem = @index(Global, Linear)
    first, last = _dense_fill_bounds(workitem, length(destination), chunk_elements)
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _normal_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_normal_dense_cpu!(rng, position, destination, T, first:last)
end

KernelAbstractions.@kernel function _normal_fill_dense_serial_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    _fill_normal_dense_cpu!(rng, rng.position, destination, T, eachindex(destination))
end

function _launch_normal!(backend, rng, destination, ::Type{T}) where {T}
    _normal_fill_kernel!(backend)(rng, destination, T; ndrange = length(destination))
    return destination
end

function _launch_normal!(
    backend::KernelAbstractions.CPU,
    rng,
    destination::Array{T},
    ::Type{T},
) where {T}
    chunk_elements = _normal_fill_chunk_elements(T)
    workitems = cld(length(destination), chunk_elements)
    if workitems < _CPU_FILL_MIN_WORKITEMS
        _normal_fill_dense_serial_kernel!(backend)(rng, destination, T; ndrange = 1)
        return destination
    end
    _normal_fill_dense_kernel!(backend)(
        rng,
        destination,
        T,
        chunk_elements;
        ndrange = workitems,
        workgroupsize = 1,
    )
    return destination
end

@inline function _randn_next_fill!(
    rng::_ScalarUniformFamily,
    destination::AbstractArray{T},
    threaded::Bool,
) where {T}
    device = _check_fill_device(rng, destination)
    _check_serviceability(rng, T)
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _normal_bits(T))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return next_rng, destination
    if !threaded && rng.device isa MLDataDevices.CPUDevice
        _fill_normal_dense_cpu!(rng, rng.position, destination, T, eachindex(destination))
        return next_rng, destination
    end
    _with_device(device) do
        backend = _fill_backend(destination)
        _launch_normal!(backend, rng, destination, T)
    end
    return next_rng, destination
end

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randn!(
            rng::_ScalarUniformFamily,
            destination::AbstractArray{$T};
            threaded::Bool = true,
        )
            _, result = _randn_next_fill!(rng, destination, threaded)
            return result
        end

        @inline function randn_next!(
            rng::_ScalarUniformFamily,
            destination::AbstractArray{$T};
            threaded::Bool = true,
        )
            return _randn_next_fill!(rng, destination, threaded)
        end
    end
end
