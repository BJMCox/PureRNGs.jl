const FAMILY_EXP = UInt32(0x00000002)

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

@inline function _exponential_transform(::_CPUBackend, ::Type{Float32}, v::Float32)
    sqrt2 = reinterpret(Float32, UInt32(0x3fb504f3))
    half = reinterpret(Float32, UInt32(0x3f000000))
    one_ = reinterpret(Float32, UInt32(0x3f800000))
    ln2_hi = reinterpret(Float32, UInt32(0x3f317200))
    ln2_lo = reinterpret(Float32, UInt32(0x35c00000))
    c3 = reinterpret(Float32, UInt32(0x3eaaaaab))
    c5 = reinterpret(Float32, UInt32(0x3e4ccccd))
    c7 = reinterpret(Float32, UInt32(0x3e124925))
    c9 = reinterpret(Float32, UInt32(0x3de38e39))
    c11 = reinterpret(Float32, UInt32(0x3dba2e8c))

    bits = reinterpret(UInt32, v)
    exponent = Int32((bits >> 23) & 0xff) - Int32(127)
    m = reinterpret(Float32, (bits & 0x007fffff) | 0x3f800000)
    upper = m > sqrt2
    m = ifelse(upper, m * half, m)
    exponent += ifelse(upper, Int32(1), Int32(0))
    t = (m - one_) / (m + one_)
    z = t * t
    p = fma(z, c11, c9)
    p = fma(z, p, c7)
    p = fma(z, p, c5)
    p = fma(z, p, c3)
    tz = t * z
    log_m = fma(tz, p, t)
    log_m += log_m
    n = -Float32(exponent)
    return fma(n, ln2_hi, fma(n, ln2_lo, -log_m))
end

@inline function _exponential_transform(::_CPUBackend, ::Type{Float64}, v::Float64)
    sqrt2 = reinterpret(Float64, UInt64(0x3ff6a09e667f3bcd))
    half = reinterpret(Float64, UInt64(0x3fe0000000000000))
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

    bits = reinterpret(UInt64, v)
    exponent = Int64((bits >> 52) & 0x7ff) - Int64(1023)
    m = reinterpret(Float64, (bits & 0x000fffffffffffff) | 0x3ff0000000000000)
    upper = m > sqrt2
    m = ifelse(upper, m * half, m)
    exponent += ifelse(upper, Int64(1), Int64(0))
    t = (m - one_) / (m + one_)
    z = t * t
    p = fma(z, c23, c21)
    p = fma(z, p, c19)
    p = fma(z, p, c17)
    p = fma(z, p, c15)
    p = fma(z, p, c13)
    p = fma(z, p, c11)
    p = fma(z, p, c9)
    p = fma(z, p, c7)
    p = fma(z, p, c5)
    p = fma(z, p, c3)
    tz = t * z
    log_m = fma(tz, p, t)
    log_m += log_m
    n = -Float64(exponent)
    return fma(n, ln2_hi, fma(n, ln2_lo, -log_m))
end

@inline _exponential_transform(::_CUDABackend, ::Type{T}, v::T) where {T} = -Base.log(v)
@inline _exponential_transform(::_AMDGPUBackend, ::Type{T}, v::T) where {T} = -Base.log(v)
@inline _exponential_transform(::_MetalBackend, ::Type{T}, v::T) where {T} = -Base.log(v)

@inline function _exponential_from_bits(device, ::Type{T}, value::UInt64) where {T}
    _, v = _exponential_lattice(T, value)
    return _exponential_transform(device, T, v)
end

@inline function _draw_exponential_unchecked(
    rng::_ScalarUniformFamily,
    position,
    ::Type{T},
) where {T}
    block = _position_block(position)
    value = if T === Float32
        _extract_bits_unchecked(rng, FAMILY_EXP, block, position.bit, Val(24))
    else
        _extract_bits_unchecked(rng, FAMILY_EXP, block, position.bit, Val(53))
    end
    return _exponential_from_bits(rng.device, T, value)
end

@inline _draw_exponential_unchecked(rng::_ScalarUniformFamily, ::Type{T}) where {T} =
    _draw_exponential_unchecked(rng, rng.position, T)

function Random.randexp(::AbstractPureRNG)
    throw(ArgumentError("untyped immutable draws are forbidden; use randexp(rng, T)"))
end

@inline function _randexp_scalar(rng::_ScalarUniformFamily, ::Type{T}) where {T}
    _reserve(rng, UInt64(_exponential_bits(T)), UInt64(0))
    return _draw_exponential_unchecked(rng, T)
end

@inline randexp_next(rng::_ScalarUniformFamily) = randexp_next(rng, Float64)

@inline function _randexp_next_scalar(rng::_ScalarUniformFamily, ::Type{T}) where {T}
    next_rng = _reserve(rng, UInt64(_exponential_bits(T)), UInt64(0))
    return next_rng, _draw_exponential_unchecked(rng, T)
end

for T in (Float32, Float64)
    @eval begin
        @inline Random.randexp(rng::_ScalarUniformFamily, ::Type{$T}) =
            _randexp_scalar(rng, $T)
        @inline randexp_next(rng::_ScalarUniformFamily, ::Type{$T}) =
            _randexp_next_scalar(rng, $T)
        @inline randexpat(rng::_ScalarUniformFamily, ::Type{$T}, i::Integer) =
            _draw_exponential_unchecked(_addressed_rng(rng, _exponential_bits($T), i), $T)
    end
end

@doc """
    randexp_next(rng[, T]) -> (next_rng, value)
    randexp_next(rng[, T], dims...) -> (next_rng, values)

Draw standard exponential values from `rng` and return the advanced immutable
generator with the result. Omitting `T` selects `Float64`; `T` may be `Float32`
or `Float64`.

The allocating form creates an array on the generator's device. The input
generator never changes.
""" randexp_next

@doc """
    randexpat(rng, T, i)

Return the `i`th standard exponential draw at or after the current position of
`rng`, where `i` is one-based and `T` is `Float32` or `Float64`.

Addressed draws do not advance or change `rng`. They throw when `i` is not
positive or the addressed draw exceeds the family's counter capacity.
""" randexpat

@inline _device_exponential_fill_group(::Type{Float32}) = Val(8)
@inline _device_exponential_fill_group(::Type{Float64}) = Val(4)

@inline _cooperative_exponential_fill(::Philox4x32, ::Type{Float32}) = (Val(512), Val(32))
@inline _cooperative_exponential_fill(::Philox4x32, ::Type{Float64}) = (Val(512), Val(32))
@inline _cooperative_exponential_fill(rng, T) = nothing
@inline _device_exponential_fill_plan(backend, rng, T) = nothing

@inline function _fill_exponential_unchecked!(
    rng::_ScalarUniformFamily,
    position,
    destination,
    ::Type{T},
    indices,
) where {T}
    width = UInt64(_exponential_bits(T))
    shift = _block_shift(rng)
    remaining = length(indices)
    @inbounds for index in indices
        destination[index] = _draw_exponential_unchecked(rng, position, T)
        remaining -= 1
        iszero(remaining) ||
            (position = _advance_position_unchecked(position, width, UInt64(0), shift))
    end
    return nothing
end

@inline function _fill_exponential_grouped_unchecked!(
    rng,
    position,
    destination,
    ::Type{T},
    first::Int,
    group::Val{N},
) where {T,N}
    return _fill_grouped_cursor!(
        rng,
        position,
        destination,
        T,
        first,
        group,
        Val(_exponential_bits(T)),
        FAMILY_EXP,
        rng.device,
    )
end

@inline function _fill_exponential_dense_cpu!(
    rng,
    position,
    destination::Array{T},
    ::Type{T},
    indices,
) where {T<:Union{Float32,Float64}}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, FAMILY_EXP, _position_block(position), position.bit)
    width = Val(_exponential_bits(T))
    @inbounds for index in indices
        raw, cursor = _take_dense_bits_unchecked(rng, FAMILY_EXP, cursor, width)
        destination[index] = _exponential_from_bits(rng.device, T, raw)
    end
    return nothing
end

@inline function _fill_exponential_dense_cpu!(
    rng,
    position,
    destination,
    ::Type{T},
    indices,
) where {T}
    return _fill_exponential_unchecked!(rng, position, destination, T, indices)
end

KernelAbstractions.@kernel function _exponential_fill_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    ordinal = @index(Global, Linear)
    indices = eachindex(destination)
    index = @inbounds indices[firstindex(indices)+ordinal-1]
    bits_lo, bits_hi = _bit_span(UInt64(ordinal - 1), _exponential_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = _draw_exponential_unchecked(rng, position, T)
end

KernelAbstractions.@kernel function _exponential_fill_grouped_kernel!(
    rng,
    destination,
    ::Type{T},
    group::Val{N},
) where {T,N}
    workitem = @index(Global, Linear)
    first = (workitem - 1) * N + 1
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _exponential_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_exponential_grouped_unchecked!(rng, position, destination, T, first, group)
end

@inline _cooperative_value(device::_BackendToken, ::Type{T}, raw) where {T} =
    _exponential_from_bits(device, T, raw)
@inline _fill_width(::_BackendToken, ::Type{T}) where {T} = _exponential_bits(T)
@inline _fill_family(::_BackendToken) = FAMILY_EXP
@inline _fill_kernel(::_BackendToken) = _exponential_fill_kernel!
@inline _fill_grouped_kernel(::_BackendToken) = _exponential_fill_grouped_kernel!

@inline _exponential_fill_chunk_elements(::Type{T}) where {T} =
    Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_exponential_bits(T)))

KernelAbstractions.@kernel function _exponential_fill_dense_kernel!(
    rng,
    destination,
    ::Type{T},
    chunk_elements,
) where {T}
    workitem = @index(Global, Linear)
    first, last = _dense_fill_bounds(workitem, length(destination), chunk_elements)
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _exponential_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_exponential_dense_cpu!(rng, position, destination, T, first:last)
end

KernelAbstractions.@kernel function _exponential_fill_dense_serial_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    _fill_exponential_dense_cpu!(rng, rng.position, destination, T, eachindex(destination))
end

@inline function _launch_exponential!(backend, rng, destination, ::Type{T}) where {T}
    plan = _device_exponential_fill_plan(backend, rng, T)
    return _launch_device_fill!(backend, rng, destination, T, rng.device, plan)
end

function _launch_exponential!(
    backend::KernelAbstractions.CPU,
    rng,
    destination::Array{T},
    ::Type{T},
) where {T}
    chunk_elements = _exponential_fill_chunk_elements(T)
    workitems = cld(length(destination), chunk_elements)
    if workitems < _CPU_FILL_MIN_WORKITEMS
        _exponential_fill_dense_serial_kernel!(backend)(rng, destination, T; ndrange = 1)
        return destination
    end
    _exponential_fill_dense_kernel!(backend)(
        rng,
        destination,
        T,
        chunk_elements;
        ndrange = workitems,
        workgroupsize = 1,
    )
    return destination
end

@inline function _randexp_next_fill!(
    rng::_ScalarUniformFamily,
    destination::AbstractArray{T},
    threaded::Bool,
) where {T}
    _check_fill_device(rng, destination)
    _check_serviceability(rng, T)
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _exponential_bits(T))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return next_rng, destination
    if !threaded && rng.device isa _CPUBackend
        _fill_exponential_dense_cpu!(
            rng,
            rng.position,
            destination,
            T,
            eachindex(destination),
        )
        return next_rng, destination
    end
    backend = _fill_backend(destination)
    _launch_exponential!(backend, rng, destination, T)
    return next_rng, destination
end

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randexp!(
            rng::_ScalarUniformFamily,
            destination::AbstractArray{$T};
            threaded::Bool = true,
        )
            _, result = _randexp_next_fill!(rng, destination, threaded)
            return result
        end

        @inline function randexp_next!(
            rng::_ScalarUniformFamily,
            destination::AbstractArray{$T};
            threaded::Bool = true,
        )
            return _randexp_next_fill!(rng, destination, threaded)
        end
    end
end

@doc """
    randexp_next!(rng, destination; threaded=true) -> (next_rng, destination)

Fill a `Float32` or `Float64` destination with standard exponential values and
return the advanced immutable generator with the same destination. The
destination's device must match the generator.

Set `threaded=false` to request the serial CPU fill path. The keyword does not
change the generated stream. The input generator never changes.
""" randexp_next!
