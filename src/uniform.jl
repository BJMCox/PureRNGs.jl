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

@noinline function _fill_device_mismatch()
    throw(ArgumentError("destination device differs from the generator device"))
end

@inline _same_fill_device(::MLDataDevices.CPUDevice, ::MLDataDevices.CPUDevice) = true
@inline _same_fill_device(generator_device, destination_device) =
    generator_device == destination_device

@inline function _check_fill_device(rng::_ScalarUniformFamily, destination)
    _same_fill_device(rng.device, MLDataDevices.get_device(destination)) ||
        _fill_device_mismatch()
    return nothing
end

@inline _check_fill_serviceability(rng, destination, ::Type) = nothing

@inline function _fill_uniform_unchecked!(
    rng::_ScalarUniformFamily,
    position,
    destination,
    ::Type{T},
    indices,
) where {T}
    width = UInt64(_draw_bits(T))
    shift = _block_shift(rng)
    remaining = length(indices)
    @inbounds for index in indices
        destination[index] = _draw_unchecked(rng, position, T)
        remaining -= 1
        iszero(remaining) ||
            (position = _advance_position_unchecked(position, width, UInt64(0), shift))
    end
    return nothing
end

@inline _fill_uniform_unchecked!(rng, destination, ::Type{T}, indices) where {T} =
    _fill_uniform_unchecked!(rng, rng.position, destination, T, indices)

@inline _fill_uniform_unchecked!(rng, destination, ::Type{T}) where {T} =
    _fill_uniform_unchecked!(rng, destination, T, eachindex(destination))

KernelAbstractions.@kernel function _uniform_fill_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    _fill_uniform_unchecked!(rng, destination, T)
end

const _CPU_FILL_CHUNK_BITS = UInt64(4096 * 32)
const _CPU_FILL_MIN_WORKITEMS = 4

@inline _dense_fill_chunk_elements(::Type{T}) where {T} =
    Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_draw_bits(T)))
@inline function _dense_fill_bounds(workitem::Int, count::Int, chunk_elements::Int)
    first = (workitem - 1) * chunk_elements + 1
    chunk_count = min(chunk_elements, count - first + 1)
    return first, first + chunk_count - 1
end

KernelAbstractions.@kernel function _uniform_fill_dense_kernel!(
    rng,
    destination,
    ::Type{T},
    chunk_elements,
) where {T}
    workitem = @index(Global, Linear)
    first, last = _dense_fill_bounds(workitem, length(destination), chunk_elements)
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _draw_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_uniform_unchecked!(rng, position, destination, T, first:last)
end

KernelAbstractions.@kernel function _uniform_fill_dense_serial_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    _fill_uniform_unchecked!(rng, destination, T)
end

@inline _fill_backend(destination) = KernelAbstractions.get_backend(destination)
@inline _fill_backend(destination::BitArray) =
    KernelAbstractions.get_backend(destination.chunks)

function _launch_uniform!(backend, rng, destination, ::Type{T}) where {T}
    _uniform_fill_kernel!(backend)(rng, destination, T; ndrange = 1)
    return destination
end

function _launch_uniform!(
    backend::KernelAbstractions.CPU,
    rng,
    destination::Array{T},
    ::Type{T},
) where {T}
    chunk_elements = _dense_fill_chunk_elements(T)
    workitems = cld(length(destination), chunk_elements)
    if workitems < _CPU_FILL_MIN_WORKITEMS
        _uniform_fill_dense_serial_kernel!(backend)(rng, destination, T; ndrange = 1)
        return destination
    end
    _uniform_fill_dense_kernel!(backend)(
        rng,
        destination,
        T,
        chunk_elements;
        ndrange = workitems,
        workgroupsize = 1,
    )
    return destination
end

@inline function _rand_next_fill!(
    rng::_ScalarUniformFamily,
    destination::AbstractArray{T},
    threaded::Bool,
) where {T}
    _check_fill_device(rng, destination)
    _check_fill_serviceability(rng, destination, T)
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _draw_bits(T))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return next_rng, destination
    if !threaded && rng.device isa MLDataDevices.CPUDevice
        _fill_uniform_unchecked!(rng, destination, T)
        return next_rng, destination
    end
    backend = _fill_backend(destination)
    _launch_uniform!(backend, rng, destination, T)
    return next_rng, destination
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline function Random.rand!(
            rng::_ScalarUniformFamily,
            destination::AbstractArray{$T};
            threaded::Bool = true,
        )
            _, result = _rand_next_fill!(rng, destination, threaded)
            return result
        end

        @inline function rand_next!(
            rng::_ScalarUniformFamily,
            destination::AbstractArray{$T};
            threaded::Bool = true,
        )
            return _rand_next_fill!(rng, destination, threaded)
        end
    end
end

const _AddressIndex64 = Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}

@noinline function _invalid_address_index()
    throw(ArgumentError("addressed draw index must be positive"))
end

@noinline function _address_capacity_error()
    throw(ArgumentError("draw exceeds the generator counter capacity"))
end

@inline function _addressed_rng_device(
    rng::_ScalarUniformFamily,
    width::UInt16,
    i::_AddressIndex64,
)
    i < 1 && _invalid_address_index()
    index = UInt64(i)
    end_lo, end_hi = _bit_span(index, width)
    _, ok = _try_advance(rng, end_lo, end_hi)
    ok || _address_capacity_error()
    start_lo, start_hi = _bit_span(index - UInt64(1), width)
    position, _ = _try_advance(rng, start_lo, start_hi)
    position == rng.position && return rng
    return _rebuild(rng, position, rng.device)
end

@inline _addressed_rng(rng::_Position64Family, width::UInt16, i::_AddressIndex64) =
    _addressed_rng_device(rng, width, i)
@inline _addressed_rng(rng::_Position128Family, width::UInt16, i::_AddressIndex64) =
    _addressed_rng_device(rng, width, i)

@noinline function _addressed_rng(rng::_Position64Family, width::UInt16, i::Integer)
    i < 1 && _invalid_address_index()
    _, valid = _try_advance(rng, UInt64(0), UInt64(0))
    valid || _address_capacity_error()
    position = rng.position
    block_bits = BigInt(_block_bits(rng))
    current = BigInt(position.block) * block_bits + BigInt(position.bit)
    capacity = (BigInt(_max_block(rng)) + 1) * block_bits
    offset = (BigInt(i) - 1) * BigInt(width)
    current + offset + BigInt(width) <= capacity || _address_capacity_error()
    block, bit = divrem(current + offset, block_bits)
    return _rebuild(rng, _Position64(UInt64(block), UInt16(bit)), rng.device)
end

@noinline function _addressed_rng(rng::_Position128Family, width::UInt16, i::Integer)
    i < 1 && _invalid_address_index()
    _, valid = _try_advance(rng, UInt64(0), UInt64(0))
    valid || _address_capacity_error()
    position = rng.position
    block_bits = BigInt(_block_bits(rng))
    block = (BigInt(position.hi) << 64) + BigInt(position.lo)
    current = block * block_bits + BigInt(position.bit)
    capacity = (BigInt(1) << 128) * block_bits
    offset = (BigInt(i) - 1) * BigInt(width)
    current + offset + BigInt(width) <= capacity || _address_capacity_error()
    start_block, bit = divrem(current + offset, block_bits)
    mask = BigInt(typemax(UInt64))
    lo = UInt64(start_block & mask)
    hi = UInt64(start_block >> 64)
    return _rebuild(rng, _Position128(lo, hi, UInt16(bit)), rng.device)
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline randat(rng::_ScalarUniformFamily, ::Type{$T}, i::Integer) =
            _draw_unchecked(_addressed_rng(rng, _draw_bits($T), i), $T)
    end
end
