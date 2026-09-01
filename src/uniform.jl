@inline function _launch_uniform!(backend, rng, destination, ::Type{T}) where {T}
    plan = _device_uniform_fill_plan(backend, rng, T)
    return _launch_device_fill!(backend, rng, destination, T, Val(:uniform), plan)
end

function _launch_uniform!(
    backend::KernelAbstractions.CPU,
    rng,
    destination::Union{Array{T},BitArray},
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
    _check_serviceability(rng, T)
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _draw_bits(T))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return next_rng, destination
    if !threaded && rng.device isa _CPUBackend
        _fill_uniform_dense_cpu!(rng, rng.position, destination, T, eachindex(destination))
        return next_rng, destination
    end
    backend = _fill_backend(destination)
    _launch_uniform!(backend, rng, destination, T)
    return next_rng, destination
end

for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
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

@doc """
    rand_next!(rng, destination; threaded=true) -> (next_rng, destination)

Fill `destination` from `rng` and return the advanced immutable generator with
the same destination. The destination element type must be `Bool`, `UInt32`,
`Int32`, `UInt64`, `Int64`, `Float32`, or `Float64`, and its device must match
the generator.

Set `threaded=false` to request the serial CPU fill path. The keyword does not
change the generated stream. The input generator never changes.
""" rand_next!

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
    position = _advance_position_unchecked(rng, start_lo, start_hi)
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

for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
    @eval begin
        @inline randat(rng::_ScalarUniformFamily, ::Type{$T}, i::Integer) =
            _draw_unchecked(_addressed_rng(rng, _draw_bits($T), i), $T)
    end
end

@doc """
    randat(rng, T, i)

Return the `i`th uniform draw at or after the current position of `rng`, where
`i` is one-based. Supported result types are `Bool`, `UInt32`, `Int32`,
`UInt64`, `Int64`, `Float32`, and `Float64`.

Addressed draws do not advance or change `rng`. They throw when `i` is not
positive or the addressed draw exceeds the family's counter capacity.
""" randat
