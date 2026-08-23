const FAMILY_BITS = UInt32(0x00000000)

@inline _block(rng::Philox2x32, family::UInt32, block::UInt64) = _philox2x32(
    (block % UInt32, (family << 24) | (((block >> 32) & 0x00ffffff) % UInt32)),
    rng.key,
)

@inline _block(rng::Threefry2x32, family::UInt32, block::UInt64) = _threefry2x32(
    (block % UInt32, (family << 24) | (((block >> 32) & 0x00ffffff) % UInt32)),
    rng.key,
)

@inline _block(rng::Philox4x32, family::UInt32, block::UInt64) =
    _philox4x32((block % UInt32, (block >> 32) % UInt32, family, UInt32(0)), rng.key)

@inline _block(rng::Threefry4x32, family::UInt32, block::UInt64) =
    _threefry4x32((block % UInt32, (block >> 32) % UInt32, family, UInt32(0)), rng.key)

@inline _block(rng::Philox2x64, family::UInt32, block::UInt64) =
    _philox2x64((block, UInt64(family)), rng.key)

@inline _block(rng::Threefry2x64, family::UInt32, block::UInt64) =
    _threefry2x64((block, UInt64(family)), rng.key)

@inline _block(rng::Philox4x64, family::UInt32, block_lo::UInt64, block_hi::UInt64) =
    _philox4x64((block_lo, block_hi, UInt64(family), UInt64(0)), rng.key)

@inline _block(rng::Threefry4x64, family::UInt32, block_lo::UInt64, block_hi::UInt64) =
    _threefry4x64((block_lo, block_hi, UInt64(family), UInt64(0)), rng.key)

const _ScalarUniform32Family = Union{Philox2x32,Philox4x32,Threefry2x32,Threefry4x32}
const _ScalarUniform64Family = Union{Philox2x64,Philox4x64,Threefry2x64,Threefry4x64}
const _ScalarUniformWordType = Union{Bool,UInt32,Float32}

@inline _draw_words(::Type{<:_ScalarUniformWordType}) = UInt64(1)
@inline _draw_words(::Type{<:Union{UInt64,Float64}}) = UInt64(2)

@inline function _select_word(block::NTuple{N,UInt32}, lane::UInt8) where {N}
    word = block[1]
    for index = 2:N
        word = ifelse(lane == index - 1, block[index], word)
    end
    return word
end

@inline function _select_native_word(block::NTuple{N,UInt64}, lane::UInt8) where {N}
    word = block[1]
    for index = 2:N
        word = ifelse(lane == index - 1, block[index], word)
    end
    return word
end

@inline function _raw32(rng::_ScalarUniform32Family)
    position = rng.position
    block = _block(rng, FAMILY_BITS, position.block)
    return _select_word(block, position.lane)
end

@inline function _raw64(rng::_ScalarUniform32Family, family::UInt32)
    position = rng.position
    block = _block(rng, family, position.block)
    high = _select_word(block, position.lane)
    low = _select_word(block, position.lane + UInt8(1))
    return (UInt64(high) << 32) | UInt64(low)
end

@inline _raw64(rng::_ScalarUniform32Family) = _raw64(rng, FAMILY_BITS)

const _TwoWord64Family = Union{Philox2x64,Threefry2x64}
const _FourWord64Family = Union{Philox4x64,Threefry4x64}
const _ScalarUniformPosition64Family = Union{_ScalarUniform32Family,_TwoWord64Family}

@inline _native_block(rng::_TwoWord64Family, family::UInt32) =
    _block(rng, family, rng.position.block)
@inline _native_block(rng::_FourWord64Family, family::UInt32) =
    _block(rng, family, rng.position.lo, rng.position.hi)

@inline function _raw32(rng::_ScalarUniform64Family)
    lane = rng.position.lane
    word = _select_native_word(_native_block(rng, FAMILY_BITS), lane >> 1)
    return ifelse(iszero(lane & UInt8(1)), (word >> 32) % UInt32, word % UInt32)
end

@inline function _raw64(rng::_ScalarUniform64Family)
    lane = rng.position.lane
    return _select_native_word(_native_block(rng, FAMILY_BITS), lane >> 1)
end

const _TwoWord32Family = Union{Philox2x32,Threefry2x32}
const _FourWord32Family = Union{Philox4x32,Threefry4x32}

@inline function _raw128(rng::_TwoWord32Family, family::UInt32)
    position = rng.position
    first_block = _block(rng, family, position.block)
    second_block = _block(rng, family, position.block + UInt64(1))
    high = (UInt64(first_block[1]) << 32) | UInt64(first_block[2])
    low = (UInt64(second_block[1]) << 32) | UInt64(second_block[2])
    return high, low
end

@inline function _raw128(rng::_FourWord32Family, family::UInt32)
    position = rng.position
    block = _block(rng, family, position.block)
    high = (UInt64(block[1]) << 32) | UInt64(block[2])
    low = (UInt64(block[3]) << 32) | UInt64(block[4])
    return high, low
end

@inline _from_word(::Type{UInt32}, word::UInt32) = word
@inline _from_word(::Type{Bool}, word::UInt32) = isodd(word)
@inline _from_word(::Type{Float32}, word::UInt32) = Float32(word >> 8) * Float32(0x1p-24)
@inline _from_word(::Type{UInt64}, word::UInt64) = word
@inline _from_word(::Type{Float64}, word::UInt64) = Float64(word >> 11) * 0x1p-53

@inline _draw_unchecked(
    rng::_ScalarUniform32Family,
    ::Type{T},
) where {T<:_ScalarUniformWordType} = _from_word(T, _raw32(rng))
@inline _draw_unchecked(
    rng::_ScalarUniform32Family,
    ::Type{T},
) where {T<:Union{UInt64,Float64}} = _from_word(T, _raw64(rng))
@inline _draw_unchecked(
    rng::_ScalarUniform64Family,
    ::Type{T},
) where {T<:_ScalarUniformWordType} = _from_word(T, _raw32(rng))
@inline _draw_unchecked(
    rng::_ScalarUniform64Family,
    ::Type{T},
) where {T<:Union{UInt64,Float64}} = _from_word(T, _raw64(rng))

function Random.rand(::AbstractPureRNG)
    throw(ArgumentError("untyped immutable draws are forbidden; use rand(rng, T)"))
end

@inline function _rand_scalar(rng::_ScalarUniform32Family, ::Type{T}) where {T}
    words = _draw_words(T)
    start, _ = _reserve_aligned(rng, words, words)
    return _draw_unchecked(start, T)
end

@inline function _rand_scalar(rng::_ScalarUniform64Family, ::Type{T}) where {T}
    words = _draw_words(T)
    start, _ = _reserve_aligned(rng, words, words)
    return _draw_unchecked(start, T)
end

@inline rand_next(rng::_ScalarUniform32Family) = rand_next(rng, Float64)
@inline rand_next(rng::_ScalarUniform64Family) = rand_next(rng, Float64)

@inline function _rand_next_scalar(rng::_ScalarUniform32Family, ::Type{T}) where {T}
    words = _draw_words(T)
    start, next_rng = _reserve_aligned(rng, words, words)
    return next_rng, _draw_unchecked(start, T)
end

@inline function _rand_next_scalar(rng::_ScalarUniform64Family, ::Type{T}) where {T}
    words = _draw_words(T)
    start, next_rng = _reserve_aligned(rng, words, words)
    return next_rng, _draw_unchecked(start, T)
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline Random.rand(rng::_ScalarUniform32Family, ::Type{$T}) = _rand_scalar(rng, $T)
        @inline rand_next(rng::_ScalarUniform32Family, ::Type{$T}) =
            _rand_next_scalar(rng, $T)
        @inline Random.rand(rng::_ScalarUniform64Family, ::Type{$T}) = _rand_scalar(rng, $T)
        @inline rand_next(rng::_ScalarUniform64Family, ::Type{$T}) =
            _rand_next_scalar(rng, $T)
    end
end

@noinline function _fill_device_mismatch()
    throw(ArgumentError("destination device differs from the generator device"))
end

@inline _same_fill_device(::MLDataDevices.CPUDevice, ::MLDataDevices.CPUDevice) = true
@inline _same_fill_device(generator_device, destination_device) =
    generator_device == destination_device

@inline function _check_fill_device(rng::_ScalarUniform32Family, destination)
    _same_fill_device(rng.device, MLDataDevices.get_device(destination)) ||
        _fill_device_mismatch()
    return nothing
end

@inline _check_fill_serviceability(rng, destination, ::Type) = nothing

@inline function _fill_uniform_unchecked!(
    rng::_ScalarUniform32Family,
    destination,
    ::Type{T},
    indices,
) where {T<:_ScalarUniformWordType}
    position = rng.position
    block_index = position.block
    lane = position.lane
    width = _words_per_block(rng)
    block = _block(rng, FAMILY_BITS, block_index)
    remaining = length(indices)

    @inbounds for index in indices
        destination[index] = _from_word(T, _select_word(block, lane))
        remaining -= 1
        lane += UInt8(1)
        if lane == width && remaining != 0
            block_index += UInt64(1)
            lane = UInt8(0)
            block = _block(rng, FAMILY_BITS, block_index)
        end
    end
    return nothing
end

@inline function _fill_uniform_unchecked!(
    rng::_ScalarUniform32Family,
    destination,
    ::Type{T},
    indices,
) where {T<:Union{UInt64,Float64}}
    position = rng.position
    block_index = position.block
    lane = position.lane
    width = _words_per_block(rng)
    block = _block(rng, FAMILY_BITS, block_index)
    remaining = length(indices)

    @inbounds for index in indices
        high = _select_word(block, lane)
        lane += UInt8(1)
        if lane == width
            block_index += UInt64(1)
            lane = UInt8(0)
            block = _block(rng, FAMILY_BITS, block_index)
        end

        low = _select_word(block, lane)
        destination[index] = _from_word(T, (UInt64(high) << 32) | UInt64(low))
        remaining -= 1
        lane += UInt8(1)
        if lane == width && remaining != 0
            block_index += UInt64(1)
            lane = UInt8(0)
            block = _block(rng, FAMILY_BITS, block_index)
        end
    end
    return nothing
end

@inline _fill_uniform_unchecked!(rng, destination, ::Type{T}) where {T} =
    _fill_uniform_unchecked!(rng, destination, T, eachindex(destination))

KernelAbstractions.@kernel function _uniform_fill_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    _fill_uniform_unchecked!(rng, destination, T)
end

const _CPU_FILL_CHUNK_WORDS = UInt64(4096)
const _CPU_FILL_MIN_WORKITEMS = 4

@inline _dense_fill_chunk_elements(::Type{T}) where {T} =
    Int(_CPU_FILL_CHUNK_WORDS ÷ _draw_words(T))
@inline _dense_fill_workitems(count::Int, ::Type{T}) where {T} =
    cld(count, _dense_fill_chunk_elements(T))
@inline _use_parallel_dense_fill(workitems::Int) = workitems >= _CPU_FILL_MIN_WORKITEMS
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
    span = _draw_words(T)
    chunk_rng = _reserve(rng, UInt64(first - 1) * span)
    _fill_uniform_unchecked!(chunk_rng, destination, T, first:last)
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
    workitems = _dense_fill_workitems(length(destination), T)
    if !_use_parallel_dense_fill(workitems)
        _uniform_fill_kernel!(backend)(rng, destination, T; ndrange = 1)
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

@inline function _fill_word_count(count::Int, span::UInt64)
    # Array length is at most typemax(Int), and uniform spans are at most two words.
    # The product is therefore at most typemax(UInt64) - 1 on a 64-bit host.
    return UInt64(count) * span
end

@inline function _rand_next_fill!(
    rng::_ScalarUniform32Family,
    destination::AbstractArray{T},
) where {T}
    _check_fill_device(rng, destination)
    _check_fill_serviceability(rng, destination, T)
    words = _fill_word_count(length(destination), _draw_words(T))
    read_rng, next_rng = _reserve_aligned(rng, words, _draw_words(T))
    isempty(destination) && return next_rng, destination
    backend = _fill_backend(destination)
    _launch_uniform!(backend, read_rng, destination, T)
    return next_rng, destination
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline function Random.rand!(
            rng::_ScalarUniform32Family,
            destination::AbstractArray{$T},
        )
            _, result = _rand_next_fill!(rng, destination)
            return result
        end

        @inline rand_next!(rng::_ScalarUniform32Family, destination::AbstractArray{$T}) =
            _rand_next_fill!(rng, destination)
    end
end

const _AddressIndex64 = Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}

@noinline function _invalid_address_index()
    throw(ArgumentError("addressed draw index must be positive"))
end

@noinline function _address_capacity_error()
    throw(ArgumentError("draw exceeds the generator counter capacity"))
end

@inline function _addressed_rng(
    rng::_ScalarUniformPosition64Family,
    span::UInt64,
    i::_AddressIndex64,
)
    i < 1 && _invalid_address_index()
    start, _ = _reserve_aligned(rng, span, span)
    position = start.position

    width = UInt64(_words_per_block(start))
    elements_per_block = width ÷ span
    block_delta, element_lane = divrem(UInt64(i) - 1, elements_per_block)
    lane_words = UInt64(position.lane) + element_lane * span
    carry = UInt64(lane_words >= width)
    lane = ifelse(carry == 1, lane_words - width, lane_words)

    available = _max_block(start) - position.block
    block_delta > available && _address_capacity_error()
    carry > available - block_delta && _address_capacity_error()
    block = position.block + block_delta + carry
    return _rebuild(start, _Position64(block, UInt8(lane)), start.device)
end

function _addressed_rng(rng::_ScalarUniformPosition64Family, span::UInt64, i::Integer)
    i < 1 && _invalid_address_index()
    start, _ = _reserve_aligned(rng, span, span)
    position = start.position

    width = UInt64(_words_per_block(start))
    elements_per_block = width ÷ span
    block_delta, element_lane = divrem(BigInt(i) - 1, elements_per_block)
    lane_words = UInt64(position.lane) + UInt64(element_lane) * span
    carry = UInt64(lane_words >= width)
    lane = ifelse(carry == 1, lane_words - width, lane_words)

    available = BigInt(_max_block(start) - position.block)
    block_delta > available && _address_capacity_error()
    block_delta += carry
    block_delta > available && _address_capacity_error()
    block = position.block + UInt64(block_delta)
    return _rebuild(start, _Position64(block, UInt8(lane)), start.device)
end

@inline function _addressed_rng(rng::_FourWord64Family, span::UInt64, i::_AddressIndex64)
    i < 1 && _invalid_address_index()
    start, _ = _reserve_aligned(rng, span, span)
    position = start.position

    width = UInt64(_words_per_block(start))
    elements_per_block = width ÷ span
    block_delta, element_lane = divrem(UInt64(i) - 1, elements_per_block)
    lane_words = UInt64(position.lane) + element_lane * span
    lane_carry = UInt64(lane_words >= width)
    lane = ifelse(lane_carry == 1, lane_words - width, lane_words)
    block_delta += lane_carry

    block_lo = position.lo + block_delta
    block_carry = UInt64(block_lo < position.lo)
    block_hi = position.hi + block_carry
    block_hi < position.hi && _address_capacity_error()
    return _rebuild(start, _Position128(block_lo, block_hi, UInt8(lane)), start.device)
end

function _addressed_rng(rng::_FourWord64Family, span::UInt64, i::Integer)
    i < 1 && _invalid_address_index()
    start, _ = _reserve_aligned(rng, span, span)
    position = start.position

    width = UInt64(_words_per_block(start))
    elements_per_block = width ÷ span
    block_delta, element_lane = divrem(BigInt(i) - 1, elements_per_block)
    lane_words = UInt64(position.lane) + UInt64(element_lane) * span
    lane_carry = UInt64(lane_words >= width)
    lane = ifelse(lane_carry == 1, lane_words - width, lane_words)
    block_delta += lane_carry

    mask = BigInt(typemax(UInt64))
    available =
        (BigInt(typemax(UInt64) - position.hi) << 64) +
        BigInt(typemax(UInt64) - position.lo)
    block_delta > available && _address_capacity_error()
    delta_lo = UInt64(block_delta & mask)
    delta_hi = UInt64(block_delta >> 64)
    block_lo = position.lo + delta_lo
    block_carry = UInt64(block_lo < position.lo)
    block_hi = position.hi + delta_hi + block_carry
    return _rebuild(start, _Position128(block_lo, block_hi, UInt8(lane)), start.device)
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline randat(rng::_ScalarUniform32Family, ::Type{$T}, i::Integer) =
            _draw_unchecked(_addressed_rng(rng, _draw_words($T), i), $T)
        @inline randat(rng::_ScalarUniform64Family, ::Type{$T}, i::Integer) =
            _draw_unchecked(_addressed_rng(rng, _draw_words($T), i), $T)
    end
end
