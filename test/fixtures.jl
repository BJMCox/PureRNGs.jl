using PureRNGs

# Test helpers used by more than one test file. Every other test file may be
# included alone once this file is included.

const IR = PureRNGs
const MLD = PureRNGs.MLDataDevices

const GENERATOR_TYPES = (
    Philox2x32,
    Philox4x32,
    Philox2x64,
    Philox4x64,
    Threefry2x32,
    Threefry4x32,
    Threefry2x64,
    Threefry4x64,
    ChaCha,
)

const PURE_UNIFORM_TYPES = (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
const NORMAL_TYPES = (Float32, Float64)
const EXPONENTIAL_TYPES = (Float32, Float64)
const RANGE_INTS = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)

mutable struct TaskWriteProbe{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    writers::Array{Task,N}
end

TaskWriteProbe(data::AbstractArray{T,N}) where {T,N} =
    TaskWriteProbe(data, Array{Task}(undef, size(data)))
Base.size(array::TaskWriteProbe) = size(array.data)
Base.axes(array::TaskWriteProbe) = axes(array.data)
Base.IndexStyle(::Type{<:TaskWriteProbe{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.getindex(array::TaskWriteProbe, indices...) = getindex(array.data, indices...)
function Base.setindex!(array::TaskWriteProbe, value, indices...)
    array.writers[indices...] = current_task()
    return setindex!(array.data, value, indices...)
end
MLD.get_device(array::TaskWriteProbe) = MLD.get_device(array.data)

struct WrongDeviceArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end

Base.size(array::WrongDeviceArray) = size(array.data)
Base.axes(array::WrongDeviceArray) = axes(array.data)
Base.IndexStyle(::Type{<:WrongDeviceArray{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.getindex(array::WrongDeviceArray, indices...) = getindex(array.data, indices...)
Base.setindex!(array::WrongDeviceArray, value, indices...) =
    setindex!(array.data, value, indices...)
MLD.get_device_type(::WrongDeviceArray) = MLD.UnknownDevice

struct ZeroBasedVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.size(array::ZeroBasedVector) = size(array.data)
Base.axes(array::ZeroBasedVector) = (Base.IdentityUnitRange(0:(length(array.data)-1)),)
Base.IndexStyle(::Type{<:ZeroBasedVector}) = IndexLinear()
function Base.getindex(array::ZeroBasedVector, index::Int)
    checkbounds(array, index)
    return @inbounds array.data[index+1]
end
function Base.setindex!(array::ZeroBasedVector, value, index::Int)
    checkbounds(array, index)
    @inbounds array.data[index+1] = value
    return value
end
MLD.get_device(::ZeroBasedVector) = MLD.CPUDevice()

struct IdentityAxesMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
end

Base.size(array::IdentityAxesMatrix) = size(array.data)
Base.axes(::IdentityAxesMatrix) =
    (Base.IdentityUnitRange(0:1), Base.IdentityUnitRange(-1:1))
Base.IndexStyle(::Type{<:IdentityAxesMatrix}) = IndexCartesian()
function Base.getindex(array::IdentityAxesMatrix, row::Int, column::Int)
    checkbounds(array, row, column)
    return @inbounds array.data[row+1, column+2]
end
function Base.setindex!(array::IdentityAxesMatrix, value, row::Int, column::Int)
    checkbounds(array, row, column)
    @inbounds array.data[row+1, column+2] = value
    return value
end
MLD.get_device(::IdentityAxesMatrix) = MLD.CPUDevice()

struct SamplingCUDAProbe{T} <: AbstractVector{T}
    values::Vector{T}
end

Base.size(population::SamplingCUDAProbe) = size(population.values)
Base.getindex(population::SamplingCUDAProbe, index::Int) = population.values[index]
MLD.get_device(::SamplingCUDAProbe) = MLD.CUDADevice(:named)
MLD.get_device_type(::SamplingCUDAProbe) = MLD.CUDADevice

struct SamplingSerialProbe{T} <: AbstractVector{T}
    values::Vector{T}
end

Base.size(destination::SamplingSerialProbe) = size(destination.values)
Base.getindex(destination::SamplingSerialProbe, index::Int) = destination.values[index]
Base.setindex!(destination::SamplingSerialProbe, value, index::Int) =
    setindex!(destination.values, value, index)
MLD.get_device(::SamplingSerialProbe) = MLD.CPUDevice()
MLD.get_device_type(::SamplingSerialProbe) = MLD.CPUDevice
IR._fill_backend(::IR._CPUBackend, ::SamplingSerialProbe) =
    error("serial sampling must not get a backend")

_reference_block(rng::IR._Position64Generators, block::UInt64) = IR._block(rng, block)
_reference_block(rng::IR._Position128Generators, block::NTuple{2,UInt64}) =
    IR._block(rng, block...)

_reference_next(block::UInt64) = block + UInt64(1)
function _reference_next(block::NTuple{2,UInt64})
    lo = block[1] + UInt64(1)
    return lo, block[2] + UInt64(iszero(lo))
end

function _reference_extract(rng, block, bit::UInt16, width::Int)
    value = UInt64(0)
    remaining = width
    offset = Int(bit)
    while remaining != 0
        words = _reference_block(rng, block)
        word_bits = 8sizeof(first(words))
        block_bits = word_bits * length(words)
        word_lane, word_bit = divrem(offset, word_bits)
        take = min(remaining, word_bits - word_bit)
        mask = (UInt64(1) << take) - UInt64(1)
        piece = (UInt64(words[word_lane+1]) >> (word_bits - word_bit - take)) & mask
        value = (value << take) | piece
        remaining -= take
        offset += take
        if offset == block_bits
            block = _reference_next(block)
            offset = 0
        end
    end
    return value
end

function _reference_advance(rng, block, bit::UInt16, count::Int)
    block_bits =
        8sizeof(first(_reference_block(rng, block))) * length(_reference_block(rng, block))
    total = Int(bit) + count
    while total >= block_bits
        block = _reference_next(block)
        total -= block_bits
    end
    return block, UInt16(total)
end

function _reference_extract128(rng, block, bit)
    hi = _reference_extract(rng, block, bit, 64)
    next_block, next_bit = _reference_advance(rng, block, bit, 64)
    lo = _reference_extract(rng, next_block, next_bit, 64)
    return lo, hi
end

_reference_position_block(position::IR._Position64) = position.block
_reference_position_block(position::IR._Position128) = (position.lo, position.hi)

_uniform_width(::Type{Bool}) = 1
_uniform_width(::Type{Float32}) = 24
_uniform_width(::Type{UInt32}) = 32
_uniform_width(::Type{Int32}) = 32
_uniform_width(::Type{Float64}) = 53
_uniform_width(::Type{UInt64}) = 64
_uniform_width(::Type{Int64}) = 64

_reference_convert(::Type{Bool}, value::UInt64) = isone(value)
_reference_convert(::Type{UInt32}, value::UInt64) = value % UInt32
_reference_convert(::Type{Int32}, value::UInt64) = reinterpret(Int32, value % UInt32)
_reference_convert(::Type{UInt64}, value::UInt64) = value
_reference_convert(::Type{Int64}, value::UInt64) = reinterpret(Int64, value)
_reference_convert(::Type{Float32}, value::UInt64) =
    Float32(value % UInt32) * Float32(0x1p-24)
_reference_convert(::Type{Float64}, value::UInt64) = Float64(value) * 0x1p-53

function _reference_uniform(rng, ::Type{T}) where {T}
    raw = _reference_extract(
        rng,
        _reference_position_block(rng.position),
        rng.position.bit,
        _uniform_width(T),
    )
    return _reference_convert(T, raw)
end

function _reference_position(rng, additional_bits::Integer)
    block_bits = BigInt(IR._block_bits(rng))
    position = rng.position
    block =
        position isa IR._Position64 ? BigInt(position.block) :
        (BigInt(position.hi) << 64) + BigInt(position.lo)
    total = BigInt(position.bit) + BigInt(additional_bits)
    block_delta, bit = divrem(total, block_bits)
    block += block_delta
    if position isa IR._Position64
        return IR._Position64(UInt64(block), UInt16(bit))
    end
    return IR._Position128(
        UInt64(block & typemax(UInt64)),
        UInt64(block >> 64),
        UInt16(bit),
    )
end

function _positioned(F, seed, block::UInt64, bit::UInt16)
    base = F(seed)
    position =
        base.position isa IR._Position64 ? IR._Position64(block, bit) :
        IR._Position128(block, UInt64(7), bit)
    return IR._rebuild(base, position, base.device)
end

_range_reference_width(span::UInt64) =
    span != UInt64(0) && span <= UInt64(1) << 32 ? 64 : 128

function _range_position_from_absolute(rng, absolute::BigInt)
    block, bit = divrem(absolute, BigInt(IR._block_bits(rng)))
    if rng.position isa IR._Position64
        return IR._Position64(UInt64(block), UInt16(bit))
    end
    return IR._Position128(
        UInt64(block & typemax(UInt64)),
        UInt64(block >> 64),
        UInt16(bit),
    )
end

function _range_capacity(rng)
    blocks =
        rng.position isa IR._Position64 ? BigInt(IR._max_block(rng)) + 1 : big(1) << 128
    return blocks * BigInt(IR._block_bits(rng))
end

const PACKED_GOLDEN_BLOCK = UInt64(0x00123456789abcde)
const PACKED_GOLDEN_BIT = UInt16(61)
const PACKED_GOLDEN_GENERATORS = (
    (Philox2x32, (UInt32(0x01234567),)),
    (Philox4x32, (UInt32(0x01234567), UInt32(0x89abcdef))),
    (Philox2x64, (UInt64(0x0000000001234567),)),
    (Philox4x64, (UInt64(0x0000000001234567), UInt64(0x0000000089abcdef))),
    (Threefry2x32, (UInt32(0x01234567), UInt32(0x89abcdef))),
    (
        Threefry4x32,
        (UInt32(0x01234567), UInt32(0x89abcdef), UInt32(0xfedcba98), UInt32(0x76543210)),
    ),
    (Threefry2x64, (UInt64(0x0000000001234567), UInt64(0x0000000089abcdef))),
    (
        Threefry4x64,
        (
            UInt64(0x0000000001234567),
            UInt64(0x0000000089abcdef),
            UInt64(0x00000000fedcba98),
            UInt64(0x0000000076543210),
        ),
    ),
)

function _packed_golden_rng(F, key)
    base = F(key)
    position =
        base.position isa IR._Position64 ?
        IR._Position64(PACKED_GOLDEN_BLOCK, PACKED_GOLDEN_BIT) :
        IR._Position128(PACKED_GOLDEN_BLOCK, UInt64(0), PACKED_GOLDEN_BIT)
    return IR._rebuild(base, position, base.device)
end
