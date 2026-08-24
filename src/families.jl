abstract type AbstractPureRNG end

struct _CPUBackend end
struct _CUDABackend end
struct _AMDGPUBackend end
struct _MetalBackend end

const _BackendToken = Union{_CPUBackend,_CUDABackend,_AMDGPUBackend,_MetalBackend}

const _CPU_BACKEND = _CPUBackend()
const _CUDA_BACKEND = _CUDABackend()
const _AMDGPU_BACKEND = _AMDGPUBackend()
const _METAL_BACKEND = _MetalBackend()

MLDataDevices.get_device_type(::_CPUBackend) = MLDataDevices.CPUDevice
MLDataDevices.get_device_type(::_CUDABackend) = MLDataDevices.CUDADevice
MLDataDevices.get_device_type(::_AMDGPUBackend) = MLDataDevices.AMDGPUDevice
MLDataDevices.get_device_type(::_MetalBackend) = MLDataDevices.MetalDevice

const _EXHAUSTED_BIT = typemax(UInt16)

struct _Position64
    block::UInt64
    bit::UInt16
end

struct _Position128
    lo::UInt64
    hi::UInt64
    bit::UInt16
end

struct _ConstructionToken end
const _CONSTRUCTION_TOKEN = _ConstructionToken()

struct Philox2x32{D<:_BackendToken} <: AbstractPureRNG
    key::NTuple{1,UInt32}
    position::_Position64
    device::D

    Philox2x32{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Philox4x32{D<:_BackendToken} <: AbstractPureRNG
    key::NTuple{2,UInt32}
    position::_Position64
    device::D

    Philox4x32{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Philox2x64{D<:_BackendToken} <: AbstractPureRNG
    key::NTuple{1,UInt64}
    position::_Position64
    device::D

    Philox2x64{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Philox4x64{D<:_BackendToken} <: AbstractPureRNG
    key::NTuple{2,UInt64}
    position::_Position128
    device::D

    Philox4x64{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Threefry2x32{D<:_BackendToken} <: AbstractPureRNG
    key::NTuple{2,UInt32}
    position::_Position64
    device::D

    Threefry2x32{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Threefry4x32{D<:_BackendToken} <: AbstractPureRNG
    key::NTuple{4,UInt32}
    position::_Position64
    device::D

    Threefry4x32{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Threefry2x64{D<:_BackendToken} <: AbstractPureRNG
    key::NTuple{2,UInt64}
    position::_Position64
    device::D

    Threefry2x64{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Threefry4x64{D<:_BackendToken} <: AbstractPureRNG
    key::NTuple{4,UInt64}
    position::_Position128
    device::D

    Threefry4x64{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

const _Position64Family =
    Union{Philox2x32,Philox4x32,Philox2x64,Threefry2x32,Threefry4x32,Threefry2x64}
const _Position128Family = Union{Philox4x64,Threefry4x64}
const _NarrowFamily = Union{Philox2x32,Threefry2x32}
@inline _zero_position(::Type{<:_Position64Family}) = _Position64(0, 0)
@inline _zero_position(::Type{<:_Position128Family}) = _Position128(0, 0, 0)

for F in (
    :Philox2x32,
    :Philox4x32,
    :Philox2x64,
    :Philox4x64,
    :Threefry2x32,
    :Threefry4x32,
    :Threefry2x64,
    :Threefry4x64,
)
    @eval function $F(key::fieldtype($F, :key))
        device = _CPU_BACKEND
        return $F{typeof(device)}(_CONSTRUCTION_TOKEN, key, _zero_position($F), device)
    end
end

function _seed_key(::Type{T}, ::Val{N}, seed::Integer) where {T<:Unsigned,N}
    seed < 0 && throw(ArgumentError("seed must be non-negative"))
    value = BigInt(seed)
    bits = 8 * sizeof(T)
    value < (big(1) << (bits * N)) ||
        throw(ArgumentError("seed exceeds the family key width"))
    mask = BigInt(typemax(T))
    return ntuple(i -> T((value >> (bits * (i - 1))) & mask), Val(N))
end

Philox2x32(seed::Integer) = Philox2x32(_seed_key(UInt32, Val(1), seed))
Philox4x32(seed::Integer) = Philox4x32(_seed_key(UInt32, Val(2), seed))
Philox2x64(seed::Integer) = Philox2x64(_seed_key(UInt64, Val(1), seed))
Philox4x64(seed::Integer) = Philox4x64(_seed_key(UInt64, Val(2), seed))
Threefry2x32(seed::Integer) = Threefry2x32(_seed_key(UInt32, Val(2), seed))
Threefry4x32(seed::Integer) = Threefry4x32(_seed_key(UInt32, Val(4), seed))
Threefry2x64(seed::Integer) = Threefry2x64(_seed_key(UInt64, Val(2), seed))
Threefry4x64(seed::Integer) = Threefry4x64(_seed_key(UInt64, Val(4), seed))

for F in (
    :Philox2x32,
    :Philox4x32,
    :Philox2x64,
    :Philox4x64,
    :Threefry2x32,
    :Threefry4x32,
    :Threefry2x64,
    :Threefry4x64,
)
    @eval @inline _rebuild(rng::$F, position, device::D) where {D<:_BackendToken} =
        $F{D}(_CONSTRUCTION_TOKEN, rng.key, position, device)
end

for (Device, token) in (
    (MLDataDevices.CPUDevice, :_CPU_BACKEND),
    (MLDataDevices.CUDADevice, :_CUDA_BACKEND),
    (MLDataDevices.AMDGPUDevice, :_AMDGPU_BACKEND),
    (MLDataDevices.MetalDevice, :_METAL_BACKEND),
)
    @eval @inline (::$(Device))(rng::AbstractPureRNG) =
        _rebuild(rng, rng.position, $token)
end

@noinline function _unsupported_device(device)
    throw(ArgumentError("unsupported MLDataDevices device type: $(typeof(device))"))
end

@inline (device::MLDataDevices.AbstractDevice)(::AbstractPureRNG) =
    _unsupported_device(device)

@inline _block_shift(::_NarrowFamily) = UInt8(6)
@inline _block_shift(::Union{Philox4x32,Philox2x64,Threefry4x32,Threefry2x64}) = UInt8(7)
@inline _block_shift(::_Position128Family) = UInt8(8)
@inline _block_bits(rng::AbstractPureRNG) = UInt16(1) << _block_shift(rng)

@inline _max_block(::_NarrowFamily) = UInt64(0x00ffffffffffffff)
@inline _max_block(::_Position64Family) = typemax(UInt64)

@inline _terminal64(maximum::UInt64) = _Position64(maximum, _EXHAUSTED_BIT)
@inline _terminal128() = _Position128(typemax(UInt64), typemax(UInt64), _EXHAUSTED_BIT)

@inline _is_terminal(position::_Position64, maximum::UInt64) =
    position.block == maximum && position.bit == _EXHAUSTED_BIT
@inline _is_terminal(position::_Position128, ::Nothing) =
    position.lo == typemax(UInt64) &&
    position.hi == typemax(UInt64) &&
    position.bit == _EXHAUSTED_BIT

@inline _valid_position(position::_Position64, shift::UInt8, maximum::UInt64) =
    position.block <= maximum && UInt64(position.bit) < (UInt64(1) << shift)
@inline _valid_position(position::_Position128, shift::UInt8, ::Nothing) =
    UInt64(position.bit) < (UInt64(1) << shift)

@inline function _bit_span(count::UInt64, width::UInt16)
    hi, lo = _mulhilo64(count, UInt64(width))
    return lo, hi
end

@inline function _split_bit_advance(
    bit::UInt16,
    bits_lo::UInt64,
    bits_hi::UInt64,
    shift::UInt8,
)
    sum_lo = bits_lo + UInt64(bit)
    carry = UInt64(sum_lo < bits_lo)
    sum_hi = bits_hi + carry
    top = UInt64(sum_hi < bits_hi)
    mask = (UInt64(1) << shift) - UInt64(1)
    block_lo = (sum_lo >> shift) | (sum_hi << (UInt8(64) - shift))
    block_hi = (sum_hi >> shift) | (top << (UInt8(64) - shift))
    return block_lo, block_hi, UInt16(sum_lo & mask)
end

@inline function _advance_position_unchecked(
    position::_Position64,
    bits_lo::UInt64,
    bits_hi::UInt64,
    shift::UInt8,
)
    block_lo, _, bit = _split_bit_advance(position.bit, bits_lo, bits_hi, shift)
    return _Position64(position.block + block_lo, bit)
end

@inline function _advance_position_unchecked(
    position::_Position128,
    bits_lo::UInt64,
    bits_hi::UInt64,
    shift::UInt8,
)
    block_lo, block_hi, bit = _split_bit_advance(position.bit, bits_lo, bits_hi, shift)
    lo = position.lo + block_lo
    carry = UInt64(lo < position.lo)
    return _Position128(lo, position.hi + block_hi + carry, bit)
end

@inline _advance_position_unchecked(
    rng::AbstractPureRNG,
    bits_lo::UInt64,
    bits_hi::UInt64,
) = _advance_position_unchecked(rng.position, bits_lo, bits_hi, _block_shift(rng))

@inline function _try_advance(
    position::_Position64,
    bits_lo::UInt64,
    bits_hi::UInt64,
    shift::UInt8,
    maximum::UInt64,
)
    terminal = _terminal64(maximum)
    if _is_terminal(position, maximum)
        return iszero(bits_lo | bits_hi) ? (position, true) : (terminal, false)
    end
    _valid_position(position, shift, maximum) || return terminal, false
    iszero(bits_lo | bits_hi) && return position, true

    delta_lo, delta_hi, next_bit = _split_bit_advance(position.bit, bits_lo, bits_hi, shift)
    available = maximum - position.block
    available_lo = available + UInt64(1)
    available_hi = UInt64(iszero(available_lo))
    if delta_hi < available_hi || (delta_hi == available_hi && delta_lo < available_lo)
        return _Position64(position.block + delta_lo, next_bit), true
    elseif delta_lo == available_lo && delta_hi == available_hi && iszero(next_bit)
        return terminal, true
    end
    return terminal, false
end

@inline function _try_advance(
    position::_Position128,
    bits_lo::UInt64,
    bits_hi::UInt64,
    shift::UInt8,
    ::Nothing,
)
    terminal = _terminal128()
    if _is_terminal(position, nothing)
        return iszero(bits_lo | bits_hi) ? (position, true) : (terminal, false)
    end
    _valid_position(position, shift, nothing) || return terminal, false
    iszero(bits_lo | bits_hi) && return position, true

    delta_lo, delta_hi, next_bit = _split_bit_advance(position.bit, bits_lo, bits_hi, shift)
    next_lo = position.lo + delta_lo
    carry = UInt64(next_lo < position.lo)
    partial_hi = position.hi + delta_hi
    overflow = partial_hi < position.hi
    next_hi = partial_hi + carry
    overflow |= next_hi < partial_hi
    if !overflow
        return _Position128(next_lo, next_hi, next_bit), true
    elseif iszero(next_lo | next_hi | UInt64(next_bit))
        return terminal, true
    end
    return terminal, false
end

@inline _try_advance(rng::_Position64Family, bits_lo::UInt64, bits_hi::UInt64) =
    _try_advance(rng.position, bits_lo, bits_hi, _block_shift(rng), _max_block(rng))
@inline _try_advance(rng::_Position128Family, bits_lo::UInt64, bits_hi::UInt64) =
    _try_advance(rng.position, bits_lo, bits_hi, _block_shift(rng), nothing)

@inline function _reserve(rng::AbstractPureRNG, bits_lo::UInt64, bits_hi::UInt64)
    position, ok = _try_advance(rng, bits_lo, bits_hi)
    ok || throw(ArgumentError("draw exceeds the generator counter capacity"))
    position == rng.position && return rng
    return _rebuild(rng, position, rng.device)
end
