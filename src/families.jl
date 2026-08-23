abstract type AbstractPureRNG end

const _EXHAUSTED_LANE = typemax(UInt8)

struct _Position64
    block::UInt64
    lane::UInt8
end

struct _Position128
    lo::UInt64
    hi::UInt64
    lane::UInt8
end

struct _ConstructionToken end
const _CONSTRUCTION_TOKEN = _ConstructionToken()

struct Philox2x32{D} <: AbstractPureRNG
    key::NTuple{1,UInt32}
    position::_Position64
    device::D

    Philox2x32{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Philox4x32{D} <: AbstractPureRNG
    key::NTuple{2,UInt32}
    position::_Position64
    device::D

    Philox4x32{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Philox2x64{D} <: AbstractPureRNG
    key::NTuple{1,UInt64}
    position::_Position64
    device::D

    Philox2x64{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Philox4x64{D} <: AbstractPureRNG
    key::NTuple{2,UInt64}
    position::_Position128
    device::D

    Philox4x64{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Threefry2x32{D} <: AbstractPureRNG
    key::NTuple{2,UInt32}
    position::_Position64
    device::D

    Threefry2x32{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Threefry4x32{D} <: AbstractPureRNG
    key::NTuple{4,UInt32}
    position::_Position64
    device::D

    Threefry4x32{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Threefry2x64{D} <: AbstractPureRNG
    key::NTuple{2,UInt64}
    position::_Position64
    device::D

    Threefry2x64{D}(::_ConstructionToken, key, position, device) where {D} =
        new{D}(key, position, device)
end

struct Threefry4x64{D} <: AbstractPureRNG
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
        device = MLDataDevices.CPUDevice()
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
    @eval @inline _rebuild(rng::$F, position, device::D) where {D} =
        $F{D}(_CONSTRUCTION_TOKEN, rng.key, position, device)
end

@inline (device::MLDataDevices.CPUDevice)(rng::AbstractPureRNG) =
    _rebuild(rng, rng.position, device)

@inline _words_per_block(::_NarrowFamily) = UInt8(2)
@inline _words_per_block(::Union{Philox4x32,Philox2x64,Threefry4x32,Threefry2x64}) =
    UInt8(4)
@inline _words_per_block(::_Position128Family) = UInt8(8)

@inline _max_block(::_NarrowFamily) = UInt64(0x00ffffffffffffff)
@inline _max_block(::_Position64Family) = typemax(UInt64)

@inline _is_exhausted(position::_Position64) = position.lane == _EXHAUSTED_LANE
@inline _is_exhausted(position::_Position128) = position.lane == _EXHAUSTED_LANE

@inline _terminal64(maximum::UInt64) = _Position64(maximum, _EXHAUSTED_LANE)
@inline _terminal128() = _Position128(typemax(UInt64), typemax(UInt64), _EXHAUSTED_LANE)

@inline function _try_advance(
    position::_Position64,
    words::UInt64,
    width::UInt8,
    maximum::UInt64,
)
    words == 0 && return position, true
    terminal = _terminal64(maximum)
    _is_exhausted(position) && return terminal, false

    block = position.block
    lane = UInt64(position.lane)
    block_width = UInt64(width)
    in_block = block_width - lane
    if words < in_block
        return _Position64(block, UInt8(lane + words)), true
    elseif words == in_block
        block == maximum && return terminal, true
        return _Position64(block + 1, 0), true
    end

    block == maximum && return terminal, false
    remaining = words - in_block
    first_block = block + 1
    full_blocks, next_lane = divrem(remaining, block_width)
    blocks_left = maximum - first_block
    if full_blocks <= blocks_left
        return _Position64(first_block + full_blocks, UInt8(next_lane)), true
    elseif full_blocks == blocks_left + 1 && next_lane == 0
        return terminal, true
    end
    return terminal, false
end

@inline function _try_advance(
    position::_Position128,
    words::UInt64,
    width::UInt8,
    ::Nothing,
)
    words == 0 && return position, true
    terminal = _terminal128()
    _is_exhausted(position) && return terminal, false

    if position.hi == typemax(UInt64)
        next, ok = _try_advance(
            _Position64(position.lo, position.lane),
            words,
            width,
            typemax(UInt64),
        )
        _is_exhausted(next) && return terminal, ok
        return _Position128(next.block, position.hi, next.lane), ok
    end

    block_width = UInt64(width)
    blocks, remainder = divrem(words, block_width)
    lane_sum = UInt64(position.lane) + remainder
    blocks += lane_sum >= block_width
    next_lane = UInt8(ifelse(lane_sum >= block_width, lane_sum - block_width, lane_sum))
    next_lo = position.lo + blocks
    carry = next_lo < position.lo
    return _Position128(next_lo, position.hi + UInt64(carry), next_lane), true
end

@inline _try_advance(rng::_Position64Family, words::UInt64) =
    _try_advance(rng.position, words, _words_per_block(rng), _max_block(rng))
@inline _try_advance(rng::_Position128Family, words::UInt64) =
    _try_advance(rng.position, words, _words_per_block(rng), nothing)

@inline function _reserve(rng::AbstractPureRNG, words::UInt64)
    words == 0 && return rng
    position, ok = _try_advance(rng, words)
    ok || throw(ArgumentError("draw exceeds the generator counter capacity"))
    return _rebuild(rng, position, rng.device)
end

@inline _low_block(position::_Position64) = position.block
@inline _low_block(position::_Position128) = position.lo

@inline function _alignment_padding(rng::AbstractPureRNG, alignment::UInt64)
    mask = alignment - UInt64(1)
    position = rng.position
    block_residue = _low_block(position) & mask
    width_residue = UInt64(_words_per_block(rng)) & mask
    residue = (block_residue * width_residue + UInt64(position.lane)) & mask
    return (alignment - residue) & mask
end

@inline function _reserve_aligned(
    rng::AbstractPureRNG,
    words::UInt64,
    alignment::UInt64,
)
    words == 0 && return rng, rng
    (alignment == 1 || alignment == 2 || alignment == 4) ||
        throw(ArgumentError("alignment must be one, two, or four logical words"))

    padding = _alignment_padding(rng, alignment)
    start_position, padding_ok = _try_advance(rng, padding)
    padding_ok || throw(ArgumentError("draw exceeds the generator counter capacity"))
    start = _rebuild(rng, start_position, rng.device)
    next_position, draw_ok = _try_advance(start, words)
    draw_ok || throw(ArgumentError("draw exceeds the generator counter capacity"))
    return start, _rebuild(rng, next_position, rng.device)
end
