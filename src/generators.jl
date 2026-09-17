"""
    AbstractPureRNG

The abstract supertype of PureRNGs' generators. It exists for dispatch and
inspection. Subtyping it outside the package is unsupported: the package's
methods assume the key, position, and device layout of its own generators.
"""
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

"""
    Philox2x32(seed)
    Philox2x32(key::NTuple{1,UInt32})
    Philox2x32(seed, position)
    Philox2x32(key, position)

Construct a Philox generator with a 32-bit key and two 32-bit output words per
counter block. The generator starts at the beginning of its stream on the CPU.

The integer `seed` must be non-negative and fit in 32 bits. The tuple form sets
the key exactly. The two-argument forms start at `position` consumed bits, as
returned by [`rngposition`](@ref). This generator's small key space makes
derived-key collisions likely beyond a few thousand program-wide derivations.
"""
struct Philox2x32{D<:_BackendToken,R} <: AbstractPureRNG
    key::NTuple{1,UInt32}
    position::_Position64
    device::D
    block_words::NTuple{1,UInt64}

    Philox2x32{D,R}(::_ConstructionToken, key, position, device, block_words) where {D,R} =
        new{D,R}(key, position, device, block_words)
end

"""
    Philox4x32(seed)
    Philox4x32(key::NTuple{2,UInt32})
    Philox4x32(seed, position)
    Philox4x32(key, position)

Construct a Philox generator with a 64-bit key and four 32-bit output words per
counter block. The generator starts at the beginning of its stream on the CPU.

The integer `seed` must be non-negative and fit in 64 bits. The tuple form sets
the key words exactly, from least to most significant. The two-argument forms
start at `position` consumed bits, as returned by [`rngposition`](@ref).
"""
struct Philox4x32{D<:_BackendToken,R} <: AbstractPureRNG
    key::NTuple{2,UInt32}
    position::_Position64
    device::D
    block_words::NTuple{2,UInt64}

    Philox4x32{D,R}(::_ConstructionToken, key, position, device, block_words) where {D,R} =
        new{D,R}(key, position, device, block_words)
end

"""
    Philox2x64(seed)
    Philox2x64(key::NTuple{1,UInt64})
    Philox2x64(seed, position)
    Philox2x64(key, position)

Construct a Philox generator with a 64-bit key and two 64-bit output words per
counter block. The generator starts at the beginning of its stream on the CPU.

The integer `seed` must be non-negative and fit in 64 bits. The tuple form sets
the key exactly. The two-argument forms start at `position` consumed bits, as
returned by [`rngposition`](@ref).
"""
struct Philox2x64{D<:_BackendToken,R} <: AbstractPureRNG
    key::NTuple{1,UInt64}
    position::_Position64
    device::D
    block_words::NTuple{2,UInt64}

    Philox2x64{D,R}(::_ConstructionToken, key, position, device, block_words) where {D,R} =
        new{D,R}(key, position, device, block_words)
end

"""
    Philox4x64(seed)
    Philox4x64(key::NTuple{2,UInt64})
    Philox4x64(seed, position)
    Philox4x64(key, position)

Construct a Philox generator with a 128-bit key and four 64-bit output words per
counter block. The generator starts at the beginning of its stream on the CPU.

The integer `seed` must be non-negative and fit in 128 bits. The tuple form sets
the key words exactly, from least to most significant. The two-argument forms
start at `position` consumed bits, as returned by [`rngposition`](@ref).
"""
struct Philox4x64{D<:_BackendToken,R} <: AbstractPureRNG
    key::NTuple{2,UInt64}
    position::_Position128
    device::D
    block_words::NTuple{4,UInt64}

    Philox4x64{D,R}(::_ConstructionToken, key, position, device, block_words) where {D,R} =
        new{D,R}(key, position, device, block_words)
end

"""
    Threefry2x32(seed)
    Threefry2x32(key::NTuple{2,UInt32})
    Threefry2x32(seed, position)
    Threefry2x32(key, position)

Construct a Threefry generator with a 64-bit key and two 32-bit output words per
counter block. The generator starts at the beginning of its stream on the CPU.

The integer `seed` must be non-negative and fit in 64 bits. The tuple form sets
the key words exactly, from least to most significant. The two-argument forms
start at `position` consumed bits, as returned by [`rngposition`](@ref).
"""
struct Threefry2x32{D<:_BackendToken,R} <: AbstractPureRNG
    key::NTuple{2,UInt32}
    position::_Position64
    device::D
    block_words::NTuple{1,UInt64}

    Threefry2x32{D,R}(
        ::_ConstructionToken,
        key,
        position,
        device,
        block_words,
    ) where {D,R} = new{D,R}(key, position, device, block_words)
end

"""
    Threefry4x32(seed)
    Threefry4x32(key::NTuple{4,UInt32})
    Threefry4x32(seed, position)
    Threefry4x32(key, position)

Construct a Threefry generator with a 128-bit key and four 32-bit output words
per counter block. The generator starts at the beginning of its stream on the
CPU.

The integer `seed` must be non-negative and fit in 128 bits. The tuple form sets
the key words exactly, from least to most significant. The two-argument forms
start at `position` consumed bits, as returned by [`rngposition`](@ref).
"""
struct Threefry4x32{D<:_BackendToken,R} <: AbstractPureRNG
    key::NTuple{4,UInt32}
    position::_Position64
    device::D
    block_words::NTuple{2,UInt64}

    Threefry4x32{D,R}(
        ::_ConstructionToken,
        key,
        position,
        device,
        block_words,
    ) where {D,R} = new{D,R}(key, position, device, block_words)
end

"""
    Threefry2x64(seed)
    Threefry2x64(key::NTuple{2,UInt64})
    Threefry2x64(seed, position)
    Threefry2x64(key, position)

Construct a Threefry generator with a 128-bit key and two 64-bit output words per
counter block. The generator starts at the beginning of its stream on the CPU.

The integer `seed` must be non-negative and fit in 128 bits. The tuple form sets
the key words exactly, from least to most significant. The two-argument forms
start at `position` consumed bits, as returned by [`rngposition`](@ref).
"""
struct Threefry2x64{D<:_BackendToken,R} <: AbstractPureRNG
    key::NTuple{2,UInt64}
    position::_Position64
    device::D
    block_words::NTuple{2,UInt64}

    Threefry2x64{D,R}(
        ::_ConstructionToken,
        key,
        position,
        device,
        block_words,
    ) where {D,R} = new{D,R}(key, position, device, block_words)
end

"""
    Threefry4x64(seed)
    Threefry4x64(key::NTuple{4,UInt64})
    Threefry4x64(seed, position)
    Threefry4x64(key, position)

Construct a Threefry generator with a 256-bit key and four 64-bit output words per
counter block. The generator starts at the beginning of its stream on the CPU.

The integer `seed` must be non-negative and fit in 256 bits. The tuple form sets
the key words exactly, from least to most significant. The two-argument forms
start at `position` consumed bits, as returned by [`rngposition`](@ref).
"""
struct Threefry4x64{D<:_BackendToken,R} <: AbstractPureRNG
    key::NTuple{4,UInt64}
    position::_Position128
    device::D
    block_words::NTuple{4,UInt64}

    Threefry4x64{D,R}(
        ::_ConstructionToken,
        key,
        position,
        device,
        block_words,
    ) where {D,R} = new{D,R}(key, position, device, block_words)
end

"""
    ChaCha(seed)
    ChaCha(key::NTuple{8,UInt32})
    ChaCha(seed, position)
    ChaCha(key, position)

Construct a ChaCha generator with a 256-bit key, twelve rounds, and sixteen
32-bit output words per counter block. The generator starts at the beginning of
its stream on the CPU. [`ChaCha8`](@ref) and [`ChaCha20`](@ref) select the
other common round counts.

The integer `seed` must be non-negative and fit in 256 bits. The tuple form sets
the key words exactly, from least to most significant. The two-argument forms
start at `position` consumed bits, as returned by [`rngposition`](@ref).
"""
struct ChaCha{D<:_BackendToken,R} <: AbstractPureRNG
    key::NTuple{8,UInt32}
    position::_Position64
    device::D
    block_words::NTuple{8,UInt64}

    ChaCha{D,R}(::_ConstructionToken, key, position, device, block_words) where {D,R} =
        new{D,R}(key, position, device, block_words)
end

const _GENERATOR_SYMBOLS = (
    :Philox2x32,
    :Philox4x32,
    :Philox2x64,
    :Philox4x64,
    :Threefry2x32,
    :Threefry4x32,
    :Threefry2x64,
    :Threefry4x64,
    :ChaCha,
)
const _Backend32Generators{D} =
    Union{Philox2x32{D},Philox4x32{D},Threefry2x32{D},Threefry4x32{D},ChaCha{D}}
const _Backend64Generators{D} =
    Union{Philox2x64{D},Philox4x64{D},Threefry2x64{D},Threefry4x64{D}}
const _BackendGenerators{D} = Union{_Backend32Generators{D},_Backend64Generators{D}}

const _Position64Generators =
    Union{Philox2x32,Philox4x32,Philox2x64,Threefry2x32,Threefry4x32,Threefry2x64,ChaCha}
const _Position128Generators = Union{Philox4x64,Threefry4x64}
const _NarrowGenerators = Union{Philox2x32,Threefry2x32}
@inline _zero_position(::Type{<:_Position64Generators}) = _Position64(0, 0)
@inline _zero_position(::Type{<:_Position128Generators}) = _Position128(0, 0, 0)

# Round counts. The second type parameter holds the count, the bare type name
# means the Random123 default, and the round-reduced names are aliases.
@inline _default_rounds(::Type{<:Union{Philox2x32,Philox4x32,Philox2x64,Philox4x64}}) =
    _PHILOX_DEFAULT_ROUNDS
@inline _default_rounds(
    ::Type{<:Union{Threefry2x32,Threefry4x32,Threefry2x64,Threefry4x64}},
) = _THREEFRY_DEFAULT_ROUNDS
@inline _default_rounds(::Type{<:ChaCha}) = _CHACHA_DEFAULT_ROUNDS

for F in _GENERATOR_SYMBOLS
    @eval begin
        @inline _rounds(::Type{<:$F{<:_BackendToken,R}}) where {R} = R
        @inline _with_device(::Type{<:$F{<:_BackendToken,R}}, ::Type{D}) where {R,D} =
            $F{D,R}

        function $F(key::fieldtype($F, :key))
            return $F{_CPUBackend,_default_rounds($F)}(
                _CONSTRUCTION_TOKEN,
                key,
                _zero_position($F),
                _CPU_BACKEND,
            )
        end
    end
end

"""
    Philox4x32R7(seed)
    Philox4x32R7(key::NTuple{2,UInt32})

`Philox4x32` with seven rounds, the smallest round count that passes BigCrush
in Random123. It shares every method with `Philox4x32` and produces a different
stream.
"""
const Philox4x32R7 = Philox4x32{D,7} where {D<:_BackendToken}

"""
    Threefry4x64R13(seed)
    Threefry4x64R13(key::NTuple{4,UInt64})

`Threefry4x64` with thirteen rounds, the smallest round count that passes
BigCrush in Random123. It shares every method with `Threefry4x64` and produces a
different stream.
"""
const Threefry4x64R13 = Threefry4x64{D,13} where {D<:_BackendToken}

"""
    ChaCha8(seed)
    ChaCha8(key::NTuple{8,UInt32})

[`ChaCha`](@ref) with eight rounds, the fastest common choice for random number
generation. It shares every method with `ChaCha` and produces a different
stream.
"""
const ChaCha8 = ChaCha{D,8} where {D<:_BackendToken}

"""
    ChaCha12(seed)
    ChaCha12(key::NTuple{8,UInt32})

[`ChaCha`](@ref) with its default twelve rounds, the round count with a
comfortable margin over the best known attacks. `ChaCha12(seed)` and
`ChaCha(seed)` are the same generator.
"""
const ChaCha12 = ChaCha{D,12} where {D<:_BackendToken}

"""
    ChaCha20(seed)
    ChaCha20(key::NTuple{8,UInt32})

[`ChaCha`](@ref) with twenty rounds, the cipher's round count. It shares every
method with `ChaCha` and produces a different stream.
"""
const ChaCha20 = ChaCha{D,20} where {D<:_BackendToken}

const _ROUND_ALIASES = (
    (:Philox4x32R7, :Philox4x32, 7),
    (:Threefry4x64R13, :Threefry4x64, 13),
    (:ChaCha8, :ChaCha, 8),
    (:ChaCha12, :ChaCha, 12),
    (:ChaCha20, :ChaCha, 20),
)
for (alias, F, R) in _ROUND_ALIASES
    @eval begin
        function (::Type{$alias})(key::fieldtype($F, :key))
            position = _zero_position($F)
            return $F{_CPUBackend,$R}(_CONSTRUCTION_TOKEN, key, position, _CPU_BACKEND)
        end
        (::Type{$alias})(seed::Integer) = $alias(_family_key($F, seed))
        function (::Type{$alias})(key::fieldtype($F, :key), position::Integer)
            return $F{_CPUBackend,$R}(
                _CONSTRUCTION_TOKEN,
                key,
                _position_from_bits($F, position),
                _CPU_BACKEND,
            )
        end
        (::Type{$alias})(seed::Integer, position::Integer) =
            $alias(_family_key($F, seed), position)
    end
end

function _seed_key(::Type{T}, ::Val{N}, seed::Integer) where {T<:Unsigned,N}
    seed < 0 && throw(ArgumentError("seed must be non-negative"))
    bits = 8 * sizeof(T)
    value = seed isa Base.BitInteger ? unsigned(seed) : BigInt(seed)
    iszero(value >> (bits * N)) || throw(ArgumentError("seed exceeds the key width"))
    return ntuple(i -> (value >> (bits * (i - 1))) % T, Val(N))
end

@inline _family_key(::Type{<:Philox2x32}, seed::Integer) = _seed_key(UInt32, Val(1), seed)
@inline _family_key(::Type{<:Philox4x32}, seed::Integer) = _seed_key(UInt32, Val(2), seed)
@inline _family_key(::Type{<:Philox2x64}, seed::Integer) = _seed_key(UInt64, Val(1), seed)
@inline _family_key(::Type{<:Philox4x64}, seed::Integer) = _seed_key(UInt64, Val(2), seed)
@inline _family_key(::Type{<:Threefry2x32}, seed::Integer) = _seed_key(UInt32, Val(2), seed)
@inline _family_key(::Type{<:Threefry4x32}, seed::Integer) = _seed_key(UInt32, Val(4), seed)
@inline _family_key(::Type{<:Threefry2x64}, seed::Integer) = _seed_key(UInt64, Val(2), seed)
@inline _family_key(::Type{<:Threefry4x64}, seed::Integer) = _seed_key(UInt64, Val(4), seed)
@inline _family_key(::Type{<:ChaCha}, seed::Integer) = _seed_key(UInt32, Val(8), seed)

for F in _GENERATOR_SYMBOLS
    @eval $F(seed::Integer) = $F(_family_key($F, seed))
end

# Every generator carries the decoded block at its position, so equal key and
# position always give identical values. The four-argument token constructor
# decodes the block; `_rebuild` keeps the carried block while the address holds.
for F in _GENERATOR_SYMBOLS
    @eval begin
        @inline function $F{D,R}(
            token::_ConstructionToken,
            key,
            position,
            device::D,
        ) where {D,R}
            words = _decoded_block_words($F{D,R}, key, position)
            return $F{D,R}(token, key, position, device, words)
        end
    end
end

@inline function _rebuild(
    rng::AbstractPureRNG,
    position,
    device::D,
) where {D<:_BackendToken}
    target = _with_device(typeof(rng), D)
    key = _by_element(rng.key)
    if _position_block(position) == _position_block(rng.position)
        words = _by_element(rng.block_words)
        return target(_CONSTRUCTION_TOKEN, key, position, device, words)
    end
    return target(_CONSTRUCTION_TOKEN, key, position, device)
end

for (Device, token) in (
    (MLDataDevices.CPUDevice, :_CPU_BACKEND),
    (MLDataDevices.CUDADevice, :_CUDA_BACKEND),
    (MLDataDevices.AMDGPUDevice, :_AMDGPU_BACKEND),
    (MLDataDevices.MetalDevice, :_METAL_BACKEND),
)
    @eval @inline (::$(Device))(rng::AbstractPureRNG) = _rebuild(rng, rng.position, $token)
end

@noinline function _unsupported_device(device)
    throw(ArgumentError("unsupported MLDataDevices device type: $(typeof(device))"))
end

@inline (device::MLDataDevices.AbstractDevice)(::AbstractPureRNG) =
    _unsupported_device(device)

@inline _block_shift(::Type{<:_NarrowGenerators}) = UInt8(6)
@inline _block_shift(::Type{<:Union{Philox4x32,Philox2x64,Threefry4x32,Threefry2x64}}) =
    UInt8(7)
@inline _block_shift(::Type{<:_Position128Generators}) = UInt8(8)
@inline _block_shift(::Type{<:ChaCha}) = UInt8(9)
@inline _block_shift(rng::AbstractPureRNG) = _block_shift(typeof(rng))
@inline _block_bits(rng::AbstractPureRNG) = UInt16(1) << _block_shift(rng)

@inline _max_block(::Type{<:_NarrowGenerators}) = UInt64(0x00ffffffffffffff)
@inline _max_block(::Type{<:_Position64Generators}) = typemax(UInt64)
@inline _max_block(rng::_Position64Generators) = _max_block(typeof(rng))

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

@inline _try_advance(rng::_Position64Generators, bits_lo::UInt64, bits_hi::UInt64) =
    _try_advance(rng.position, bits_lo, bits_hi, _block_shift(rng), _max_block(rng))
@inline _try_advance(rng::_Position128Generators, bits_lo::UInt64, bits_hi::UInt64) =
    _try_advance(rng.position, bits_lo, bits_hi, _block_shift(rng), nothing)

@inline function _reserve(rng::AbstractPureRNG, bits_lo::UInt64, bits_hi::UInt64)
    position, ok = _try_advance(rng, bits_lo, bits_hi)
    ok || throw(ArgumentError("draw exceeds the generator counter capacity"))
    position == rng.position && return rng
    return _rebuild(rng, position, rng.device)
end

@inline _with_bit(position::_Position64, bit::UInt16) = _Position64(position.block, bit)
@inline _with_bit(position::_Position128, bit::UInt16) =
    _Position128(position.lo, position.hi, bit)

# Copying a tuple field whole into the successor leaves LLVM a memory blob
# that it copies on every draw of a chained loop once the generator has more
# than a handful of words. Rebuilding the tuple from its elements keeps every
# word in a register. A ChaCha chain of Bool draws ran five times faster.
@inline _by_element(words::NTuple{N,T}) where {N,T} = ntuple(i -> words[i], Val(N))

# A scalar draw that ends inside the current block cannot exhaust the stream
# and keeps the carried block, so only the bit offset moves. The terminal
# offset fails the width test in 32-bit arithmetic and takes the checked path.
@inline function _reserve_scalar(rng::AbstractPureRNG, width::UInt16)
    bit = rng.position.bit
    if UInt32(bit) + UInt32(width) < UInt32(_block_bits(rng))
        return typeof(rng)(
            _CONSTRUCTION_TOKEN,
            _by_element(rng.key),
            _with_bit(rng.position, bit + width),
            rng.device,
            _by_element(rng.block_words),
        )
    end
    return _reserve(rng, UInt64(width), UInt64(0))
end

# Stream capacity in bits. It is the position of the terminal state.
@inline _stream_capacity(::Type{F}) where {F<:_Position64Generators} =
    (UInt128(_max_block(F)) + UInt128(1)) << _block_shift(F)
@inline _stream_capacity(::Type{F}) where {F<:_Position128Generators} =
    BigInt(1) << (128 + Int(_block_shift(F)))

"""
    rngkey(rng) -> NTuple

Return the key words of `rng`, from least to most significant. The tuple has
the type accepted by the generator's key constructor.
"""
@inline rngkey(rng::AbstractPureRNG) = rng.key

"""
    rngposition(rng) -> Integer

Return the number of stream bits `rng` has consumed. The result is a `UInt128`
for generators with a 64-bit block counter and a `BigInt` for `Philox4x64` and
`Threefry4x64`. An exhausted generator reports the generator's full stream
capacity.

Pass the result to the generator's constructor together with [`rngkey`](@ref) to
rebuild the generator at the same position.
"""
function rngposition(rng::_Position64Generators)
    F = typeof(rng)
    _is_terminal(rng.position, _max_block(F)) && return _stream_capacity(F)
    return (UInt128(rng.position.block) << _block_shift(F)) | UInt128(rng.position.bit)
end

function rngposition(rng::_Position128Generators)
    F = typeof(rng)
    _is_terminal(rng.position, nothing) && return _stream_capacity(F)
    block = (BigInt(rng.position.hi) << 64) | BigInt(rng.position.lo)
    return (block << _block_shift(F)) | BigInt(rng.position.bit)
end

function _checked_position_bits(::Type{F}, bits::Integer) where {F}
    bits < 0 && throw(ArgumentError("position must be non-negative"))
    bits <= _stream_capacity(F) ||
        throw(ArgumentError("position exceeds the stream capacity"))
    return bits
end

function _position_from_bits(::Type{F}, bits::Integer) where {F<:_Position64Generators}
    value = UInt128(_checked_position_bits(F, bits))
    value == _stream_capacity(F) && return _terminal64(_max_block(F))
    shift = _block_shift(F)
    return _Position64(UInt64(value >> shift), UInt16(value & ((UInt128(1) << shift) - 1)))
end

function _position_from_bits(::Type{F}, bits::Integer) where {F<:_Position128Generators}
    value = BigInt(_checked_position_bits(F, bits))
    value == _stream_capacity(F) && return _terminal128()
    shift = _block_shift(F)
    block = value >> shift
    return _Position128(
        UInt64(block & typemax(UInt64)),
        UInt64(block >> 64),
        UInt16(value & ((BigInt(1) << shift) - 1)),
    )
end

for F in _GENERATOR_SYMBOLS
    @eval begin
        function $F(key::fieldtype($F, :key), position::Integer)
            return $F{_CPUBackend,_default_rounds($F)}(
                _CONSTRUCTION_TOKEN,
                key,
                _position_from_bits($F, position),
                _CPU_BACKEND,
            )
        end
        $F(seed::Integer, position::Integer) = $F(_family_key($F, seed), position)
    end
end
