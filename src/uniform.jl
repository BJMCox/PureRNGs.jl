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

@inline function _raw32(rng::_ScalarUniform32Family)
    position = rng.position
    block = _block(rng, FAMILY_BITS, position.block)
    return _select_word(block, position.lane)
end

@inline function _raw64(rng::_ScalarUniform32Family)
    position = rng.position
    block = _block(rng, FAMILY_BITS, position.block)
    high = _select_word(block, position.lane)
    next_lane = position.lane + UInt8(1)
    low = if next_lane < _words_per_block(rng)
        _select_word(block, next_lane)
    else
        _select_word(_block(rng, FAMILY_BITS, position.block + UInt64(1)), UInt8(0))
    end
    return (UInt64(high) << 32) | UInt64(low)
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

function Random.rand(::AbstractPureRNG)
    throw(ArgumentError("untyped immutable draws are forbidden; use rand(rng, T)"))
end

@inline function _rand_scalar(rng::_ScalarUniform32Family, ::Type{T}) where {T}
    _reserve(rng, _draw_words(T))
    return _draw_unchecked(rng, T)
end

@inline rand_next(rng::_ScalarUniform32Family) = rand_next(rng, Float64)

@inline function _rand_next_scalar(rng::_ScalarUniform32Family, ::Type{T}) where {T}
    next_rng = _reserve(rng, _draw_words(T))
    return next_rng, _draw_unchecked(rng, T)
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline Random.rand(rng::_ScalarUniform32Family, ::Type{$T}) = _rand_scalar(rng, $T)
        @inline rand_next(rng::_ScalarUniform32Family, ::Type{$T}) =
            _rand_next_scalar(rng, $T)
    end
end
