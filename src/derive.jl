const _DERIVE_TAG = UInt32(0xc0ffee00)
const _SPLIT_SUBTAG = UInt32(0)
const _FOLD_SUBTAG = UInt32(1)
const _THREEFRY_FOLD_INDEX = UInt32(0xffffffff)
const _NARROW_SPLIT_COUNT = UInt64(0xffffffff)

for F in _FAMILY_SYMBOLS
    @eval @inline _derived_rng(rng::$F{D}, key) where {D} =
        $F{D}(_CONSTRUCTION_TOKEN, key, _zero_position($F), rng.device)
end

@inline function _narrow_index(index::UInt64)
    index < _NARROW_SPLIT_COUNT ||
        throw(ArgumentError("a narrow-family child index enters the fold namespace"))
    return index % UInt32
end

@inline function _derive_child(rng::Philox2x32, index::UInt64)
    counter = (_narrow_index(index), _DERIVE_TAG)
    block = _philox2x32(counter, rng.key)
    return _derived_rng(rng, (block[1],))
end

@inline function _derive_child(rng::Threefry2x32, index::UInt64)
    counter = (_narrow_index(index), _DERIVE_TAG)
    return _derived_rng(rng, _threefry2x32(counter, rng.key))
end

@inline function _derive_child(rng::Philox4x32, index::UInt64)
    block_index, group = divrem(index, UInt64(2))
    counter =
        (block_index % UInt32, (block_index >> 32) % UInt32, _SPLIT_SUBTAG, _DERIVE_TAG)
    block = _philox4x32(counter, rng.key)
    offset = Int(group << 1)
    return _derived_rng(rng, (block[offset+1], block[offset+2]))
end

@inline function _derive_child(rng::Threefry4x32, index::UInt64)
    counter = (index % UInt32, (index >> 32) % UInt32, _SPLIT_SUBTAG, _DERIVE_TAG)
    return _derived_rng(rng, _threefry4x32(counter, rng.key))
end

@inline function _derive_child(rng::Philox2x64, index::UInt64)
    block_index, group = divrem(index, UInt64(2))
    tag = (UInt64(_DERIVE_TAG) << 32) | UInt64(_SPLIT_SUBTAG)
    block = _philox2x64((block_index, tag), rng.key)
    return _derived_rng(rng, (block[Int(group)+1],))
end

@inline function _derive_child(rng::Threefry2x64, index::UInt64)
    tag = (UInt64(_DERIVE_TAG) << 32) | UInt64(_SPLIT_SUBTAG)
    return _derived_rng(rng, _threefry2x64((index, tag), rng.key))
end

@inline function _derive_child(rng::Philox4x64, index::UInt64)
    block_index, group = divrem(index, UInt64(2))
    counter = (block_index, UInt64(_SPLIT_SUBTAG), UInt64(0), UInt64(_DERIVE_TAG))
    block = _philox4x64(counter, rng.key)
    offset = Int(group << 1)
    return _derived_rng(rng, (block[offset+1], block[offset+2]))
end

@inline function _derive_child(rng::Threefry4x64, index::UInt64)
    counter = (index, UInt64(_SPLIT_SUBTAG), UInt64(0), UInt64(_DERIVE_TAG))
    return _derived_rng(rng, _threefry4x64(counter, rng.key))
end

@inline _check_split_count(::_NarrowFamily, count::Integer) =
    count <= _NARROW_SPLIT_COUNT ||
    throw(ArgumentError("a narrow-family child index enters the fold namespace"))
@inline _check_split_count(::AbstractPureRNG, ::Integer) = nothing

"""
    splitrng(rng)
    splitrng(rng, n)
    splitrng(rng, Val(N))

Derive child keys from distinct counter addresses. The ordinary `n` form
returns a vector. The `Val` form returns an allocation-free tuple for static or
GPU code. The default derives two children.

Derivation reads only the parent key. It ignores the parent position, preserves
the device, and starts each child at position zero. It never changes the parent.

Child keys are core output and can collide. Across `n` program-wide derivations
with `k` key bits, the collision probability is about `n^2 / 2^(k+1)`. A
collision makes both child subtrees identical. Use a family with at least 128
key bits for per-particle or per-proposal derivation at scale.
"""
splitrng(rng::AbstractPureRNG) = splitrng(rng, Val(2))

@inline function splitrng(rng::R, ::Val{N}) where {R<:AbstractPureRNG,N}
    (N isa Int && N >= 0) || throw(ArgumentError("N must be a non-negative Int"))
    _check_split_count(rng, N)
    return ntuple(i -> _derive_child(rng, UInt64(i - 1)), Val(N))
end

function splitrng(rng::R, count::Integer) where {R<:AbstractPureRNG}
    0 <= count <= typemax(Int) ||
        throw(ArgumentError("n must satisfy 0 <= n <= typemax(Int)"))
    _check_split_count(rng, count)
    children = Vector{R}(undef, count)
    for i in eachindex(children)
        @inbounds children[i] = _derive_child(rng, UInt64(i - 1))
    end
    return children
end

@inline function _subrng(rng::Philox2x32, purpose::UInt64)
    namespace_block = _philox2x32((_THREEFRY_FOLD_INDEX, _DERIVE_TAG), rng.key)
    namespace_key = (namespace_block[1],)
    counter = (purpose % UInt32, (purpose >> 32) % UInt32)
    block = _philox2x32(counter, namespace_key)
    return _derived_rng(rng, (block[1],))
end

@inline function _subrng(rng::Threefry2x32, purpose::UInt64)
    namespace_key = _threefry2x32((_THREEFRY_FOLD_INDEX, _DERIVE_TAG), rng.key)
    counter = (purpose % UInt32, (purpose >> 32) % UInt32)
    return _derived_rng(rng, _threefry2x32(counter, namespace_key))
end

@inline function _subrng(rng::Philox4x32, purpose::UInt64)
    counter = (purpose % UInt32, (purpose >> 32) % UInt32, _FOLD_SUBTAG, _DERIVE_TAG)
    block = _philox4x32(counter, rng.key)
    return _derived_rng(rng, (block[1], block[2]))
end

@inline function _subrng(rng::Threefry4x32, purpose::UInt64)
    counter = (purpose % UInt32, (purpose >> 32) % UInt32, _FOLD_SUBTAG, _DERIVE_TAG)
    return _derived_rng(rng, _threefry4x32(counter, rng.key))
end

@inline function _subrng(rng::Philox2x64, purpose::UInt64)
    tag = (UInt64(_DERIVE_TAG) << 32) | UInt64(_FOLD_SUBTAG)
    block = _philox2x64((purpose, tag), rng.key)
    return _derived_rng(rng, (block[1],))
end

@inline function _subrng(rng::Threefry2x64, purpose::UInt64)
    tag = (UInt64(_DERIVE_TAG) << 32) | UInt64(_FOLD_SUBTAG)
    return _derived_rng(rng, _threefry2x64((purpose, tag), rng.key))
end

@inline function _subrng(rng::Philox4x64, purpose::UInt64)
    counter = (purpose, UInt64(_FOLD_SUBTAG), UInt64(0), UInt64(_DERIVE_TAG))
    block = _philox4x64(counter, rng.key)
    return _derived_rng(rng, (block[1], block[2]))
end

@inline function _subrng(rng::Threefry4x64, purpose::UInt64)
    counter = (purpose, UInt64(_FOLD_SUBTAG), UInt64(0), UInt64(_DERIVE_TAG))
    return _derived_rng(rng, _threefry4x64(counter, rng.key))
end

"""
    subrng(rng, purpose)

Derive one child key for the stable integer `purpose`. Use fixed purpose ids for
independent roles, such as `subrng(root, 1)` for proposals and
`subrng(root, 2)` for resampling. Use `subrng(root, chunk_id)` to assign
explicit large-job chunks.

Derivation reads only the parent key. It ignores the parent position, preserves
the device, and starts the child at position zero. It never changes the parent.
The same key and purpose always produce the same child.

Child keys are core output and can collide. Across `n` program-wide derivations
with `k` key bits, the collision probability is about `n^2 / 2^(k+1)`. A
collision makes both child subtrees identical. Use a family with at least 128
key bits for per-particle or per-proposal derivation at scale.
"""
@inline subrng(rng::AbstractPureRNG, purpose::Integer) = _subrng(rng, purpose % UInt64)
