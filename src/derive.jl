const _DERIVE_TAG = UInt32(0xc0ffee00)
const _SPLIT_SUBTAG = UInt32(0)
const _FOLD_SUBTAG = UInt32(1)
const _THREEFRY_FOLD_INDEX = UInt32(0xffffffff)
const _NARROW_SPLIT_COUNT = UInt64(0xffffffff)
const _SPLIT_TAG64 = (UInt64(_DERIVE_TAG) << 32) | UInt64(_SPLIT_SUBTAG)
const _FOLD_TAG64 = (UInt64(_DERIVE_TAG) << 32) | UInt64(_FOLD_SUBTAG)

@inline _derived_rng(rng::AbstractPureRNG, key) =
    typeof(rng)(_CONSTRUCTION_TOKEN, key, _zero_position(typeof(rng)), rng.device)

@inline function _narrow_index(index::UInt64)
    index < _NARROW_SPLIT_COUNT ||
        throw(ArgumentError("two-word generator child index enters the fold namespace"))
    return index % UInt32
end

@inline function _derive_key(::Type{F}, key, index::UInt64) where {F<:Philox2x32}
    counter =
        (_core_constant(key[1], _narrow_index(index)), _core_constant(key[1], _DERIVE_TAG))
    return (_philox2x32(counter, key, Val(_rounds(F)))[1],)
end

@inline function _derive_key(::Type{F}, key, index::UInt64) where {F<:Threefry2x32}
    counter =
        (_core_constant(key[1], _narrow_index(index)), _core_constant(key[1], _DERIVE_TAG))
    return _threefry2x32(counter, key, Val(_rounds(F)))
end

@inline function _derive_key(::Type{F}, key, index::UInt64) where {F<:Philox4x32}
    block_index, group = divrem(index, UInt64(2))
    counter = (
        _core_constant(key[1], block_index % UInt32),
        _core_constant(key[1], (block_index >> 32) % UInt32),
        _core_constant(key[1], _SPLIT_SUBTAG),
        _core_constant(key[1], _DERIVE_TAG),
    )
    block = _philox4x32(counter, key, Val(_rounds(F)))
    offset = Int(group << 1)
    return block[offset+1], block[offset+2]
end

@inline function _derive_key(::Type{F}, key, index::UInt64) where {F<:Threefry4x32}
    counter = (
        _core_constant(key[1], index % UInt32),
        _core_constant(key[1], (index >> 32) % UInt32),
        _core_constant(key[1], _SPLIT_SUBTAG),
        _core_constant(key[1], _DERIVE_TAG),
    )
    return _threefry4x32(counter, key, Val(_rounds(F)))
end

@inline function _derive_key(::Type{F}, key, index::UInt64) where {F<:Philox2x64}
    block_index, group = divrem(index, UInt64(2))
    counter = (_core_constant(key[1], block_index), _core_constant(key[1], _SPLIT_TAG64))
    block = _philox2x64(counter, key, Val(_rounds(F)))
    return (block[Int(group)+1],)
end

@inline function _derive_key(::Type{F}, key, index::UInt64) where {F<:Threefry2x64}
    counter = (_core_constant(key[1], index), _core_constant(key[1], _SPLIT_TAG64))
    return _threefry2x64(counter, key, Val(_rounds(F)))
end

@inline function _derive_key(::Type{F}, key, index::UInt64) where {F<:Philox4x64}
    block_index, group = divrem(index, UInt64(2))
    counter = (
        _core_constant(key[1], block_index),
        _core_constant(key[1], _SPLIT_SUBTAG),
        _core_constant(key[1], 0),
        _core_constant(key[1], _DERIVE_TAG),
    )
    block = _philox4x64(counter, key, Val(_rounds(F)))
    offset = Int(group << 1)
    return block[offset+1], block[offset+2]
end

@inline function _derive_key(::Type{F}, key, index::UInt64) where {F<:Threefry4x64}
    counter = (
        _core_constant(key[1], index),
        _core_constant(key[1], _SPLIT_SUBTAG),
        _core_constant(key[1], 0),
        _core_constant(key[1], _DERIVE_TAG),
    )
    return _threefry4x64(counter, key, Val(_rounds(F)))
end

@inline function _derive_key(::Type{F}, key, index::UInt64) where {F<:ChaCha}
    block_index, group = divrem(index, UInt64(2))
    counter = (
        _core_constant(key[1], block_index % UInt32),
        _core_constant(key[1], (block_index >> 32) % UInt32),
        _core_constant(key[1], _SPLIT_SUBTAG),
        _core_constant(key[1], _DERIVE_TAG),
    )
    block = _chacha(counter, key, Val(_rounds(F)))
    offset = Int(group << 3)
    return ntuple(i -> block[offset+i], Val(8))
end

@inline _derive_child(rng::AbstractPureRNG, index::UInt64) =
    _derived_rng(rng, _derive_key(typeof(rng), rng.key, index))

@inline _check_split_count(::_NarrowGenerators, count::Integer) =
    count <= _NARROW_SPLIT_COUNT ||
    throw(ArgumentError("two-word generator child index enters the fold namespace"))
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
Calling `splitrng` again on an advanced parent therefore returns the same
children, so derive from stable identifiers rather than from a stream position.

Child keys are core output and can collide. Across `n` program-wide derivations
with `k` key bits, the collision probability is about `n^2 / 2^(k+1)`. A
collision makes both child subtrees identical. Use a generator with at least 128
key bits for per-particle or per-proposal derivation at scale.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> left, right = splitrng(rng);

julia> rand(left, UInt32), rand(right, UInt32)
(0xea1624f3, 0x34564a30)

julia> advanced = last(rand_next(rng, UInt32));

julia> rand(first(splitrng(advanced)), UInt32)
0xea1624f3
```
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

@inline function _subrng_key(::Type{F}, key, purpose) where {F<:Philox2x32}
    namespace_counter =
        (_core_constant(key[1], _THREEFRY_FOLD_INDEX), _core_constant(key[1], _DERIVE_TAG))
    namespace_block = _philox2x32(namespace_counter, key, Val(_rounds(F)))
    namespace_key = (namespace_block[1],)
    counter = (
        _core_from_value(namespace_key[1], purpose),
        _core_from_value(namespace_key[1], div(purpose, UInt64(1) << 32)),
    )
    block = _philox2x32(counter, namespace_key, Val(_rounds(F)))
    return (block[1],)
end

@inline function _subrng_key(::Type{F}, key, purpose) where {F<:Threefry2x32}
    namespace_counter =
        (_core_constant(key[1], _THREEFRY_FOLD_INDEX), _core_constant(key[1], _DERIVE_TAG))
    namespace_key = _threefry2x32(namespace_counter, key, Val(_rounds(F)))
    counter = (
        _core_from_value(namespace_key[1], purpose),
        _core_from_value(namespace_key[1], div(purpose, UInt64(1) << 32)),
    )
    return _threefry2x32(counter, namespace_key, Val(_rounds(F)))
end

@inline function _subrng_key(::Type{F}, key, purpose) where {F<:Philox4x32}
    counter = (
        _core_from_value(key[1], purpose),
        _core_from_value(key[1], div(purpose, UInt64(1) << 32)),
        _core_constant(key[1], _FOLD_SUBTAG),
        _core_constant(key[1], _DERIVE_TAG),
    )
    block = _philox4x32(counter, key, Val(_rounds(F)))
    return block[1], block[2]
end

@inline function _subrng_key(::Type{F}, key, purpose) where {F<:Threefry4x32}
    counter = (
        _core_from_value(key[1], purpose),
        _core_from_value(key[1], div(purpose, UInt64(1) << 32)),
        _core_constant(key[1], _FOLD_SUBTAG),
        _core_constant(key[1], _DERIVE_TAG),
    )
    return _threefry4x32(counter, key, Val(_rounds(F)))
end

@inline function _subrng_key(::Type{F}, key, purpose) where {F<:Philox2x64}
    counter = (_core_from_value(key[1], purpose), _core_constant(key[1], _FOLD_TAG64))
    block = _philox2x64(counter, key, Val(_rounds(F)))
    return (block[1],)
end

@inline function _subrng_key(::Type{F}, key, purpose) where {F<:Threefry2x64}
    counter = (_core_from_value(key[1], purpose), _core_constant(key[1], _FOLD_TAG64))
    return _threefry2x64(counter, key, Val(_rounds(F)))
end

@inline function _subrng_key(::Type{F}, key, purpose) where {F<:Philox4x64}
    counter = (
        _core_from_value(key[1], purpose),
        _core_constant(key[1], _FOLD_SUBTAG),
        _core_constant(key[1], 0),
        _core_constant(key[1], _DERIVE_TAG),
    )
    block = _philox4x64(counter, key, Val(_rounds(F)))
    return block[1], block[2]
end

@inline function _subrng_key(::Type{F}, key, purpose) where {F<:Threefry4x64}
    counter = (
        _core_from_value(key[1], purpose),
        _core_constant(key[1], _FOLD_SUBTAG),
        _core_constant(key[1], 0),
        _core_constant(key[1], _DERIVE_TAG),
    )
    return _threefry4x64(counter, key, Val(_rounds(F)))
end

@inline function _subrng_key(::Type{F}, key, purpose) where {F<:ChaCha}
    counter = (
        _core_from_value(key[1], purpose),
        _core_from_value(key[1], div(purpose, UInt64(1) << 32)),
        _core_constant(key[1], _FOLD_SUBTAG),
        _core_constant(key[1], _DERIVE_TAG),
    )
    block = _chacha(counter, key, Val(_rounds(F)))
    return ntuple(i -> block[i], Val(8))
end

@inline _subrng(rng::AbstractPureRNG, purpose::UInt64) =
    _derived_rng(rng, _subrng_key(typeof(rng), rng.key, purpose))

"""
    subrng(rng, purpose)

Derive one child key for the stable integer `purpose`. Use fixed purpose ids for
independent roles, such as `subrng(root, 1)` for proposals and
`subrng(root, 2)` for resampling. Use `subrng(root, chunk_id)` to assign
explicit large-job chunks.

`purpose` is reduced modulo `2^64`, so a negative value or a value at or above
`2^64` aliases the child of an existing purpose id. Purpose ids are a separate
namespace from stream positions, which [`rngposition`](@ref) returns as a
`UInt128`.

Derivation reads only the parent key. It ignores the parent position, preserves
the device, and starts the child at position zero. It never changes the parent.
The same key and purpose always produce the same child. Calling `subrng` again
on an advanced parent therefore returns the same child, so derive from stable
identifiers rather than from a stream position.

Child keys are core output and can collide. Across `n` program-wide derivations
with `k` key bits, the collision probability is about `n^2 / 2^(k+1)`. A
collision makes both child subtrees identical. Use a generator with at least 128
key bits for per-particle or per-proposal derivation at scale.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> rand(subrng(rng, 1), UInt32)
0x04970499

julia> rand(subrng(rng, 2), UInt32)
0x65feaf98

julia> rand(subrng(rng, -1), UInt32) == rand(subrng(rng, big(2)^64 - 1), UInt32)
true
```
"""
@inline subrng(rng::AbstractPureRNG, purpose::Integer) = _subrng(rng, purpose % UInt64)
