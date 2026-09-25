# A permutation orders `n` uniform 64-bit keys, drawn at the held position. The
# ordering is the same on every backend, so device results equal CPU results.
# Equal keys (probability about n^2 / 2^65) are shuffled within their run with
# draws from a child of the key's value, which keeps the permutation exactly
# uniform and the consumption at 64n bits.

@noinline _permutation_length_error(n) =
    throw(ArgumentError("permutation length must be non-negative, got $n"))

@inline function _permutation_keys(rng, n::Integer, threaded::Bool)
    n < 0 && _permutation_length_error(n)
    return rand_next(rng, UInt64, Int(n); threaded)
end

# One run of equal keys takes index order first, so the result does not depend
# on the stability of the backend's sort. A permutation then shuffles the run by
# Fisher–Yates from `rng`, and a weighted race, with no `rng`, keeps index order.
# The loops index `permutation` in place, so a device kernel runs them as well.
@inline function _order_key_run!(permutation, first_slot, last_slot, rng)
    for slot = (first_slot+1):last_slot
        index = permutation[slot]
        previous = slot - 1
        while previous >= first_slot && permutation[previous] > index
            permutation[previous+1] = permutation[previous]
            previous -= 1
        end
        permutation[previous+1] = index
    end
    return _shuffle_key_run!(permutation, first_slot, last_slot, rng)
end

@inline _shuffle_key_run!(permutation, first_slot, last_slot, ::Nothing) = permutation
@inline function _shuffle_key_run!(permutation, first_slot, last_slot, rng)
    for last_index = (last_slot-first_slot+1):-1:2
        swap, rng = rand_next(rng, 1:last_index)
        a, b = first_slot + last_index - 1, first_slot + swap - 1
        permutation[a], permutation[b] = permutation[b], permutation[a]
    end
    return permutation
end

@inline _key_run_rng(rng, key) = subrng(rng, key)
@inline _key_run_rng(::Nothing, key) = nothing

# The keys are uniform, so a counting pass over their top bits leaves a few keys
# per bucket, and one insertion pass orders the buckets by the whole key. This
# is the stable order `sortperm!` gives, in expected linear time. Past 2^27 keys
# the buckets grow and the comparison sort takes over.
const _KEY_BUCKET_BITS = 24

function _order_keys!(permutation::AbstractVector{<:Integer}, keys::Vector{UInt64})
    n = length(keys)
    n > 1 << (_KEY_BUCKET_BITS + 3) && return sortperm!(permutation, keys)
    bits = min(_KEY_BUCKET_BITS, 64 - leading_zeros(max(n, 2) - 1))
    shift = 64 - bits
    starts = zeros(Int, (1 << bits) + 1)
    for key in keys
        starts[(key>>shift)+2] += 1
    end
    for bucket = 2:length(starts)
        starts[bucket] += starts[bucket-1]
    end
    ordered = Vector{UInt64}(undef, n)
    for index = 1:n
        key = keys[index]
        slot = starts[(key>>shift)+1] += 1
        ordered[slot] = key
        permutation[slot] = index
    end
    for slot = 2:n
        key = ordered[slot]
        index = permutation[slot]
        previous = slot - 1
        while previous >= 1 && ordered[previous] > key
            ordered[previous+1] = ordered[previous]
            permutation[previous+1] = permutation[previous]
            previous -= 1
        end
        ordered[previous+1] = key
        permutation[previous+1] = index
    end
    return permutation
end

# A device orders its keys through its backend. The KernelAbstractions extension
# gives GPU backends a counting sort with the same stable order.
_order_keys!(rng, permutation::AbstractVector, keys::AbstractVector) =
    _order_device_keys!(_fill_backend(rng.device, keys), permutation, keys)
_order_keys!(rng, permutation::AbstractVector{<:Integer}, keys::Vector{UInt64}) =
    _order_keys!(permutation, keys)

_order_device_keys!(backend, permutation, keys) = sortperm!(permutation, keys)

function _resolve_key_ties!(permutation::AbstractVector, keys::Vector{UInt64}, rng)
    n = length(permutation)
    first_index = 1
    while first_index < n
        key = keys[permutation[first_index]]
        last_index = first_index
        while last_index < n && keys[permutation[last_index+1]] == key
            last_index += 1
        end
        if last_index > first_index
            _order_key_run!(permutation, first_index, last_index, subrng(rng, key))
        end
        first_index = last_index + 1
    end
    return permutation
end

# A device orders its runs of equal keys through its backend, without a host
# round trip. The KernelAbstractions extension gives GPU backends the kernel,
# which orders the runs that start in the first `limit` slots.
function _order_device_key_runs! end

_resolve_key_ties!(permutation::AbstractVector, keys::AbstractVector, rng) =
    _order_device_key_runs!(
        _fill_backend(rng.device, keys),
        permutation,
        keys,
        length(permutation),
        rng,
    )

# `destination` receives the permutation of `1:length(destination)`.
function _randperm_next!(rng::AbstractPureRNG, destination::AbstractArray, threaded::Bool)
    _check_fill_device(rng, destination)
    _check_permutation_serviceability(rng)
    keys, next_rng = _permutation_keys(rng, length(destination), threaded)
    permutation = reshape(destination, :)
    _order_keys!(rng, permutation, keys)
    _resolve_key_ties!(permutation, keys, rng)
    return destination, next_rng
end

@inline _check_permutation_serviceability(rng) = nothing

function _randperm_next(rng::AbstractPureRNG, n::T, threaded::Bool) where {T<:Integer}
    n < 0 && _permutation_length_error(n)
    destination = _allocate_array(rng.device, T, (Int(n),))
    return _randperm_next!(rng, destination, threaded)
end

# The cycle sends each element to the next one in permutation order, which makes
# a uniform permutation into a uniform cyclic permutation.
function _randcycle_next!(rng::AbstractPureRNG, destination::AbstractArray, threaded::Bool)
    _check_fill_device(rng, destination)
    order = similar(destination, length(destination))
    _, next_rng = _randperm_next!(rng, order, threaded)
    cycle = reshape(destination, :)
    isempty(order) ||
        _scatter!(_fill_backend(rng.device, order), cycle, order, circshift(order, -1))
    return destination, next_rng
end

function _randcycle_next(rng::AbstractPureRNG, n::T, threaded::Bool) where {T<:Integer}
    n < 0 && _permutation_length_error(n)
    destination = _allocate_array(rng.device, T, (Int(n),))
    return _randcycle_next!(rng, destination, threaded)
end

# Elements move in linear order, as in `Random.shuffle!`.
# Shuffled elements may be of any type, so the device check follows the storage,
# as sampling destinations do.
function _shuffle_next!(rng::AbstractPureRNG, values::AbstractArray, threaded::Bool)
    _check_sampling_fill_device(rng, values)
    order, next_rng = _randperm_next(rng, length(values), threaded)
    copyto!(values, _gather(_fill_backend(rng.device, order), vec(values), order))
    return values, next_rng
end

_shuffle_next(rng::AbstractPureRNG, values::AbstractArray, threaded::Bool) =
    _shuffle_next!(rng, copy(values), threaded)

"""
    randperm_next(rng, n; threaded=false) -> (permutation, next_rng)

Return a uniform permutation of `1:n` and the advanced immutable generator. The
permutation orders `n` uniform `UInt64` draws at the held position, so it
consumes `64n` bits, and it is the same on the CPU and on a GPU. Draws that tie
are shuffled within their run with draws from `subrng(rng, key)`, which keeps
the permutation exactly uniform without changing the consumption.

The result has the element type of `n` and lives on the generator's device.
`threaded=true` splits the CPU key draw across threads without changing values.
The input generator never changes.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> permutation, next_rng = randperm_next(rng, 5);

julia> sort(permutation) == 1:5
true
```
"""
randperm_next(rng::AbstractPureRNG, n::Integer; threaded::Bool = false) =
    _randperm_next(rng, n, threaded)

"""
    randperm_next!(rng, destination; threaded=false) -> (destination, next_rng)

Fill `destination` with the permutation of `1:length(destination)` that
[`randperm_next`](@ref) returns, in linear order, and return the advanced
immutable generator.
"""
randperm_next!(
    rng::AbstractPureRNG,
    destination::AbstractArray{<:Integer};
    threaded::Bool = false,
) = _randperm_next!(rng, destination, threaded)

"""
    randcycle_next(rng, n; threaded=false) -> (cycle, next_rng)

Return a uniform cyclic permutation of `1:n` and the advanced immutable
generator. With `p = first(randperm_next(rng, n))`, the cycle sends `p[i]` to
`p[i + 1]` and `p[n]` to `p[1]`, so it consumes what [`randperm_next`](@ref)
does.
"""
randcycle_next(rng::AbstractPureRNG, n::Integer; threaded::Bool = false) =
    _randcycle_next(rng, n, threaded)

"""
    randcycle_next!(rng, destination; threaded=false) -> (destination, next_rng)

Fill `destination` with the cyclic permutation that [`randcycle_next`](@ref)
returns for `length(destination)`, in linear order, and return the advanced
immutable generator.
"""
randcycle_next!(
    rng::AbstractPureRNG,
    destination::AbstractArray{<:Integer};
    threaded::Bool = false,
) = _randcycle_next!(rng, destination, threaded)

"""
    shuffle_next(rng, values; threaded=false) -> (shuffled, next_rng)

Return a shuffled copy of `values` and the advanced immutable generator.
Element `i` of the copy, in linear order, is element `p[i]` of `values`, where
`p = first(randperm_next(rng, length(values)))`.
"""
shuffle_next(rng::AbstractPureRNG, values::AbstractArray; threaded::Bool = false) =
    _shuffle_next(rng, values, threaded)

"""
    shuffle_next!(rng, values; threaded=false) -> (values, next_rng)

Shuffle `values` in place as [`shuffle_next`](@ref) does and return the
advanced immutable generator.
"""
shuffle_next!(rng::AbstractPureRNG, values::AbstractArray; threaded::Bool = false) =
    _shuffle_next!(rng, values, threaded)

Random.randperm(rng::AbstractPureRNG, n::Integer; threaded::Bool = false) =
    first(_randperm_next(rng, n, threaded))
Random.randperm!(
    rng::AbstractPureRNG,
    destination::AbstractArray{<:Integer};
    threaded::Bool = false,
) = first(_randperm_next!(rng, destination, threaded))
Random.randcycle(rng::AbstractPureRNG, n::Integer; threaded::Bool = false) =
    first(_randcycle_next(rng, n, threaded))
Random.randcycle!(
    rng::AbstractPureRNG,
    destination::AbstractArray{<:Integer};
    threaded::Bool = false,
) = first(_randcycle_next!(rng, destination, threaded))
Random.shuffle(rng::AbstractPureRNG, values::AbstractArray; threaded::Bool = false) =
    first(_shuffle_next(rng, values, threaded))
Random.shuffle!(rng::AbstractPureRNG, values::AbstractArray; threaded::Bool = false) =
    first(_shuffle_next!(rng, values, threaded))
