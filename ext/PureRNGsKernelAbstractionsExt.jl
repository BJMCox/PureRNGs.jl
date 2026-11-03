module PureRNGsKernelAbstractionsExt

import Adapt
import KernelAbstractions
import PureRNGs
using KernelAbstractions: @index, @localmem, @private, @synchronize, @uniform

const IR = PureRNGs

# A device destination resolves to the launchable KernelAbstractions backend.
# The core keeps the host method, which never names a KernelAbstractions type.
@inline IR._fill_backend(::IR._BackendToken, destination) =
    KernelAbstractions.get_backend(destination)

KernelAbstractions.@kernel function _transformed_fill_kernel!(
    rng,
    destination,
    ::Val{T},
    codec,
    transform,
) where {T}
    ordinal = @index(Global, Linear)
    indices = eachindex(destination)
    index = @inbounds IR._destination_index(indices, ordinal)
    bits_lo, bits_hi = IR._bit_span(UInt64(ordinal - 1), IR._fill_width(codec, T))
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = transform(codec, rng, position, T)
end

KernelAbstractions.@kernel function _grouped_fill_kernel!(
    rng,
    destination,
    ::Val{T},
    group::Val{N},
    codec,
) where {T,N}
    workitem = @index(Global, Linear)
    first = (workitem - 1) * N + 1
    bits_lo, bits_hi = IR._bit_span(UInt64(first - 1), IR._fill_width(codec, T))
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    IR._fill_group!(rng, position, destination, T, first, group, codec)
end

@inline function IR._launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    ::Nothing,
) where {T}
    _transformed_fill_kernel!(backend)(
        rng,
        destination,
        Val(T),
        codec,
        IR._draw_transform(codec);
        ndrange = length(destination),
    )
    return destination
end

@inline function IR._launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    plan::Tuple{Val{:grouped},Val{N}},
) where {T,N}
    group = plan[2]
    workitems = cld(length(destination), IR._val_count(group))
    _grouped_fill_kernel!(backend)(
        rng,
        destination,
        Val(T),
        group,
        codec;
        ndrange = workitems,
    )
    return destination
end

# A device population or Dirichlet shape vector is an array on that device, and
# a kernel argument is only converted at its top level, so the codec hands its
# field to the same adaptor.
Adapt.adapt_structure(to, codec::IR._PopulationCodec) =
    IR._PopulationCodec(Adapt.adapt(to, codec.population), codec.cardinality)
function Adapt.adapt_structure(to, codec::IR._DirichletCodec)
    alpha = Adapt.adapt(to, codec.alpha)
    recover =
        eltype(alpha) === eltype(codec.alpha) ? Val(IR._recover_log_overflow(codec)) :
        Val(true)
    return IR._DirichletCodec(alpha, recover)
end

# One workgroup folds short CUDA inputs and all AMDGPU inputs. Both backends
# support 1024 workitems; large CUDA inputs use the CUDA extension's parallel scan.
@inline _weight_fold_lanes(_backend) = Val(1024)
@inline _weight_fold_lane_count(::Val{lanes}) where {lanes} = lanes

# The cumulative table is a strict Float64 left fold in ordinal order, as on the
# CPU, so lane 1 alone accumulates while the other lanes only stage and store.
# State that lives across `@synchronize` is `@private` or `@uniform`, and the loop
# runs over a uniform range, as the CPU backend requires.
KernelAbstractions.@kernel function _weighted_fold_kernel!(
    source,
    total_result,
    cumulative,
    ::Val{lanes},
    ::Val{validate_elements},
) where {lanes,validate_elements}
    lane = @index(Local, Linear)
    staged = @localmem Float64 (lanes,)
    @uniform count = length(source)
    total = @private Float64 (1,)
    invalid = @private Bool (1,)
    ordinal = @private Int (1,)
    total[1] = zero(Float64)
    invalid[1] = false
    for start = 1:lanes:count
        ordinal[1] = start + lane - 1
        if ordinal[1] <= count
            @inbounds staged[lane] =
                Float64(IR._population_value(source, UInt64(ordinal[1])))
        end
        @synchronize

        if lane == 1
            last = min(lanes, count - start + 1)
            @inbounds for slot = 1:last
                weight = staged[slot]
                invalid[1] |=
                    validate_elements && (!isfinite(weight) || weight < zero(Float64))
                total[1] += weight
                staged[slot] = total[1]
            end
        end
        @synchronize

        if ordinal[1] <= count
            @inbounds cumulative[ordinal[1]] = staged[lane]
        end
        @synchronize
    end
    if lane == 1
        @inbounds total_result[1] = ifelse(invalid[1], NaN, total[1])
    end
end

KernelAbstractions.@kernel function _rescale_weights_kernel!(total_result, cumulative)
    index = @index(Global, Linear)
    value = cumulative[index] * IR._WEIGHT_SCALE
    cumulative[index] = value
    # The collective CUDA scan aliases its total with this last boundary.
    index == length(cumulative) && (total_result[1] = value)
end

function IR._finish_device_weights!(backend, total_result, cumulative)
    total = only(Array(total_result))
    isfinite(total) && total > 0.0 || IR._invalid_weights()
    if total < IR._WEIGHT_SCALE_BOUND
        _rescale_weights_kernel!(backend)(
            total_result,
            cumulative;
            ndrange = length(cumulative),
        )
    end
    return nothing
end

IR._prepare_weight_scan(rng, weights, agnostic::Bool) =
    IR._fold_device_weights(rng.device, weights, agnostic)

function IR._fold_device_weights(device, weights, agnostic::Bool)
    source = agnostic ? IR._transfer_array(device, IR._collect_weights(weights)) : weights
    cumulative = IR._allocate_array(device, Float64, (length(source),))
    total_result = IR._allocate_array(device, Float64, (1,))
    backend = IR._fill_backend(device, cumulative)
    lanes = _weight_fold_lanes(backend)
    lane_count = _weight_fold_lane_count(lanes)
    _weighted_fold_kernel!(backend)(
        source,
        total_result,
        cumulative,
        lanes,
        Val(!agnostic);
        ndrange = lane_count,
        workgroupsize = lane_count,
    )
    IR._finish_device_weights!(backend, total_result, cumulative)
    return nothing, total_result, cumulative
end

@inline function _write_weighted_sample!(
    population,
    cumulative,
    destination,
    threshold,
    index,
)
    # Preparation supplies a nonempty CDF matching the population.
    # The search stays within that CDF; ndrange bounds the destination ordinal.
    lower = 1
    upper = length(cumulative)
    @inbounds while lower < upper
        middle = lower + ((upper - lower) >>> 1)
        if threshold < cumulative[middle]
            upper = middle
        else
            lower = middle + 1
        end
    end
    indices = eachindex(destination)
    @inbounds begin
        destination_index = @inbounds IR._destination_index(indices, index)
        destination[destination_index] = IR._population_value(population, UInt64(lower))
    end
end

# Weight preparation supplies a one-element total; ndrange bounds each ordinal.
KernelAbstractions.@kernel function _weighted_sample_kernel!(
    rng,
    population,
    total_result,
    cumulative,
    destination,
)
    index = @index(Global, Linear)
    bits_lo, bits_hi = IR._bit_span(UInt64(index - 1), IR._WEIGHT_BITS)
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    total = @inbounds total_result[1]
    threshold = IR._weighted_threshold(rng, position, total)
    _write_weighted_sample!(population, cumulative, destination, threshold, index)
end

KernelAbstractions.@kernel function _weighted_threshold_kernel!(
    rng,
    total_result,
    thresholds,
)
    index = @index(Global, Linear)
    bits_lo, bits_hi = IR._bit_span(UInt64(index - 1), IR._WEIGHT_BITS)
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    total = @inbounds total_result[1]
    @inbounds thresholds[index] = IR._weighted_threshold(rng, position, total)
end

KernelAbstractions.@kernel function _weighted_binary_search_kernel!(
    population,
    cumulative,
    thresholds,
    destination,
)
    index = @index(Global, Linear)
    threshold = @inbounds thresholds[index]
    _write_weighted_sample!(population, cumulative, destination, threshold, index)
end

@inline function IR._fill_weighted_samples!(
    backend,
    rng,
    population,
    _weights,
    total,
    cumulative,
    destination,
)
    # On A100, fusion reduces elapsed time for small synchronized calls.
    # Keep the two-kernel path above that range: its measured latency is lower
    # with CUDA's default cooperative wait, despite the fused kernel being faster.
    count = length(destination)
    if count <= 8192
        _weighted_sample_kernel!(backend)(
            rng,
            population,
            total,
            cumulative,
            destination;
            ndrange = count,
        )
    else
        thresholds = IR._allocate_array(rng.device, Float64, (count,))
        _weighted_threshold_kernel!(backend)(rng, total, thresholds; ndrange = count)
        _weighted_binary_search_kernel!(backend)(
            population,
            cumulative,
            thresholds,
            destination;
            ndrange = count,
        )
    end
    return destination
end

# Permutation keys are uniform, so a counting pass over their top bits leaves
# about one key per bucket, and one work item then orders its bucket by key and
# index. The atomic scatter leaves each bucket in a varying order, which the
# bucket order removes, so the result is the stable order of `sortperm!`. A100:
# a 2^24 permutation takes 13.7 ms against 104 ms for CUDA.jl's `sortperm!`,
# and 0.38 against 3.0 ms at 2^20. Fewer, fuller buckets measured no better.
const _DEVICE_KEY_BUCKET_BITS = 24

KernelAbstractions.@kernel function _key_count_kernel!(counts, keys, shift)
    index = @index(Global, Linear)
    if index <= length(keys)
        bucket = Int(keys[index] >> shift) + 1
        KernelAbstractions.@atomic counts[bucket] += Int32(1)
    end
end

KernelAbstractions.@kernel function _key_scatter_kernel!(
    ordered,
    permutation,
    cursor,
    keys,
    shift,
)
    index = @index(Global, Linear)
    if index <= length(keys)
        key = keys[index]
        bucket = Int(key >> shift) + 1
        slot = KernelAbstractions.@atomic cursor[bucket] += Int32(1)
        ordered[slot] = key
        permutation[slot] = index
    end
end

KernelAbstractions.@kernel function _key_bucket_order_kernel!(ordered, permutation, ends)
    bucket = @index(Global, Linear)
    if bucket <= length(ends)
        first_slot = bucket == 1 ? 1 : Int(ends[bucket-1]) + 1
        for slot = (first_slot+1):Int(ends[bucket])
            key = ordered[slot]
            index = permutation[slot]
            previous = slot - 1
            while previous >= first_slot && (
                ordered[previous] > key ||
                (ordered[previous] == key && permutation[previous] > index)
            )
                ordered[previous+1] = ordered[previous]
                permutation[previous+1] = permutation[previous]
                previous -= 1
            end
            ordered[previous+1] = key
            permutation[previous+1] = index
        end
    end
end

function IR._order_device_keys!(backend::KernelAbstractions.Backend, permutation, keys)
    n = length(keys)
    (n < 2 || n > 1 << (_DEVICE_KEY_BUCKET_BITS + 3)) && return sortperm!(permutation, keys)
    bits = min(_DEVICE_KEY_BUCKET_BITS, 64 - leading_zeros(n - 1))
    shift = 64 - bits
    counts = KernelAbstractions.zeros(backend, Int32, 1 << bits)
    _key_count_kernel!(backend)(counts, keys, shift; ndrange = n)
    # `cumsum` widens to Int64, and Metal has no 64-bit atomics.
    ends = accumulate(+, counts)
    cursor = ends .- counts
    ordered = similar(keys)
    _key_scatter_kernel!(backend)(ordered, permutation, cursor, keys, shift; ndrange = n)
    _key_bucket_order_kernel!(backend)(ordered, permutation, ends; ndrange = length(ends))
    return permutation
end


# Retain each tile's first `count` keys. A discarded key has at least `count`
# earlier keys in its own tile, so it cannot belong to the global first `count`.
# Tile-major storage also preserves original-index order among equal keys.
KernelAbstractions.@kernel function _race_candidates_kernel!(
    candidate_keys,
    candidate_indices,
    keys,
    count,
)
    lane = @index(Local, Linear)
    group = @index(Group, Linear)
    index = (group - 1) * 256 + lane
    # KA 0.9's CPU lowering needs private storage across barrier regions.
    key = @private UInt64 (1,)
    ordinal = @private Int (1,)
    key[1] = index <= length(keys) ? keys[index] : typemax(UInt64)
    ordinal[1] = index <= length(keys) ? index : typemax(Int)
    staged_keys = @localmem UInt64 (256,)
    staged_indices = @localmem Int (256,)
    for selected = 1:count
        staged_keys[lane] = key[1]
        staged_indices[lane] = ordinal[1]
        @synchronize
        for shift = 7:-1:0
            offset = 1 << shift
            if lane <= offset
                a = (staged_keys[lane], staged_indices[lane])
                b = (staged_keys[lane+offset], staged_indices[lane+offset])
                if isless(b, a)
                    staged_keys[lane] = b[1]
                    staged_indices[lane] = b[2]
                end
            end
            @synchronize
        end
        winner = staged_indices[1]
        if lane == 1
            slot = (group - 1) * count + selected
            candidate_keys[slot] = staged_keys[1]
            candidate_indices[slot] = winner
        end
        if ordinal[1] == winner
            key[1] = typemax(UInt64)
            ordinal[1] = typemax(Int)
        end
        @synchronize
    end
end

function IR._race_candidates(backend::KernelAbstractions.Backend, keys, count)
    # A100 end-to-end timings favor the shortlist only for large populations
    # and small samples. Keep full sorting outside that range.
    (length(keys) < 1 << 20 || count > 32) && return keys, nothing
    groups = cld(length(keys), 256)
    candidate_keys = similar(keys, groups * count)
    candidate_indices = similar(keys, Int, groups * count)
    _race_candidates_kernel!(backend, 256)(
        candidate_keys,
        candidate_indices,
        keys,
        count;
        ndrange = groups * 256,
    )
    return candidate_keys, candidate_indices
end


# A workitem per slot finds the runs of equal keys that open in the first
# `limit` slots, and the workitem at a run's first slot orders the whole run.
# Other workitems read slots of a run being reordered, but every member of a run
# holds the same key, so the run boundaries they see do not change.
KernelAbstractions.@kernel function _key_run_kernel!(permutation, keys, limit, rng)
    first_slot = @index(Global, Linear)
    n = length(permutation)
    if first_slot <= limit && first_slot < n
        key = keys[permutation[first_slot]]
        opens = first_slot == 1 || keys[permutation[first_slot-1]] != key
        if opens && keys[permutation[first_slot+1]] == key
            last_slot = first_slot + 1
            while last_slot < n && keys[permutation[last_slot+1]] == key
                last_slot += 1
            end
            IR._order_key_run!(
                permutation,
                first_slot,
                last_slot,
                IR._key_run_rng(rng, key),
            )
        end
    end
end

function IR._order_device_key_runs!(
    backend::KernelAbstractions.Backend,
    permutation,
    keys,
    limit,
    rng,
)
    workitems = min(limit, length(permutation) - 1)
    workitems > 0 &&
        _key_run_kernel!(backend)(permutation, keys, limit, rng; ndrange = workitems)
    return permutation
end


KernelAbstractions.@kernel function _gather_kernel!(destination, source, indices)
    slot = @index(Global, Linear)
    destination[slot] = source[indices[slot]]
end

KernelAbstractions.@kernel function _scatter_kernel!(destination, indices, source)
    slot = @index(Global, Linear)
    destination[indices[slot]] = source[slot]
end

function IR._gather(backend::KernelAbstractions.Backend, source, indices)
    # A range has no device storage; otherwise preserve the source's storage kind.
    destination = if source isa AbstractRange
        similar(indices, eltype(source), length(indices))
    else
        similar(source, length(indices))
    end
    isempty(indices) ||
        _gather_kernel!(backend)(destination, source, indices; ndrange = length(indices))
    return destination
end

function IR._scatter!(backend::KernelAbstractions.Backend, destination, indices, source)
    isempty(indices) ||
        _scatter_kernel!(backend)(destination, indices, source; ndrange = length(indices))
    return destination
end


KernelAbstractions.@kernel function _column_kernel!(f, destination, args)
    column = @index(Global, Linear)
    f(destination, column, args...)
end

KernelAbstractions.@kernel function _element_kernel!(f, destination, args)
    index = @index(Global, Linear)
    f(destination, index, args...)
end

function IR._foreach_element!(backend::KernelAbstractions.Backend, f, destination, args...)
    isempty(destination) ||
        _element_kernel!(backend)(f, destination, args; ndrange = length(destination))
    return destination
end

function IR._foreach_column!(
    backend::KernelAbstractions.Backend,
    f,
    destination,
    ::Bool,
    args...,
)
    columns = size(destination, 2)
    columns > 0 && _column_kernel!(backend)(f, destination, args; ndrange = columns)
    return destination
end

end
