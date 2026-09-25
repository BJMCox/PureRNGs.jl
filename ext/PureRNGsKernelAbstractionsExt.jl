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
) where {T}
    ordinal = @index(Global, Linear)
    indices = eachindex(destination)
    index = @inbounds IR._destination_index(indices, ordinal)
    bits_lo, bits_hi = IR._bit_span(UInt64(ordinal - 1), IR._fill_width(codec, T))
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = IR._transformed_draw_unchecked(codec, rng, position, T)
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
        codec;
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

# A device population is an array on that device, and a kernel argument is only
# converted at its top level, so the codec hands its field to the same adaptor.
Adapt.adapt_structure(to, codec::IR._PopulationCodec) =
    IR._PopulationCodec(Adapt.adapt(to, codec.population), codec.cardinality)

# One workgroup folds the whole weight vector. Weighted sampling runs on CUDA and
# AMDGPU, whose workgroups hold 1024 workitems; Metal has no Float64 to fold.
@inline _weight_fold_lanes(_backend) = Val(1024)
@inline _weight_fold_lane_count(::Val{lanes}) where {lanes} = lanes

# The cumulative table is a strict Float64 left fold in ordinal order, as on the
# CPU, so lane 1 alone accumulates while the other lanes only stage and store.
# State that lives across `@synchronize` is `@private` or `@uniform`, and the loop
# runs over a uniform range, as the CPU backend requires.
KernelAbstractions.@kernel function _weighted_fold_kernel!(
    source,
    total_result,
    invalid_result,
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
        @inbounds begin
            total_result[1] = total[1]
            invalid_result[1] =
                invalid[1] | (!isfinite(total[1]) || total[1] <= zero(Float64))
        end
    end
end

IR._prepare_weight_scan(rng, weights, agnostic::Bool) =
    IR._fold_device_weights(rng.device, weights, agnostic)

function IR._fold_device_weights(device, weights, agnostic::Bool)
    source = agnostic ? IR._transfer_array(device, IR._collect_weights(weights)) : weights
    cumulative = IR._allocate_array(device, Float64, (length(source),))
    total_result = IR._allocate_array(device, Float64, (1,))
    invalid_result = IR._allocate_array(device, Bool, (1,))
    backend = IR._fill_backend(device, cumulative)
    lanes = _weight_fold_lanes(backend)
    lane_count = _weight_fold_lane_count(lanes)
    _weighted_fold_kernel!(backend)(
        source,
        total_result,
        invalid_result,
        cumulative,
        lanes,
        Val(!agnostic);
        ndrange = lane_count,
        workgroupsize = lane_count,
    )
    only(Array(invalid_result)) && IR._invalid_weights()
    return nothing, total_result, cumulative
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

@inline function IR._fill_weighted_thresholds!(backend, rng, total_result, thresholds)
    _weighted_threshold_kernel!(backend)(
        rng,
        total_result,
        thresholds;
        ndrange = length(thresholds),
    )
    return thresholds
end

KernelAbstractions.@kernel function _weighted_binary_search_kernel!(
    population,
    cumulative,
    thresholds,
    destination,
)
    index = @index(Global, Linear)
    threshold = @inbounds thresholds[index]
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

@inline function IR._launch_weighted_scan!(
    backend,
    population,
    cumulative,
    thresholds,
    destination,
)
    _weighted_binary_search_kernel!(backend)(
        population,
        cumulative,
        thresholds,
        destination;
        ndrange = length(destination),
    )
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

function IR._order_device_keys!(backend::KernelAbstractions.GPU, permutation, keys)
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

function IR._gather(backend::KernelAbstractions.GPU, source, indices)
    destination = similar(source, length(indices))
    isempty(indices) ||
        _gather_kernel!(backend)(destination, source, indices; ndrange = length(indices))
    return destination
end

function IR._scatter!(backend::KernelAbstractions.GPU, destination, indices, source)
    isempty(indices) ||
        _scatter_kernel!(backend)(destination, indices, source; ndrange = length(indices))
    return destination
end


KernelAbstractions.@kernel function _column_kernel!(f, destination, args)
    column = @index(Global, Linear)
    f(destination, column, args...)
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
