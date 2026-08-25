module PureRNGsCUDAExt

import CUDA
import PureRNGs
import KernelAbstractions
import Random

const IR = PureRNGs
const _CUDAFamily = IR._BackendFamily{IR._CUDABackend}
const _CUDAPhilox4x32 = IR.Philox4x32{IR._CUDABackend}
const _CUDA_FILL_THREADS = 256
# Large-fill benchmarks select this multiple of the device thread-capacity block count.
const _CUDA_FILL_THREAD_CAPACITY_MULTIPLIER = 128
const _CUDA_U32X4 = NTuple{4,VecElement{UInt32}}
const _CUDA_FILL_ALIGNMENT = sizeof(_CUDA_U32X4)

@inline IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDAPhilox4x32,
    ::Type{UInt64},
) = (Val(:philox4x32_u64),)

@inline function IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAFamily,
    ::Type{T},
) where {T}
    cooperative = IR._cooperative_uniform_fill(rng, T)
    return cooperative === nothing ?
           (Val(:grouped), IR._device_uniform_fill_group(rng, T)) :
           (Val(:cooperative), cooperative...)
end

KernelAbstractions.@kernel function _philox4x32_u64_kernel!(rng, destination)
    index = KernelAbstractions.@index(Global, Linear)
    grid = KernelAbstractions.@ndrange()
    stride = grid[1]
    while index <= length(destination)
        limbs =
            IR._stream_limbs(rng, IR.FAMILY_BITS, rng.position.block + UInt64(index - 1))
        # VecElement lanes follow little-endian memory order for two stream UInt64 values.
        @inbounds destination[index] = (
            VecElement(limbs[1] % UInt32),
            VecElement((limbs[1] >> 32) % UInt32),
            VecElement(limbs[2] % UInt32),
            VecElement((limbs[2] >> 32) % UInt32),
        )
        index += stride
    end
end

@inline function _philox4x32_u64_blocks(pairs::Int)
    blocks = cld(pairs, _CUDA_FILL_THREADS)
    blocks <= _CUDA_FILL_THREAD_CAPACITY_MULTIPLIER && return blocks
    device = CUDA.device()
    multiprocessors = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    threads_per_multiprocessor =
        CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    thread_capacity_blocks =
        multiprocessors * max(1, threads_per_multiprocessor ÷ _CUDA_FILL_THREADS)
    return min(blocks, _CUDA_FILL_THREAD_CAPACITY_MULTIPLIER * thread_capacity_blocks)
end

@inline function _aligned_philox4x32_u64_fill(rng, destination)
    return iszero(rng.position.bit) &&
           iseven(length(destination)) &&
           destination isa CUDA.DenseCuArray{UInt64} &&
           iszero(UInt(pointer(destination)) & UInt(_CUDA_FILL_ALIGNMENT - 1))
end

@inline function IR._launch_device_fill!(
    backend::CUDA.CUDABackend,
    rng::_CUDAPhilox4x32,
    destination,
    ::Type{UInt64},
    codec::Val{:uniform},
    ::Tuple{Val{:philox4x32_u64}},
)
    if !_aligned_philox4x32_u64_fill(rng, destination)
        return IR._launch_device_fill!(
            backend,
            rng,
            destination,
            UInt64,
            codec,
            (Val(:grouped), IR._device_uniform_fill_group(rng, UInt64)),
        )
    end

    packed = reinterpret(_CUDA_U32X4, vec(destination))
    pairs = length(packed)
    blocks = _philox4x32_u64_blocks(pairs)
    _philox4x32_u64_kernel!(backend)(
        rng,
        packed;
        ndrange = blocks * _CUDA_FILL_THREADS,
        workgroupsize = _CUDA_FILL_THREADS,
    )
    return destination
end

@inline function IR._device_normal_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAFamily,
    ::Type{T},
) where {T}
    cooperative = IR._cooperative_normal_fill(rng, T)
    return cooperative === nothing ? (Val(:grouped), IR._device_normal_fill_group(T)) :
           (Val(:cooperative), cooperative...)
end

@inline function IR._device_range_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAFamily,
    span::UInt64,
)
    return IR._range_bits(span) == UInt16(128) ? nothing : (Val(:grouped), Val(2))
end

@inline function IR._allocate_array(::IR._CUDABackend, ::Type{T}, dims::Tuple) where {T}
    return CUDA.CuArray{T}(undef, dims)
end

@inline IR._materialize_population(::IR._CUDABackend, population) =
    CUDA.CuArray(IR._collect_population(population))

# CUDA's allocating `sortperm` stages ordinal indices from the host. Initialize
# those indices on-device; `sortperm!` treats thresholds as read-only keys.
@inline function IR._weighted_sortperm(::IR._CUDABackend, thresholds)
    order = similar(thresholds, Int)
    order .= eachindex(order)
    sortperm!(order, thresholds; initialized = true)
    return order
end

KernelAbstractions.@kernel function _prepare_cumulative_weights_kernel!(
    source,
    total_result,
    invalid_result,
    cumulative,
    validate_elements,
)
    if KernelAbstractions.@index(Global, Linear) == 1
        IR._convert_and_fold_weights!(
            source,
            nothing,
            total_result,
            invalid_result,
            cumulative,
            validate_elements,
        )
    end
end

function IR._prepare_weight_scan(rng::_CUDAFamily, weights, agnostic::Bool)
    source =
        agnostic ? IR._transfer_weights(rng.device, IR._collect_weights(weights)) : weights
    cumulative = IR._allocate_array(rng.device, Float64, (length(source),))
    total_result = IR._allocate_array(rng.device, Float64, (1,))
    invalid_result = IR._allocate_array(rng.device, Bool, (1,))
    IR._with_device(rng.device) do
        backend = IR._fill_backend(cumulative)
        _prepare_cumulative_weights_kernel!(backend)(
            source,
            total_result,
            invalid_result,
            cumulative,
            Val(!agnostic);
            ndrange = 1,
        )
    end
    only(Array(invalid_result)) && IR._invalid_weights()
    return nothing, total_result, cumulative
end

KernelAbstractions.@kernel function _weighted_binary_search_kernel!(
    population,
    cumulative,
    thresholds,
    order,
    destination,
)
    index = KernelAbstractions.@index(Global, Linear)
    threshold = @inbounds thresholds[order[index]]
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
    @inbounds destination[order[index]] = IR._population_value(population, UInt64(lower))
end

@inline function IR._launch_weighted_scan!(
    ::IR._CUDABackend,
    backend,
    population,
    _weights,
    thresholds,
    order,
    cumulative,
    destination,
)
    _weighted_binary_search_kernel!(backend)(
        population,
        cumulative,
        thresholds,
        order,
        destination;
        ndrange = length(destination),
    )
    return destination
end

@inline function IR.rand_next(rng::_CUDAFamily, dim1::Integer, dims::Integer...)
    return IR._rand_next_uniform_array(rng, Float64, (dim1, dims...))
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline function Random.rand(
            rng::_CUDAFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._rand_next_uniform_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function IR.rand_next(
            rng::_CUDAFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return IR._rand_next_uniform_array(rng, $T, (dim1, dims...))
        end
    end
end

@inline function IR.randn_next(rng::_CUDAFamily, dim1::Integer, dims::Integer...)
    return IR._randn_next_array(rng, Float64, (dim1, dims...))
end

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randn(
            rng::_CUDAFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._randn_next_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function IR.randn_next(
            rng::_CUDAFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return IR._randn_next_array(rng, $T, (dim1, dims...))
        end
    end
end

for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
    @eval begin
        @inline function Random.rand(
            rng::_CUDAFamily,
            range::AbstractRange{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._rand_next_range_array(rng, range, (dim1, dims...))
            return destination
        end

        @inline function IR.rand_next(
            rng::_CUDAFamily,
            range::AbstractRange{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return IR._rand_next_range_array(rng, range, (dim1, dims...))
        end
    end
end

end
