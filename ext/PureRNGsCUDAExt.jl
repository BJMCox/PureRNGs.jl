module PureRNGsCUDAExt

import CUDA
import PureRNGs
import KernelAbstractions
import Random

const IR = PureRNGs
const _CUDAFamily = IR._BackendFamily{IR._CUDABackend}

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
