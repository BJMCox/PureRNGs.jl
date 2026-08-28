module PureRNGsCUDAExt

import CUDA
import PureRNGs
import KernelAbstractions

const IR = PureRNGs
const _CUDAFamily = IR._BackendFamily{IR._CUDABackend}
const _CUDAPhilox4x32 = IR.Philox4x32{IR._CUDABackend}
const _CUDAThreefry4x32 = IR.Threefry4x32{IR._CUDABackend}
const _CUDANatural128 = Union{_CUDAPhilox4x32,_CUDAThreefry4x32}
const _CUDANonPhilox4x32 = Union{
    IR.Philox2x32{IR._CUDABackend},
    IR.Philox2x64{IR._CUDABackend},
    IR.Philox4x64{IR._CUDABackend},
    IR.Threefry2x32{IR._CUDABackend},
    IR.Threefry4x32{IR._CUDABackend},
    IR.Threefry2x64{IR._CUDABackend},
    IR.Threefry4x64{IR._CUDABackend},
}
const _CUDA_FILL_THREADS = 256
# Large-fill benchmarks select this multiple of the device thread-capacity block count.
const _CUDA_FILL_THREAD_CAPACITY_MULTIPLIER = 128
const _CUDA_U32X4 = NTuple{4,VecElement{UInt32}}
const _CUDA_F32X4 = NTuple{4,VecElement{Float32}}
const _CUDA_F64X2 = NTuple{2,VecElement{Float64}}
const _CUDA_B8X16 = NTuple{16,VecElement{Bool}}
const _CUDA_FILL_ALIGNMENT = sizeof(_CUDA_U32X4)
const _CUDAPackedFloatCodec = Union{Val{:uniform},IR._CUDABackend}

@inline _bool_packs_per_block(rng) = Val(Int(IR._block_bits(rng)) ÷ 16)

struct _CUDANatural128Pack{T}
    lanes::_CUDA_U32X4
end

@inline IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDAPhilox4x32,
    ::Type{T},
) where {T<:IR._UniformInteger} = (Val(:natural128_packed),)

@inline IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDAThreefry4x32,
    ::Type{T},
) where {T<:IR._UniformInteger32} = (Val(:natural128_packed),)

@inline IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAPhilox4x32,
    ::Type{Float32},
) = (Val(:cooperative), IR._cooperative_uniform_fill(rng, Float32)..., Val(4))

@inline _packed_float_plan(::Type{Float32}) = (Val(2048), Val(32), Val(4))
@inline _packed_float_plan(::Type{Float64}) = (Val(1024), Val(64), Val(2))
@inline IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDANonPhilox4x32,
    ::Type{T},
) where {T<:Union{Float32,Float64}} = (Val(:cooperative), _packed_float_plan(T)...)

@inline function IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAFamily,
    ::Type{T},
) where {T}
    cooperative = IR._cooperative_uniform_fill(rng, T)
    cooperative === nothing || return (Val(:cooperative), cooperative...)
    T === Bool && return (Val(:bool_blocks), _bool_packs_per_block(rng))
    return (Val(:grouped), IR._device_uniform_fill_group(rng, T))
end

@inline _packed_float_layout(destination, ::Type{T}, ::Val{P}) where {T,P} =
    iszero(length(destination) % P) &&
    destination isa CUDA.DenseCuArray{T} &&
    iszero(UInt(pointer(destination)) & UInt(_CUDA_FILL_ALIGNMENT - 1))

@inline _packed_float_type(::Type{Float32}) = _CUDA_F32X4
@inline _packed_float_type(::Type{Float64}) = _CUDA_F64X2

@inline function IR._launch_device_fill!(
    backend::CUDA.CUDABackend,
    rng::_CUDANonPhilox4x32,
    destination,
    ::Type{T},
    codec::_CUDAPackedFloatCodec,
    plan::Tuple{Val{:cooperative},Val{O},Val{L},Val{P}},
) where {T<:Union{Float32,Float64},O,L,P}
    if !_packed_float_layout(destination, T, plan[4])
        return IR._launch_device_fill!(
            backend,
            rng,
            destination,
            T,
            codec,
            (Val(:grouped), IR._device_uniform_fill_group(rng, T)),
        )
    end

    packed = reinterpret(_packed_float_type(T), vec(destination))
    IR._launch_cooperative_fill!(backend, rng, packed, T, codec, plan, Val(false))
    return destination
end

@inline function IR._launch_device_fill!(
    backend::CUDA.CUDABackend,
    rng::_CUDAPhilox4x32,
    destination,
    ::Type{Float32},
    codec::_CUDAPackedFloatCodec,
    plan::Tuple{Val{:cooperative},Val{O},Val{L},Val{4}},
) where {O,L}
    if !_packed_float_layout(destination, Float32, plan[4])
        return IR._launch_device_fill!(
            backend,
            rng,
            destination,
            Float32,
            codec,
            (Val(:cooperative), plan[2], plan[3]),
        )
    end

    packed = reinterpret(_CUDA_F32X4, vec(destination))
    stream_aligned = Val(_stream_aligned_philox4x32_f32_fill(rng, destination, plan[2]))
    IR._launch_cooperative_fill!(backend, rng, packed, Float32, codec, plan, stream_aligned)
    return destination
end

@inline _natural128_packed(limbs, ::Type{UInt32}) = (
    VecElement((limbs[1] >> 32) % UInt32),
    VecElement(limbs[1] % UInt32),
    VecElement((limbs[2] >> 32) % UInt32),
    VecElement(limbs[2] % UInt32),
)

@inline _natural128_packed(limbs, ::Type{UInt64}) = (
    VecElement(limbs[1] % UInt32),
    VecElement((limbs[1] >> 32) % UInt32),
    VecElement(limbs[2] % UInt32),
    VecElement((limbs[2] >> 32) % UInt32),
)

@inline _natural128_packed(limbs, ::Type{Int32}) = _natural128_packed(limbs, UInt32)
@inline _natural128_packed(limbs, ::Type{Int64}) = _natural128_packed(limbs, UInt64)

@inline _natural128_packed(limbs, ::Type{_CUDANatural128Pack{T}}) where {T} =
    _CUDANatural128Pack{T}(_natural128_packed(limbs, T))

KernelAbstractions.@kernel function _natural128_packed_kernel!(rng, destination)
    index = KernelAbstractions.@index(Global, Linear)
    stride = KernelAbstractions.@ndrange()[1]
    while index <= length(destination)
        limbs =
            IR._stream_limbs(rng, IR.FAMILY_BITS, rng.position.block + UInt64(index - 1))
        # VecElement lanes follow the result type's little-endian memory order.
        @inbounds destination[index] = _natural128_packed(limbs, eltype(destination))
        index += stride
    end
end

@inline function _cuda_packed_blocks(packs::Int)
    blocks = cld(packs, _CUDA_FILL_THREADS)
    blocks <= _CUDA_FILL_THREAD_CAPACITY_MULTIPLIER && return blocks
    device = CUDA.device()
    multiprocessors = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    threads_per_multiprocessor =
        CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    thread_capacity_blocks =
        multiprocessors * max(1, threads_per_multiprocessor ÷ _CUDA_FILL_THREADS)
    return min(blocks, _CUDA_FILL_THREAD_CAPACITY_MULTIPLIER * thread_capacity_blocks)
end

@inline _aligned_bool_block_fill(rng, destination) =
    iszero(rng.position.bit) &&
    iszero(length(destination) % Int(IR._block_bits(rng))) &&
    destination isa CUDA.DenseCuArray{Bool} &&
    iszero(UInt(pointer(destination)) & UInt(_CUDA_FILL_ALIGNMENT - 1))

@inline function IR._launch_device_fill!(
    backend::CUDA.CUDABackend,
    rng::_CUDAFamily,
    destination,
    ::Type{Bool},
    codec::Val{:uniform},
    plan::Tuple{Val{:bool_blocks},Val{P}},
) where {P}
    if !_aligned_bool_block_fill(rng, destination)
        return IR._launch_device_fill!(
            backend,
            rng,
            destination,
            Bool,
            codec,
            (Val(:grouped), IR._device_uniform_fill_group(rng, Bool)),
        )
    end

    packed = reinterpret(_CUDA_B8X16, vec(destination))
    workitems = length(packed) ÷ P
    blocks = _cuda_packed_blocks(workitems)
    IR._uniform_fill_bool_blocks_kernel!(backend)(
        rng,
        packed,
        plan[2];
        ndrange = blocks * _CUDA_FILL_THREADS,
        workgroupsize = _CUDA_FILL_THREADS,
    )
    return destination
end

@inline function _aligned_natural128_fill(rng, destination, ::Type{T}) where {T}
    return iszero(rng.position.bit) &&
           iszero(length(destination) % (_CUDA_FILL_ALIGNMENT ÷ sizeof(T))) &&
           destination isa CUDA.DenseCuArray{T} &&
           iszero(UInt(pointer(destination)) & UInt(_CUDA_FILL_ALIGNMENT - 1))
end

@inline _stream_aligned_philox4x32_f32_fill(rng, destination, outputs) =
    iszero(rng.position.bit) && iszero(length(destination) % IR._fill_group_size(outputs))

@inline function IR._launch_device_fill!(
    backend::CUDA.CUDABackend,
    rng::_CUDANatural128,
    destination,
    ::Type{T},
    codec::Val{:uniform},
    ::Tuple{Val{:natural128_packed}},
) where {T<:IR._UniformInteger}
    if !_aligned_natural128_fill(rng, destination, T)
        return IR._launch_device_fill!(
            backend,
            rng,
            destination,
            T,
            codec,
            (Val(:grouped), IR._device_uniform_fill_group(rng, T)),
        )
    end

    packed = reinterpret(_CUDANatural128Pack{T}, vec(destination))
    blocks = _cuda_packed_blocks(length(packed))
    _natural128_packed_kernel!(backend)(
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

@inline function IR._transformed_fill_plan(
    ::IR._CUDABackend,
    backend::CUDA.CUDABackend,
    rng::_CUDAFamily,
    ::Type{T},
) where {T<:Union{Float32,Float64}}
    return IR._device_uniform_fill_plan(backend, rng, T)
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

function IR._prepare_weight_scan(rng::_CUDAFamily, weights, agnostic::Bool)
    source =
        agnostic ? IR._transfer_weights(rng.device, IR._collect_weights(weights)) : weights
    cumulative = IR._allocate_array(rng.device, Float64, (length(source),))
    total_result = IR._allocate_array(rng.device, Float64, (1,))
    invalid_result = IR._allocate_array(rng.device, Bool, (1,))
    backend = IR._fill_backend(cumulative)
    IR._prepare_weights_kernel!(backend)(
        source,
        nothing,
        total_result,
        invalid_result,
        cumulative,
        Val(!agnostic);
        ndrange = 1,
    )
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

end
