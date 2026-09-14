module PureRNGsCUDAExt

import CUDA
import PureRNGs
import KernelAbstractions

const IR = PureRNGs
const _CUDAGenerators = IR._BackendGenerators{IR._CUDABackend}
const _CUDAPhilox4x32 = IR.Philox4x32{IR._CUDABackend}
const _CUDAThreefry4x32 = IR.Threefry4x32{IR._CUDABackend}
const _CUDAPhilox2x64 = IR.Philox2x64{IR._CUDABackend}
const _CUDAPhilox4x64 = IR.Philox4x64{IR._CUDABackend}
const _CUDAThreefry4x64 = IR.Threefry4x64{IR._CUDABackend}
const _CUDANatural128 = Union{_CUDAPhilox4x32,_CUDAThreefry4x32}
const _CUDANonNatural128 = Union{
    IR.Philox2x32{IR._CUDABackend},
    IR.Philox2x64{IR._CUDABackend},
    IR.Philox4x64{IR._CUDABackend},
    IR.Threefry2x32{IR._CUDABackend},
    IR.Threefry2x64{IR._CUDABackend},
    IR.Threefry4x64{IR._CUDABackend},
    IR.ChaCha{IR._CUDABackend},
}
const _CUDANonPhilox4x32 = Union{_CUDANonNatural128,_CUDAThreefry4x32}
# A100 trials retain only generators where cooperative stores beat grouped fills.
const _CUDAPackedIntegerGenerators =
    Union{_CUDAPhilox2x64,_CUDAPhilox4x64,_CUDAThreefry4x64}
const _CUDA_FILL_THREADS = 256
const _CUDA_WEIGHT_FOLD_LANES = 1024
# Large-fill benchmarks select this multiple of the device thread-capacity block count.
const _CUDA_FILL_THREAD_CAPACITY_MULTIPLIER = 128
const _CUDA_U32X4 = NTuple{4,VecElement{UInt32}}
const _CUDA_I32X4 = NTuple{4,VecElement{Int32}}
const _CUDA_U64X2 = NTuple{2,VecElement{UInt64}}
const _CUDA_I64X2 = NTuple{2,VecElement{Int64}}
const _CUDA_F32X4 = NTuple{4,VecElement{Float32}}
const _CUDA_F64X2 = NTuple{2,VecElement{Float64}}
const _CUDA_B8X16 = NTuple{16,VecElement{Bool}}
const _CUDA_FILL_ALIGNMENT = sizeof(_CUDA_U32X4)
# Mapped codecs using this path must share the uniform-width fallback contract.
const _CUDAPackedValue = Union{IR._UniformInteger,Float32,Float64}
const _CUDAPackedCodec =
    Union{Val{:uniform},Val{:normal},IR._CUDABackend,IR._MappedFillCodec}

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
) where {T<:IR._UniformInteger} = (Val(:natural128_packed),)

@inline IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAPhilox4x32,
    ::Type{Float32},
) = (Val(:cooperative), IR._cooperative_uniform_fill(rng, Float32)..., Val(4))

@inline _packed_16byte_plan(::Type{T}) where {T<:Union{UInt32,Int32,Float32}} =
    (Val(2048), Val(32), Val(4))
@inline _packed_16byte_plan(::Type{T}) where {T<:Union{UInt64,Int64,Float64}} =
    (Val(1024), Val(64), Val(2))
@inline _packed_integer_plan(::Type{T}) where {T<:Union{UInt32,Int32}} =
    _packed_16byte_plan(T)
# A100 trials favor twice the output tile for 64-bit integer stores.
@inline _packed_integer_plan(::Type{T}) where {T<:Union{UInt64,Int64}} =
    (Val(2048), Val(64), Val(2))
@inline IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDANonPhilox4x32,
    ::Type{T},
) where {T<:Union{Float32,Float64}} = (Val(:cooperative), _packed_16byte_plan(T)...)

@inline IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDAPackedIntegerGenerators,
    ::Type{T},
) where {T<:IR._UniformInteger} = (Val(:cooperative), _packed_integer_plan(T)...)

@inline function IR._device_uniform_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAGenerators,
    ::Type{T},
) where {T}
    cooperative = IR._cooperative_uniform_fill(rng, T)
    cooperative === nothing || return (Val(:cooperative), cooperative...)
    T === Bool && return (Val(:bool_blocks), _bool_packs_per_block(rng))
    return (Val(:grouped), IR._device_uniform_fill_group(rng, T))
end

@inline _dense_16byte_aligned(destination, ::Type{T}) where {T} =
    destination isa CUDA.DenseCuArray{T} &&
    iszero(UInt(pointer(destination)) & UInt(_CUDA_FILL_ALIGNMENT - 1))

@inline _packed_16byte_layout(destination, ::Type{T}, ::Val{P}) where {T,P} =
    iszero(length(destination) % P) && _dense_16byte_aligned(destination, T)

@inline _packed_type(::Type{Float32}) = _CUDA_F32X4
@inline _packed_type(::Type{Float64}) = _CUDA_F64X2
@inline _packed_type(::Type{UInt32}) = _CUDA_U32X4
@inline _packed_type(::Type{Int32}) = _CUDA_I32X4
@inline _packed_type(::Type{UInt64}) = _CUDA_U64X2
@inline _packed_type(::Type{Int64}) = _CUDA_I64X2

@inline _packed_integer_min_length(::_CUDAPhilox2x64, ::Type{<:IR._UniformInteger}) =
    1 << 20
@inline _packed_integer_min_length(::_CUDAPhilox4x64, ::Type{<:IR._UniformInteger}) = 4096
@inline _packed_integer_min_length(::_CUDAThreefry4x64, ::Type{<:Union{UInt32,Int32}}) =
    8192
@inline _packed_integer_min_length(::_CUDAThreefry4x64, ::Type{<:Union{UInt64,Int64}}) =
    4096

@inline _packed_fallback_group(rng, ::Type{T}, ::Val{:normal}) where {T} =
    IR._device_normal_fill_group(T)
@inline _packed_fallback_group(
    rng,
    ::Type{T},
    ::Union{Val{:uniform},IR._CUDABackend,IR._MappedFillCodec},
) where {T} = IR._device_uniform_fill_group(rng, T)

@inline function IR._launch_device_fill!(
    backend::CUDA.CUDABackend,
    rng::_CUDAGenerators,
    destination,
    ::Type{T},
    codec::_CUDAPackedCodec,
    plan::Tuple{Val{:cooperative},Val{O},Val{L},Val{P}},
) where {T<:_CUDAPackedValue,O,L,P}
    below_integer_crossover =
        T <: IR._UniformInteger && length(destination) < _packed_integer_min_length(rng, T)
    if below_integer_crossover || !_packed_16byte_layout(destination, T, plan[4])
        return IR._launch_device_fill!(
            backend,
            rng,
            destination,
            T,
            codec,
            (Val(:grouped), _packed_fallback_group(rng, T, codec)),
        )
    end

    IR._launch_cooperative_fill!(
        backend,
        rng,
        destination,
        T,
        codec,
        plan,
        Val(false),
        _packed_type(T),
    )
    return destination
end

@inline function IR._launch_device_fill!(
    backend::CUDA.CUDABackend,
    rng::_CUDAPhilox4x32,
    destination,
    ::Type{Float32},
    codec::_CUDAPackedCodec,
    plan::Tuple{Val{:cooperative},Val{O},Val{L},Val{4}},
) where {O,L}
    if !_packed_16byte_layout(destination, Float32, plan[4])
        return IR._launch_device_fill!(
            backend,
            rng,
            destination,
            Float32,
            codec,
            (Val(:cooperative), plan[2], plan[3]),
        )
    end

    stream_aligned = Val(_stream_aligned_philox4x32_f32_fill(rng, destination, plan[2]))
    IR._launch_cooperative_fill!(
        backend,
        rng,
        destination,
        Float32,
        codec,
        plan,
        stream_aligned,
        _CUDA_F32X4,
    )
    return destination
end

@inline _natural128_packed(block_words, ::Type{UInt32}) = (
    VecElement((block_words[1] >> 32) % UInt32),
    VecElement(block_words[1] % UInt32),
    VecElement((block_words[2] >> 32) % UInt32),
    VecElement(block_words[2] % UInt32),
)

@inline _natural128_packed(block_words, ::Type{UInt64}) = (
    VecElement(block_words[1] % UInt32),
    VecElement((block_words[1] >> 32) % UInt32),
    VecElement(block_words[2] % UInt32),
    VecElement((block_words[2] >> 32) % UInt32),
)

@inline _natural128_packed(block_words, ::Type{Int32}) =
    _natural128_packed(block_words, UInt32)
@inline _natural128_packed(block_words, ::Type{Int64}) =
    _natural128_packed(block_words, UInt64)

@inline _natural128_packed(block_words, ::Type{_CUDANatural128Pack{T}}) where {T} =
    _CUDANatural128Pack{T}(_natural128_packed(block_words, T))

KernelAbstractions.@kernel function _natural128_packed_kernel!(
    rng,
    destination,
    ::Val{D},
) where {D}
    destination = reinterpret(D, vec(destination))
    index = KernelAbstractions.@index(Global, Linear)
    stride = KernelAbstractions.@ndrange()[1]
    while index <= length(destination)
        block_words = IR._block_words(rng, rng.position.block + UInt64(index - 1))
        # VecElement lanes follow the result type's little-endian memory order.
        @inbounds destination[index] = _natural128_packed(block_words, eltype(destination))
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
    _dense_16byte_aligned(destination, Bool)

@inline function IR._launch_device_fill!(
    backend::CUDA.CUDABackend,
    rng::_CUDAGenerators,
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
           _dense_16byte_aligned(destination, T)
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

    storage_type = _CUDANatural128Pack{T}
    blocks = _cuda_packed_blocks(length(destination) ÷ (_CUDA_FILL_ALIGNMENT ÷ sizeof(T)))
    _natural128_packed_kernel!(backend)(
        rng,
        destination,
        Val(storage_type);
        ndrange = blocks * _CUDA_FILL_THREADS,
        workgroupsize = _CUDA_FILL_THREADS,
    )
    return destination
end

@inline function IR._device_normal_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAGenerators,
    ::Type{T},
) where {T}
    cooperative = IR._cooperative_normal_fill(rng, T)
    return cooperative === nothing ? (Val(:grouped), IR._device_normal_fill_group(T)) :
           (Val(:cooperative), cooperative...)
end

@inline IR._device_normal_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDANonPhilox4x32,
    ::Type{T},
) where {T<:Union{Float32,Float64}} = (Val(:cooperative), _packed_16byte_plan(T)...)

@inline function IR._transformed_fill_plan(
    ::IR._CUDABackend,
    backend::CUDA.CUDABackend,
    rng::_CUDAGenerators,
    ::Type{T},
) where {T<:Union{Float32,Float64}}
    return IR._device_uniform_fill_plan(backend, rng, T)
end

@inline function IR._device_range_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAGenerators,
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

KernelAbstractions.@kernel function _prepare_weights_cuda_fold_kernel!(
    source,
    total_result,
    invalid_result,
    cumulative,
    ::Val{validate_elements},
) where {validate_elements}
    lane = KernelAbstractions.@index(Local, Linear)
    staged = KernelAbstractions.@localmem Float64 (_CUDA_WEIGHT_FOLD_LANES,)
    total = zero(Float64)
    invalid = false
    first = 1
    while first <= length(source)
        ordinal = first + lane - 1
        if ordinal <= length(source)
            @inbounds staged[lane] = Float64(IR._population_value(source, UInt64(ordinal)))
        end
        KernelAbstractions.@synchronize

        if lane == 1
            last = min(_CUDA_WEIGHT_FOLD_LANES, length(source) - first + 1)
            @inbounds for slot = 1:last
                weight = staged[slot]
                invalid |=
                    validate_elements && (!isfinite(weight) || weight < zero(Float64))
                total += weight
                staged[slot] = total
            end
        end
        KernelAbstractions.@synchronize

        if ordinal <= length(source)
            @inbounds cumulative[ordinal] = staged[lane]
        end
        KernelAbstractions.@synchronize
        first += _CUDA_WEIGHT_FOLD_LANES
    end
    if lane == 1
        invalid |= !isfinite(total) || total <= zero(Float64)
        @inbounds begin
            total_result[1] = total
            invalid_result[1] = invalid
        end
    end
end

function IR._prepare_weight_scan(rng::_CUDAGenerators, weights, agnostic::Bool)
    source =
        agnostic ? IR._transfer_weights(rng.device, IR._collect_weights(weights)) : weights
    cumulative = IR._allocate_array(rng.device, Float64, (length(source),))
    total_result = IR._allocate_array(rng.device, Float64, (1,))
    invalid_result = IR._allocate_array(rng.device, Bool, (1,))
    backend = IR._fill_backend(cumulative)
    _prepare_weights_cuda_fold_kernel!(backend)(
        source,
        total_result,
        invalid_result,
        cumulative,
        Val(!agnostic);
        ndrange = _CUDA_WEIGHT_FOLD_LANES,
        workgroupsize = _CUDA_WEIGHT_FOLD_LANES,
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
    indices = eachindex(destination)
    @inbounds begin
        destination_index = IR._sampling_destination_index(indices, order[index])
        destination[destination_index] = IR._population_value(population, UInt64(lower))
    end
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
