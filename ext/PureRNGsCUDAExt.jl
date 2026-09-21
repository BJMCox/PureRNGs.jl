module PureRNGsCUDAExt

import CUDA
import PureRNGs
import KernelAbstractions
using KernelAbstractions: @index, @localmem, @synchronize

const IR = PureRNGs
const _CUDAGenerators = IR._BackendGenerators{IR._CUDABackend}
const _CUDAPhilox4x32 = IR.Philox4x32{IR._CUDABackend}
const _CUDAThreefry4x32 = IR.Threefry4x32{IR._CUDABackend}
const _CUDAPhilox2x64 = IR.Philox2x64{IR._CUDABackend}
const _CUDAPhilox4x64 = IR.Philox4x64{IR._CUDABackend}
const _CUDAThreefry4x64 = IR.Threefry4x64{IR._CUDABackend}
const _CUDAChaCha = IR.ChaCha{IR._CUDABackend}
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
# The two-word generators are absent because their grouped store already
# coalesces: Threefry2x64 measures 881 against 861 GiB/s on a 2^27 `UInt64` fill.
const _CUDAPackedIntegerGenerators =
    Union{_CUDAPhilox2x64,_CUDAPhilox4x64,_CUDAThreefry4x64,_CUDAChaCha}
const _CUDA_FILL_THREADS = 256
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
const _CUDAPackedCodec = Union{
    Val{:uniform},
    IR._NormalCodec{IR._CUDABackend},
    IR._ExponentialCodec{IR._CUDABackend},
    IR._MappedFillCodec,
}

# The portable four-product multiply costs Philox2x64 1.9x on sm_80, where one
# `mul.hi.u64` gives the high half. `widemul` would give it too, but [R30]
# forbids a 128-bit integer in device typed IR, so the high half comes from
# libdevice. The override reaches device code only, and the host keeps the
# portable form that its own range arithmetic shares.
@inline _mul_hi_u64(a::UInt64, b::UInt64) =
    ccall("extern __nv_umul64hi", llvmcall, UInt64, (UInt64, UInt64), a, b)

CUDA.@device_override @inline IR._mulhilo64(a::UInt64, b::UInt64) =
    (_mul_hi_u64(a, b), a * b)

@inline _bool_packs_per_block(rng) = Val(Int(IR._block_bits(rng)) ÷ 16)

struct _CUDANatural128Pack{T}
    lanes::_CUDA_U32X4
end

const _CUDAUniformLikeCodec =
    Union{Val{:uniform},IR._ExponentialCodec{IR._CUDABackend},IR._MappedFillCodec}

# Elements one work item writes in a grouped fill. The exponential and the
# mapped codecs share the uniform table: they draw one value of the same width.
@inline _fill_group_elements(::_CUDAUniformLikeCodec, rng, ::Type{Bool}) = Val(4)
@inline _fill_group_elements(
    ::_CUDAUniformLikeCodec,
    rng::IR._NarrowGenerators,
    ::Type{UInt32},
) = Val(2)
@inline _fill_group_elements(
    ::_CUDAUniformLikeCodec,
    rng::IR._Position64Generators,
    ::Type{UInt32},
) = Val(4)
@inline _fill_group_elements(
    ::_CUDAUniformLikeCodec,
    rng::IR._Position128Generators,
    ::Type{UInt32},
) = Val(8)
@inline _fill_group_elements(::_CUDAUniformLikeCodec, rng::IR.ChaCha, ::Type{UInt32}) =
    Val(16)
@inline _fill_group_elements(codec::_CUDAUniformLikeCodec, rng, ::Type{Int32}) =
    _fill_group_elements(codec, rng, UInt32)
@inline _fill_group_elements(
    ::_CUDAUniformLikeCodec,
    rng::IR._NarrowGenerators,
    ::Type{UInt64},
) = Val(1)
@inline _fill_group_elements(
    ::_CUDAUniformLikeCodec,
    rng::IR._Position64Generators,
    ::Type{UInt64},
) = Val(2)
@inline _fill_group_elements(
    ::_CUDAUniformLikeCodec,
    rng::IR._Position128Generators,
    ::Type{UInt64},
) = Val(4)
@inline _fill_group_elements(::_CUDAUniformLikeCodec, rng::IR.ChaCha, ::Type{UInt64}) =
    Val(8)
@inline _fill_group_elements(codec::_CUDAUniformLikeCodec, rng, ::Type{Int64}) =
    _fill_group_elements(codec, rng, UInt64)
@inline _fill_group_elements(::_CUDAUniformLikeCodec, rng, ::Type{Float32}) = Val(4)
@inline _fill_group_elements(::_CUDAUniformLikeCodec, rng, ::Type{Float64}) = Val(4)
@inline _fill_group_elements(::IR._NormalCodec{IR._CUDABackend}, rng, ::Type{Float32}) =
    Val(8)
@inline _fill_group_elements(::IR._NormalCodec{IR._CUDABackend}, rng, ::Type{Float64}) =
    Val(4)

# A100 trials picked these output tiles and workgroup sizes.
@inline _cooperative_uniform_fill(::IR.Philox4x32, ::Type{Bool}) = (Val(4096), Val(32))
@inline _cooperative_uniform_fill(::IR.Philox4x32, ::Type{Float32}) = (Val(2048), Val(32))
@inline _cooperative_uniform_fill(::IR.Philox4x32, ::Type{Float64}) = (Val(1024), Val(64))
@inline _cooperative_uniform_fill(rng, T) = nothing

@inline _cooperative_normal_fill(::IR.Philox4x32, ::Type{Float32}) = (Val(512), Val(32))
@inline _cooperative_normal_fill(::IR.Philox4x32, ::Type{Float64}) = (Val(512), Val(32))
@inline _cooperative_normal_fill(rng, T) = nothing

@inline IR._device_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDAPhilox4x32,
    ::Val{:uniform},
    ::Type{T},
) where {T<:IR._UniformInteger} = (Val(:natural128_packed),)

@inline IR._device_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDAThreefry4x32,
    ::Val{:uniform},
    ::Type{T},
) where {T<:IR._UniformInteger} = (Val(:natural128_packed),)

@inline IR._device_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAPhilox4x32,
    ::Val{:uniform},
    ::Type{Float32},
) = (Val(:cooperative), _cooperative_uniform_fill(rng, Float32)..., Val(4))

# A100 trials: a 2048-output tile for every result type, with the workgroup one
# warp wide for a four-output store and two for a two-output store. The small
# workgroup wins because the kernel's one barrier is free inside a warp and
# stalls the group across warps.
@inline _packed_16byte_plan(::Type{T}) where {T<:Union{UInt32,Int32,Float32}} =
    (Val(2048), Val(32), Val(4))
@inline _packed_16byte_plan(::Type{T}) where {T<:Union{UInt64,Int64,Float64}} =
    (Val(2048), Val(64), Val(2))
@inline IR._device_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDANonPhilox4x32,
    ::Val{:uniform},
    ::Type{T},
) where {T<:Union{Float32,Float64}} = (Val(:cooperative), _packed_16byte_plan(T)...)

@inline IR._device_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDAPackedIntegerGenerators,
    ::Val{:uniform},
    ::Type{T},
) where {T<:IR._UniformInteger} = (Val(:cooperative), _packed_16byte_plan(T)...)

@inline function IR._device_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAGenerators,
    ::Val{:uniform},
    ::Type{T},
) where {T}
    cooperative = _cooperative_uniform_fill(rng, T)
    cooperative === nothing || return (Val(:cooperative), cooperative...)
    T === Bool && return (Val(:bool_blocks), _bool_packs_per_block(rng))
    return (Val(:grouped), _fill_group_elements(Val(:uniform), rng, T))
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
@inline _packed_integer_min_length(::_CUDAChaCha, ::Type{<:IR._UniformInteger}) = 4096

@inline _stream_block_offset(block::UInt64, offset::UInt64) = block + offset
@inline function _stream_block_offset(block::NTuple{2,UInt64}, offset::UInt64)
    lo = block[1] + offset
    return lo, block[2] + UInt64(lo < block[1])
end

@inline _cooperative_shared_words(::Val{B}, ::Val{O}, ::Val{W}) where {B,O,W} =
    (B ÷ 64) * cld((B - 1) + O * W, B)

KernelAbstractions.@kernel function _cooperative_fill_kernel!(
    rng,
    destination,
    ::Val{T},
    ::Val{D},
    ::Val{W},
    block_width::Val{B},
    ::Val{O},
    ::Val{L},
    ::Val{P},
    ::Val{S},
    codec,
) where {T,D,W,B,O,L,P,S}
    destination = reinterpret(D, vec(destination))
    group = @index(Group, Linear)
    lane = @index(Local, Linear)
    shared = @localmem UInt64 (
        S ? 2 * cld(O * W, B) : _cooperative_shared_words(block_width, Val(O), Val(W)),
    )
    if S
        blocks = cld(O * W, B)
        first_block = IR._position_block(rng.position) + UInt64((group - 1) * blocks)
        block_offset = lane - 1
        while block_offset < blocks
            block_words = IR._block_words(rng, first_block + UInt64(block_offset))
            @inbounds begin
                shared[2block_offset+1] = block_words[1]
                shared[2block_offset+2] = block_words[2]
            end
            block_offset += L
        end
    else
        first = (group - 1) * O + 1
        outputs = min(O, P * length(destination) - first + 1)
        bits_lo, bits_hi = IR._bit_span(UInt64(first - 1), UInt16(W))
        position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
        block_bits = B
        blocks = cld(Int(position.bit) + outputs * W, block_bits)

        block_offset = lane - 1
        while block_offset < blocks
            block = _stream_block_offset(IR._position_block(position), UInt64(block_offset))
            block_words = IR._block_words(rng, block)
            shared_first = block_offset * length(block_words)
            @inbounds for word in eachindex(block_words)
                shared[shared_first+word] = block_words[word]
            end
            block_offset += L
        end
    end
    @synchronize

    write_group = @index(Group, Linear)
    write_lane = @index(Local, Linear)
    if S
        pack = write_lane - 1
        packs = O ÷ P
        first_pack = (write_group - 1) * packs + 1
        while pack < packs
            @inbounds destination[first_pack+pack] =
                _cooperative_pack(codec, T, shared, P * pack * W, Val(P), Val(W))
            pack += L
        end
    else
        write_first = (write_group - 1) * O + 1
        write_outputs = min(O, P * length(destination) - write_first + 1)
        write_bits_lo, write_bits_hi = IR._bit_span(UInt64(write_first - 1), UInt16(W))
        write_position = IR._advance_position_unchecked(rng, write_bits_lo, write_bits_hi)
        if P == 1
            output = write_lane - 1
            while output < write_outputs
                raw = IR._local_dense_bits(
                    shared,
                    Int(write_position.bit) + output * W,
                    Val(W),
                )
                @inbounds destination[write_first+output] =
                    IR._cooperative_value(codec, T, raw)
                output += L
            end
        else
            output = P * (write_lane - 1)
            while output < write_outputs
                bit = Int(write_position.bit) + output * W
                index = ((write_first - 1) + output) ÷ P + 1
                @inbounds destination[index] =
                    _cooperative_pack(codec, T, shared, bit, Val(P), Val(W))
                output += P * L
            end
        end
    end
end

@inline function _cooperative_pack(
    codec,
    T,
    shared,
    bit,
    outputs_per_store::Val{P},
    ::Val{W},
) where {P,W}
    return ntuple(outputs_per_store) do lane
        raw = IR._local_dense_bits(shared, bit + (lane - 1) * W, Val(W))
        VecElement(IR._cooperative_value(codec, T, raw))
    end
end

KernelAbstractions.@kernel function _bool_blocks_fill_kernel!(
    rng,
    destination,
    ::Val{P},
) where {P}
    block_ordinal = @index(Global, Linear)
    stride = KernelAbstractions.@ndrange()[1]
    block_count = length(destination) ÷ P
    while block_ordinal <= block_count
        bits_lo, bits_hi = IR._bit_span(UInt64(block_ordinal - 1), IR._block_bits(rng))
        position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
        block_words = IR._block_words(rng, IR._position_block(position))
        first_pack = (block_ordinal - 1) * P + 1
        pack = 0
        while pack < P
            @inbounds destination[first_pack+pack] =
                _cooperative_pack(Val(:uniform), Bool, block_words, 16pack, Val(16), Val(1))
            pack += 1
        end
        block_ordinal += stride
    end
end

@inline _outputs_per_store(::Tuple{Val{:cooperative},Val{O},Val{L}}) where {O,L} = Val(1)
@inline _outputs_per_store(
    plan::Tuple{Val{:cooperative},Val{O},Val{L},Val{P}},
) where {O,L,P} = plan[4]

@inline function _launch_cooperative_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    plan,
    stream_aligned::Val{S},
    ::Type{D} = eltype(destination),
) where {T,S,D}
    outputs, workgroup = plan[2], plan[3]
    outputs_per_store = _outputs_per_store(plan)
    output_count = IR._val_count(outputs)
    workgroup_size = IR._val_count(workgroup)
    groups = cld(length(destination), output_count)
    _cooperative_fill_kernel!(backend)(
        rng,
        destination,
        Val(T),
        Val(D),
        Val(IR._fill_width(codec, T)),
        Val(Int(IR._block_bits(rng))),
        outputs,
        workgroup,
        outputs_per_store,
        stream_aligned,
        codec;
        ndrange = groups * workgroup_size,
        workgroupsize = workgroup_size,
    )
    return destination
end

@inline function IR._launch_device_fill!(
    backend::CUDA.CUDABackend,
    rng,
    destination,
    ::Type{T},
    codec,
    plan::Tuple{Val{:cooperative},Val{O},Val{L}},
) where {T,O,L}
    return _launch_cooperative_fill!(backend, rng, destination, T, codec, plan, Val(false))
end

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
            (Val(:grouped), _fill_group_elements(codec, rng, T)),
        )
    end

    _launch_cooperative_fill!(
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
    _launch_cooperative_fill!(
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

KernelAbstractions.@kernel function _natural128_fill_kernel!(
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
            (Val(:grouped), _fill_group_elements(codec, rng, Bool)),
        )
    end

    packed = reinterpret(_CUDA_B8X16, vec(destination))
    workitems = length(packed) ÷ P
    blocks = _cuda_packed_blocks(workitems)
    _bool_blocks_fill_kernel!(backend)(
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
    iszero(rng.position.bit) && iszero(length(destination) % IR._val_count(outputs))

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
            (Val(:grouped), _fill_group_elements(codec, rng, T)),
        )
    end

    storage_type = _CUDANatural128Pack{T}
    blocks = _cuda_packed_blocks(length(destination) ÷ (_CUDA_FILL_ALIGNMENT ÷ sizeof(T)))
    _natural128_fill_kernel!(backend)(
        rng,
        destination,
        Val(storage_type);
        ndrange = blocks * _CUDA_FILL_THREADS,
        workgroupsize = _CUDA_FILL_THREADS,
    )
    return destination
end

@inline function IR._device_fill_plan(
    ::CUDA.CUDABackend,
    rng::_CUDAGenerators,
    codec::IR._NormalCodec{IR._CUDABackend},
    ::Type{T},
) where {T}
    cooperative = _cooperative_normal_fill(rng, T)
    return cooperative === nothing ? (Val(:grouped), _fill_group_elements(codec, rng, T)) :
           (Val(:cooperative), cooperative...)
end

@inline IR._device_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDANonPhilox4x32,
    ::IR._NormalCodec{IR._CUDABackend},
    ::Type{T},
) where {T<:Union{Float32,Float64}} = (Val(:cooperative), _packed_16byte_plan(T)...)

@inline function IR._device_fill_plan(
    backend::CUDA.CUDABackend,
    rng::_CUDAGenerators,
    ::IR._ExponentialCodec{IR._CUDABackend},
    ::Type{T},
) where {T<:Union{Float32,Float64}}
    return IR._device_fill_plan(backend, rng, Val(:uniform), T)
end

@inline _grouped_candidate_plan(span::UInt64) =
    IR._range_bits(span) == UInt16(128) ? nothing : (Val(:grouped), Val(2))

@inline IR._device_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDAGenerators,
    codec::IR._RangeCodec,
    ::Type,
) = _grouped_candidate_plan(codec.span)

@inline IR._device_fill_plan(
    ::CUDA.CUDABackend,
    ::_CUDAGenerators,
    codec::IR._PopulationCodec,
    ::Type,
) = _grouped_candidate_plan(codec.cardinality)

@inline function IR._allocate_array(::IR._CUDABackend, ::Type{T}, dims::Tuple) where {T}
    return CUDA.CuArray{T}(undef, dims)
end

@inline IR._materialize_population(::IR._CUDABackend, population) =
    CUDA.CuArray(IR._collect_population(population))

end
