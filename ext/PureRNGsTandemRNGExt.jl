module PureRNGsTandemRNGExt

import PureRNGs
import TandemRNG
using PrecompileTools: @setup_workload, @compile_workload

const TR = TandemRNG
const Generator = Union{TR.Tandem8x32,TR._ReactantRNG}

PureRNGs.rngkey(rng::Generator) = TR.rngkey(rng)
PureRNGs.rngposition(rng::Generator) = TR.rngposition(rng)
PureRNGs.rand_next(rng::Generator) = TR.rand_next(rng, Float64)
PureRNGs.rand_next(rng::Generator, ::Type{T}) where {T<:TR.DrawTypes} = TR.rand_next(rng, T)
PureRNGs.rand_at(rng::Generator, ::Type{T}, i::Integer) where {T<:TR.DrawTypes} =
    TR.rand_at(rng, T, i)

function PureRNGs.rand_next!(rng::Generator, destination::AbstractArray; threaded = true)
    PureRNGs._check_threaded(threaded)
    next_rng = TR.rand_fill!(rng, destination; nthreads = threaded ? Threads.nthreads() : 1)
    return destination, next_rng
end

PureRNGs.rand_next(rng::Generator, ::Type{T}, dims::Dims) where {T<:TR.DrawTypes} =
    TR.rand_next(rng, T, dims)

PureRNGs.rand_next(
    rng::Generator,
    ::Type{T},
    n::Integer,
    dims::Integer...,
) where {T<:TR.DrawTypes} = PureRNGs.rand_next(rng, T, Int.((n, dims...)))
PureRNGs.rand_next(rng::Generator, dims::Dims) = PureRNGs.rand_next(rng, Float64, dims)
PureRNGs.rand_next(rng::Generator, n::Integer, dims::Integer...) =
    PureRNGs.rand_next(rng, Float64, Int.((n, dims...)))

PureRNGs.splitrng(rng::Generator) = TR.splitrng(rng)
PureRNGs.splitrng(rng::Generator, n::Val) = TR.splitrng(rng, n)
function PureRNGs.splitrng(rng::TR.Tandem8x32, n::Integer; threaded = true)
    PureRNGs._check_threaded(threaded)
    return TR.splitrng(rng, n; threaded)
end
PureRNGs.subrng(rng::Generator, purpose::Integer) = TR.subrng(rng, purpose)
PureRNGs.StatefulRNG(rng::TR.Tandem8x32) = TR.Stateful(rng)

@setup_workload begin
    @compile_workload for K in (1, 32, 64)
        rng = TR.Tandem8x32{K}(42)
        for T in (
            Bool,
            UInt8,
            Int8,
            UInt16,
            Int16,
            UInt32,
            Int32,
            UInt64,
            Int64,
            Float32,
            Float64,
        )
            _, next_rng = PureRNGs.rand_next(rng, T)
            PureRNGs.rand_at(next_rng, T, 3)
            PureRNGs.rand_next(next_rng, T, 3, 5)
            PureRNGs.rand_next!(next_rng, Vector{T}(undef, 17); threaded = false)
        end
        PureRNGs.rand_next(rng)
        PureRNGs.rand_next(rng, (3, 5))
        PureRNGs.rand_next(rng, 3, 5)
        PureRNGs.splitrng(rng)
        PureRNGs.splitrng(rng, Val(2))
        PureRNGs.splitrng(rng, 3)
        PureRNGs.splitrng(rng, 3; threaded = false)
        PureRNGs.subrng(rng, 3)
        PureRNGs.StatefulRNG(rng)
        for device in (
            PureRNGs.MLDataDevices.CUDADevice(),
            PureRNGs.MLDataDevices.AMDGPUDevice(),
            PureRNGs.MLDataDevices.MetalDevice(),
        )
            bound = device(rng)
            for T in Base.uniontypes(TR.DrawTypes)
                PureRNGs.rand_next(bound, T)
                PureRNGs.rand_at(bound, T, 3)
            end
            PureRNGs.splitrng(bound, Val(2))
            PureRNGs.splitrng(bound, 3; threaded = false)
            PureRNGs.subrng(bound, 3)
            PureRNGs.StatefulRNG(bound)
        end
    end
end

end
