module PureRNGs

import KernelAbstractions
using KernelAbstractions: @index, @localmem, @synchronize
import MLDataDevices
import Random

export Philox2x32,
    Philox4x32,
    Philox2x64,
    Philox4x64,
    Threefry2x32,
    Threefry4x32,
    Threefry2x64,
    Threefry4x64,
    Philox4x32R7,
    Threefry4x64R13,
    ChaCha,
    ChaCha8,
    ChaCha12,
    ChaCha20,
    AbstractPureRNG,
    splitrng,
    subrng,
    rngkey,
    rngposition,
    rand_next,
    rand_next!,
    randn_next,
    randn_next!,
    randexp_next,
    randexp_next!,
    randat,
    randnat,
    randexpat,
    randsample,
    randsample_next,
    randsample!,
    randsample_next!,
    WeightTable,
    StatefulRNG,
    StreamExhausted

# Both Reactant extensions dispatch on this type, so the core owns the declaration.
struct _ReactantRNG{R,A}
    state::A
end

include("core_words.jl")
include("philox.jl")
include("threefry.jl")
include("chacha.jl")
include("generators.jl")
include("cpu_allocation.jl")
include("bits.jl")
include("derive.jl")
include("uniform_scalar.jl")
include("uniform_fill.jl")
include("uniform_kernels.jl")
include("cpu_scheduler.jl")
include("transformed_fill.jl")
include("uniform.jl")
include("uniform_allocating.jl")
include("normal.jl")
include("normal_allocating.jl")
include("exponential.jl")
include("exponential_allocating.jl")
include("integers.jl")
include("range_allocating.jl")
include("sampling.jl")
include("weighted_sampling.jl")
include("stateful.jl")
include("docstrings.jl")

end # module PureRNGs
