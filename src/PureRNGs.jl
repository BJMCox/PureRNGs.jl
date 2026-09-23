module PureRNGs

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
    Philox2x64R6,
    Philox4x64R7,
    Threefry2x64R13,
    Threefry4x32R12,
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
    rand_at,
    randn_at,
    randexp_at,
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
include("allocation.jl")
include("bits.jl")
include("derive.jl")
include("uniform_scalar.jl")
include("validation.jl")
include("uniform_fill.jl")
include("fill_hooks.jl")
include("cpu_scheduler.jl")
include("transformed_fill.jl")
include("uniform.jl")
include("addressed.jl")
include("normal.jl")
include("exponential.jl")
include("integers.jl")
include("range_fill.jl")
include("sampling.jl")
include("weighted_sampling.jl")
include("stateful.jl")
include("docstrings.jl")
include("precompile.jl")

end # module PureRNGs
