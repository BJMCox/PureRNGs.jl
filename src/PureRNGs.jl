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
    splitrng,
    subrng,
    rand_next,
    rand_next!,
    randn_next,
    randn_next!,
    randat,
    randnat,
    StatefulRNG

include("philox.jl")
include("threefry.jl")
include("families.jl")
include("cpu_allocation.jl")
include("bits.jl")
include("derive.jl")
include("uniform.jl")
include("uniform_allocating.jl")
include("normal.jl")
include("normal_allocating.jl")
include("integers.jl")
include("range_allocating.jl")
include("stateful.jl")

end # module PureRNGs
