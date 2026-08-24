module PureRNGs

import KernelAbstractions
using KernelAbstractions: @index
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
    randat

include("philox.jl")
include("threefry.jl")
include("families.jl")
include("cpu_allocation.jl")
include("bits.jl")
include("derive.jl")
include("uniform.jl")
include("uniform_allocating.jl")
include("integers.jl")

end # module PureRNGs
