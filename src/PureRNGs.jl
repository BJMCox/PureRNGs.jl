module PureRNGs

import KernelAbstractions
using KernelAbstractions: @index
import MLDataDevices
import Random
import Serialization

export Philox2x32,
    Philox4x32,
    Philox2x64,
    Philox4x64,
    Threefry2x32,
    Threefry4x32,
    Threefry2x64,
    Threefry4x64,
    rand_next,
    rand_next!,
    randat

include("philox.jl")
include("threefry.jl")
include("families.jl")
include("uniform.jl")
include("integers.jl")

end # module PureRNGs
