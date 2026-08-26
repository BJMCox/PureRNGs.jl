module PureRNGsAMDGPUExt

import AMDGPU
import PureRNGs

const IR = PureRNGs

@inline function IR._allocate_array(::IR._AMDGPUBackend, ::Type{T}, dims::Tuple) where {T}
    return AMDGPU.ROCArray{T}(undef, dims)
end

@inline IR._materialize_population(::IR._AMDGPUBackend, population) =
    AMDGPU.ROCArray(IR._collect_population(population))

end
