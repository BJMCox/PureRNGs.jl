module PureRNGsAMDGPUExt

import AMDGPU
import PureRNGs

const IR = PureRNGs

IR._backend_device(::IR._AMDGPUBackend) = IR.MLDataDevices.AMDGPUDevice(AMDGPU.device())

@inline IR._kernel_population(::AMDGPU.ROCDeviceArray) = true
@inline IR._pick_value(::IR._AMDGPUBackend, population, ordinal) =
    AMDGPU.@allowscalar IR._population_value(population, ordinal)

@inline function IR._allocate_array(::IR._AMDGPUBackend, ::Type{T}, dims::Tuple) where {T}
    return AMDGPU.ROCArray{T}(undef, dims)
end

@inline IR._materialize_population(::IR._AMDGPUBackend, population) =
    AMDGPU.ROCArray(IR._collect_population(population))

end
