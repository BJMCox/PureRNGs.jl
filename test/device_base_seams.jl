const DeviceIR = PureRNGs
const DeviceMLD = PureRNGs.MLDataDevices
const DEVICE_VALIDATION_EVENTS = Symbol[]

struct DeviceValidationProbe{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.size(array::DeviceValidationProbe) =
    (push!(DEVICE_VALIDATION_EVENTS, :size); size(array.data))
Base.getindex(array::DeviceValidationProbe, index::Int) = array.data[index]
Base.setindex!(array::DeviceValidationProbe, value, index::Int) =
    setindex!(array.data, value, index)

function DeviceMLD.get_device_type(::DeviceValidationProbe)
    push!(DEVICE_VALIDATION_EVENTS, :device)
    return DeviceMLD.CPUDevice
end

function DeviceIR._fill_backend(device::DeviceIR._CPUBackend, array::DeviceValidationProbe)
    push!(DEVICE_VALIDATION_EVENTS, :backend)
    return device
end

function DeviceIR._check_serviceability(::Philox2x32{DeviceIR._CPUBackend}, ::Type{UInt32})
    push!(DEVICE_VALIDATION_EVENTS, :serviceability)
    return nothing
end

@testset "R38-R40 closed backend validation" begin
    cpu_rng = Philox2x32(0x751)
    cuda_rng = DeviceMLD.CUDADevice(:discarded)(cpu_rng)
    empty = DeviceValidationProbe(UInt32[])

    empty!(DEVICE_VALIDATION_EVENTS)
    @test_throws TypeError rand_next!(cpu_rng, empty; threaded = 1)
    @test isempty(DEVICE_VALIDATION_EVENTS)

    empty!(DEVICE_VALIDATION_EVENTS)
    returned, next_rng = rand_next!(cpu_rng, empty)
    @test returned === empty
    @test next_rng === cpu_rng
    @test DEVICE_VALIDATION_EVENTS[1:3] == [:device, :serviceability, :size]
    @test :backend ∉ DEVICE_VALIDATION_EVENTS

    for destination in (UInt32[], BitArray(undef, 0), falses(3))
        @test_throws ArgumentError rand!(cuda_rng, destination)
        @test_throws ArgumentError rand_next!(cuda_rng, destination)
    end

    empty!(DEVICE_VALIDATION_EVENTS)
    @test_throws ArgumentError rand_next!(cuda_rng, empty)
    @test DEVICE_VALIDATION_EVENTS == [:device]
end
