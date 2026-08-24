const SeamIR = PureRNGs
const SeamKA = PureRNGs.KernelAbstractions
const SeamMLD = PureRNGs.MLDataDevices

mutable struct SeamDevice
    events::Vector{Symbol}
    active::Bool
    serviceable::Bool
end

mutable struct SeamArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    device::SeamDevice
end

const _SeamRNG = Union{Philox4x32{SeamDevice},Philox4x64{SeamDevice}}

Base.size(array::SeamArray) = size(array.data)
Base.axes(array::SeamArray) = axes(array.data)
Base.IndexStyle(::Type{<:SeamArray}) = IndexLinear()
Base.getindex(array::SeamArray, index::Int) = array.data[index]
function Base.setindex!(array::SeamArray, value, index::Int)
    push!(array.device.events, array.device.active ? :write_inside : :write_outside)
    return setindex!(array.data, value, index)
end

function SeamMLD.get_device(array::SeamArray)
    push!(array.device.events, :destination_device)
    return array.device
end

function SeamKA.get_backend(array::SeamArray)
    push!(array.device.events, array.device.active ? :backend_inside : :backend_outside)
    return SeamKA.CPU()
end

function SeamIR._check_serviceability(rng::_SeamRNG, ::Type)
    push!(rng.device.events, :serviceability)
    rng.device.serviceable || throw(ArgumentError("fake result is not serviceable"))
    return nothing
end

function SeamIR._allocate_array(device::SeamDevice, ::Type{T}, dims::Tuple) where {T}
    push!(device.events, :allocate)
    return SeamArray(Array{T}(undef, dims), device)
end

function SeamIR._with_device(f, device::SeamDevice)
    push!(device.events, :context_enter)
    device.active = true
    try
        return f()
    finally
        device.active = false
        push!(device.events, :context_exit)
    end
end

_seam_position_code(position::SeamIR._Position64) =
    (position.block << 8) | UInt64(position.bit)
_seam_position_code(position::SeamIR._Position128) =
    xor(xor(position.lo, position.hi << 16), UInt64(position.bit))

SeamIR._draw_unchecked(::_SeamRNG, position, ::Type{UInt64}) = _seam_position_code(position)

SeamIR._draw_normal_unchecked(::Philox4x32{SeamDevice}, position, ::Type{Float64}) =
    Float64(_seam_position_code(position))

SeamIR._draw_range_unchecked(
    ::Philox4x32{SeamDevice},
    position,
    ::AbstractRange{UInt64},
    ::UInt64,
) = _seam_position_code(position)

function _seam_rng(device::SeamDevice; block::UInt64 = UInt64(5), bit::UInt16 = UInt16(61))
    base = Philox4x32(0x7a1)
    return SeamIR._rebuild(base, SeamIR._Position64(block, bit), device)
end

function _seam_rng128(
    device::SeamDevice;
    lo::UInt64 = UInt64(5),
    hi::UInt64 = UInt64(0),
    bit::UInt16 = UInt16(61),
)
    base = Philox4x64(0x7a1)
    return SeamIR._rebuild(base, SeamIR._Position128(lo, hi, bit), device)
end

function _seam_expected_positions(rng, width::UInt16, count::Int, ::Type{T}) where {T}
    values = Vector{T}(undef, count)
    for index in eachindex(values)
        bits_lo, bits_hi = SeamIR._bit_span(UInt64(index - 1), width)
        position = SeamIR._advance_position_unchecked(rng, bits_lo, bits_hi)
        values[index] = T(_seam_position_code(position))
    end
    return values
end

@testset "private backend-neutral validation and context seams" begin
    events = Symbol[]
    device = SeamDevice(events, false, true)
    rng = _seam_rng(device)
    destination = SeamArray(Vector{UInt64}(undef, 0), device)

    @test SeamIR._check_fill_device(rng, destination) === device
    @test events == [:destination_device]

    empty_next, empty_result = SeamIR._rand_next_fill!(rng, destination, true)
    @test empty_result === destination
    @test empty_next === rng
    @test events == [:destination_device, :destination_device, :serviceability]
    @test !device.active

    empty!(events)
    unserviceable = SeamDevice(events, false, false)
    bad_rng = _seam_rng(unserviceable)
    bad_destination = SeamArray(Vector{UInt64}(undef, 0), unserviceable)
    @test_throws ArgumentError SeamIR._rand_next_fill!(bad_rng, bad_destination, true)
    @test events == [:destination_device, :serviceability]

    empty!(events)
    other = SeamDevice(events, false, true)
    mismatched = SeamArray(Vector{UInt64}(undef, 0), other)
    @test_throws ArgumentError SeamIR._rand_next_fill!(rng, mismatched, true)
    @test events == [:destination_device]

    empty!(events)
    @test_throws ErrorException SeamIR._with_device(device) do
        @test device.active
        error("fake context failure")
    end
    @test events == [:context_enter, :context_exit]
    @test !device.active
end

@testset "generic positioned workitems stay inside device context" begin
    for (T, width, fill!) in (
        (UInt64, UInt16(64), SeamIR._rand_next_fill!),
        (Float64, UInt16(52), SeamIR._randn_next_fill!),
    )
        events = Symbol[]
        device = SeamDevice(events, false, true)
        rng = _seam_rng(device)
        destination = SeamArray(Vector{T}(undef, 5), device)
        next_rng, result = fill!(rng, destination, true)
        SeamKA.synchronize(SeamKA.CPU())

        @test result === destination
        @test destination.data == _seam_expected_positions(rng, width, 5, T)
        @test next_rng.position == _reference_position(rng, 5 * Int(width))
        @test events[1:4] ==
              [:destination_device, :serviceability, :context_enter, :backend_inside]
        @test count(==(:write_inside), events) == 5
        @test :write_outside ∉ events
        @test last(events) == :context_exit
        @test !device.active
    end

    events = Symbol[]
    device = SeamDevice(events, false, true)
    rng = _seam_rng(device)
    range = UInt64(0):typemax(UInt64)
    next_rng, result = SeamIR._rand_next_range_array(rng, range, (5,))
    SeamKA.synchronize(SeamKA.CPU())

    @test result.data == _seam_expected_positions(rng, UInt16(128), 5, UInt64)
    @test next_rng.position == _reference_position(rng, 5 * 128)
    @test events[1:5] ==
          [:serviceability, :allocate, :destination_device, :serviceability, :context_enter]
    @test :backend_inside in events
    @test count(==(:write_inside), events) == 5
    @test last(events) == :context_exit

    events = Symbol[]
    device = SeamDevice(events, false, true)
    rng128 = _seam_rng128(device; lo = typemax(UInt64), hi = UInt64(7), bit = UInt16(248))
    destination128 = SeamArray(Vector{UInt64}(undef, 3), device)
    next128, _ = SeamIR._rand_next_fill!(rng128, destination128, true)
    SeamKA.synchronize(SeamKA.CPU())

    @test destination128.data == _seam_expected_positions(rng128, UInt16(64), 3, UInt64)
    @test destination128[2] ==
          _seam_position_code(SeamIR._Position128(UInt64(0), UInt64(8), UInt16(56)))
    @test next128.position == SeamIR._Position128(UInt64(0), UInt64(8), UInt16(184))
end

@testset "private allocation seams and public CPU hold" begin
    for (helper, argument) in
        ((SeamIR._rand_next_uniform_array, UInt32), (SeamIR._randn_next_array, Float32))
        events = Symbol[]
        device = SeamDevice(events, false, true)
        rng = _seam_rng(device)
        next_rng, destination = helper(rng, argument, (0, 2))
        @test size(destination) == (0, 2)
        @test next_rng === rng
        @test events == [:serviceability, :allocate, :destination_device, :serviceability]

        empty!(events)
        @test_throws ArgumentError helper(rng, argument, (-1,))
        @test events == [:serviceability, :allocate]
    end

    events = Symbol[]
    unserviceable = SeamDevice(events, false, false)
    bad_rng = _seam_rng(unserviceable)
    @test_throws ArgumentError SeamIR._rand_next_uniform_array(bad_rng, UInt32, (-1,))
    @test events == [:serviceability]

    range = UInt16(2):UInt16(17)
    empty!(events)
    unserviceable.serviceable = true
    next_rng, destination = SeamIR._rand_next_range_array(bad_rng, range, (0,))
    @test isempty(destination)
    @test next_rng === bad_rng
    @test events == [:serviceability, :allocate, :destination_device, :serviceability]

    cpu_rng = Philox4x32(0x7a2)
    fake_rng = _seam_rng(SeamDevice(Symbol[], false, true))
    @test applicable(rand, cpu_rng, UInt32, 2)
    @test applicable(randn, cpu_rng, Float32, 2)
    @test applicable(rand, cpu_rng, range, 2)
    @test !applicable(rand, fake_rng, UInt32, 2)
    @test !applicable(randn, fake_rng, Float32, 2)
    @test !applicable(rand, fake_rng, range, 2)
    @test !applicable(fake_rng.device, cpu_rng)

    cpu = SeamMLD.CPUDevice()
    @test @inferred(SeamIR._with_device(() -> 17, cpu)) == 17
    SeamIR._with_device(() -> nothing, cpu)
    @test @allocated(SeamIR._with_device(() -> nothing, cpu)) == 0
end
