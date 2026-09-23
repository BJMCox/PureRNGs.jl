struct CategoricalDeviceProbe{T} <: AbstractVector{T}
    values::Vector{T}
end

Base.size(probabilities::CategoricalDeviceProbe) = size(probabilities.values)
Base.getindex(::CategoricalDeviceProbe, ::Int) =
    error("device probabilities must not be read")
IR.MLDataDevices.get_device(::CategoricalDeviceProbe) = IR.MLDataDevices.CUDADevice(:named)

@testset "Categorical matches raw weighted labels" begin
    probabilities = Float64[0, 1, 0, 3]
    distribution = Categorical(probabilities; check_args = false)
    rng = last(rand_next(Philox4x32(0x9d5), Bool))
    labels = 1:length(probabilities)
    expected, expected_next = randsample_next(rng, labels, probabilities, 33)
    first_expected, first_next = randsample_next(rng, labels, probabilities, 1)

    @test rand(rng, distribution) == only(first_expected)
    @test rand_next(rng, distribution) == (only(first_expected), first_next)
    @test rand_at(rng, distribution, 2) == expected[2]

    values = rand(rng, distribution, 3, 11)
    @test size(values) == (3, 11)
    @test vec(values) == expected

    continued, continued_next = rand_next(rng, distribution, 3, 11)
    @test vec(continued) == expected
    @test continued_next === expected_next

    destination = fill(-1, 33)
    @test rand!(rng, distribution, destination; threaded = false) === destination
    @test destination == expected
    returned, filled_next = rand_next!(rng, distribution, destination; threaded = false)
    @test returned === destination
    @test destination == expected
    @test filled_next === expected_next
end

@testset "Categorical serial and threaded fills agree above the threshold" begin
    probabilities = Float64[1, 2, 0, 3, 4]
    distribution = Categorical(probabilities; check_args = false)
    rng = Philox4x32(0x9d6)
    chunk = Int(IR._CPU_FILL_CHUNK_BITS ÷ UInt64(IR._WEIGHT_BITS))
    chunk -= chunk % IR._WEIGHTED_LOOKUP_LANES
    # Four chunks clear the threading threshold, and the fifth is one element.
    count = 4 * chunk + 1
    expected, expected_next = rand_next(rng, distribution, count)

    threaded = fill(-1, count)
    _, threaded_next = rand_next!(rng, distribution, threaded)
    @test threaded == expected
    @test threaded_next === expected_next

    serial = fill(-1, count)
    _, serial_next = rand_next!(rng, distribution, serial; threaded = false)
    @test serial == expected
    @test serial_next === expected_next
end

@testset "Categorical scalar calls retain a rebound binding" begin
    probabilities = Float64[0, 1, 0, 3]
    distribution = Categorical(probabilities; check_args = false)
    cpu_rng = Philox4x32(0x9d6)
    rebound_rng = IR.MLDataDevices.CUDADevice(:named)(cpu_rng)
    expected, expected_next =
        randsample_next(cpu_rng, 1:length(probabilities), probabilities, 1)

    value, next_rng = rand_next(rebound_rng, distribution)
    @test value == only(expected)
    @test next_rng.position == expected_next.position
    @test next_rng.device === rebound_rng.device
end

@testset "Categorical validates probability placement and fills atomically" begin
    rng = Philox4x32(0x9d7)
    valid = Categorical(Float64[0, 1, 0, 3]; check_args = false)
    device_probabilities =
        Categorical(CategoricalDeviceProbe([0.0, 1.0]); check_args = false)
    invalid = Categorical(zeros(3); check_args = false)

    @test_throws ArgumentError rand(rng, device_probabilities)
    @test_throws ArgumentError rand_at(rng, device_probabilities, 1)
    @test_throws ArgumentError rand(rng, device_probabilities, 0)
    @test_throws ArgumentError rand!(rng, device_probabilities, Int[]; threaded = false)
    @test_throws MethodError rand!(rng, valid, Int32[]; threaded = false)

    destination = fill(-1, 2)
    original = copy(destination)
    @test_throws ArgumentError rand_next!(rng, invalid, destination; threaded = false)
    @test destination == original

    alias_rng = Philox4x32(0x9d8)
    aliased = Int[0, 1, 0, 3]
    expected, expected_next =
        randsample_next(alias_rng, 1:length(aliased), copy(aliased), 4)
    returned, alias_next = rand_next!(
        alias_rng,
        Categorical(aliased; check_args = false),
        aliased;
        threaded = false,
    )
    @test returned === aliased
    @test aliased == expected
    @test alias_next === expected_next

    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    empty = Int[]
    @test rand!(exhausted, valid, empty; threaded = false) === empty
    @test rand_next!(exhausted, valid, empty; threaded = false) == (empty, exhausted)
    @test_throws ArgumentError rand!(rng, invalid, empty; threaded = false)

    last = IR._rebuild(
        rng,
        IR._Position64(IR._max_block(rng), IR._block_bits(rng) - IR._WEIGHT_BITS),
        rng.device,
    )
    near_terminal = fill(-1, 2)
    before = copy(near_terminal)
    @test_throws StreamExhausted rand_next!(last, valid, near_terminal; threaded = false)
    @test near_terminal == before
end
