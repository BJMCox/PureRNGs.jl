# Reduce the lattice in parallel chunks: the Float32 sweep visits every one of
# its 2^23 points.
function _exponential_sweep(::Type{T}, raws) where {T}
    device = IR._CPUBackend()
    chunks = collect(Iterators.partition(raws, cld(length(raws), 4Threads.nthreads())))
    lowest = Vector{T}(undef, length(chunks))
    highest = Vector{T}(undef, length(chunks))
    positive_finite = trues(length(chunks))
    Threads.@threads for index in eachindex(chunks)
        low = typemax(T)
        high = zero(T)
        healthy = true
        for raw in chunks[index]
            value = IR._exponential_from_bits(device, T, UInt64(raw))
            healthy &= value > zero(T) && isfinite(value)
            low = min(low, value)
            high = max(high, value)
        end
        lowest[index] = low
        highest[index] = high
        positive_finite[index] = healthy
    end
    return minimum(lowest), maximum(highest), all(positive_finite)
end

@testset "exponential lattice and transform" begin
    cases = (
        (
            Float32,
            UInt32,
            (0x000000, 0x000001, 0x3fffff, 0x7fffff),
            (0x33800000, 0x34400002, 0x3f317216, 0x41851592),
        ),
        (
            Float64,
            UInt64,
            (0x00000000000000, 0x00000000000001, 0x07ffffffffffff, 0x0fffffffffffff),
            (
                0x3ca0000000000000,
                0x3cb8000000000002,
                0x3fe62e42fefa39ed,
                0x40425e4f7b2737fa,
            ),
        ),
    )
    for (T, U, raw_values, expected_bits) in cases
        maximum = UInt64(last(raw_values))
        scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
        for (raw_value, expected) in zip(raw_values, expected_bits)
            raw = UInt64(raw_value)
            u, v = _reference_exponential_lattice(T, raw)
            @test IR._exponential_lattice(T, raw) === (u, v)
            @test reinterpret(U, IR._exponential_transform(IR._CPU_BACKEND, T, v)) ===
                  expected
            for token in (IR._CUDA_BACKEND, IR._AMDGPU_BACKEND, IR._METAL_BACKEND)
                @test IR._exponential_transform(token, T, v) === -Base.log(v)
            end
        end
        # Neither endpoint is on the lattice: the first and last cells give the
        # same offset from 0 and 1.
        @test IR._exponential_lattice(T, UInt64(0)) === (scale, one(T) - scale)
        @test IR._exponential_lattice(T, maximum) === (one(T) - scale, scale)
    end
end

@testset "the exponential lattice is open at both ends" begin
    # The Float32 sweep is complete, so its extremes are the reach itself; the
    # strided Float64 sweep only has to stay inside it.
    device = IR._CPUBackend()
    for (T, raws) in (
        (Float32, UInt64(0):UInt64(2^23-1)),
        (Float64, UInt64(0):UInt64(2^52÷500_000):UInt64(2^52-1)),
    )
        width = _transformed_width(T)
        smallest = -log1p(-T(2)^-(width + 1))
        largest = T(width + 1) * log(T(2))
        @test IR._exponential_from_bits(device, T, UInt64(0)) === smallest
        @test IR._exponential_from_bits(device, T, UInt64(2)^width - UInt64(1)) === largest

        low, high, positive_finite = _exponential_sweep(T, raws)
        @test positive_finite
        @test low === smallest
        @test high <= largest
    end
end

@testset "exponential scalar defaults" begin
    rng = Philox4x32(0x864)
    @test randexp_next(rng) === randexp_next(rng, Float64)
    @test_throws ArgumentError randexp(rng)
end

@testset "exponential stream" begin
    for F in GENERATOR_TYPES, T in EXPONENTIAL_TYPES
        rng = _positioned(F, 0x865, UInt64(4), UInt16(61))
        _, next_rng = randexp_next(rng, T)
        final_value, final_rng = rand_next(next_rng, UInt32)
        @test final_value === rand(next_rng, UInt32)
        @test final_rng.position ==
              _reference_position(rng, _transformed_width(T) + _uniform_width(UInt32))
    end

    rng = _packed_golden_rng(PACKED_GOLDEN_GENERATORS[1]...)
    device_rng = MLD.CUDADevice()(rng)
    @test rand(rng, Float32) === rand(device_rng, Float32)
    # The token selects the normal transform, so the two generators share
    # the midpoint the transform reads, not the value it returns.
    @test _reference_normal(rng, Float32) === _reference_normal(device_rng, Float32)
    @test _reference_exponential_lattice(
        Float32,
        _reference_extract(
            rng,
            _reference_position_block(rng.position),
            rng.position.bit,
            23,
        ),
    ) === _reference_exponential_lattice(
        Float32,
        _reference_extract(
            device_rng,
            _reference_position_block(device_rng.position),
            device_rng.position.bit,
            23,
        ),
    )
end

@testset "exponential codegen" begin
    rng = MLD.CUDADevice()(Philox4x64(0x86b))
    for (function_, signature) in (
        (randexp, Tuple{typeof(rng),Type{Float64}}),
        (randexp_next!, Tuple{typeof(rng),Vector{Float64}}),
    )
        typed_ir, llvm_ir = _codegen_ir(function_, signature)
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        @test !occursin(r"\bi128\b", llvm_ir)
    end
end

@testset "A backend token is not a fill codec" begin
    rng = Philox4x32(0x3e9)
    destination = Vector{Float64}(undef, 8)
    @test_throws MethodError PureRNGs._rand_transformed_next_fill!(
        rng,
        destination,
        false,
        rng.device,
    )
    randexp_next!(rng, destination)
    @test destination == first(randexp_next(rng, Float64, 8))
end
