function _thrown_message(call)
    try
        call()
    catch err
        return sprint(showerror, err)
    end
    return ""
end

@testset "R23 untyped draw guards name both draw forms" begin
    rng = Philox4x32(0x6a1)
    cases = (
        (() -> rand(rng), "rand(rng, T)", "rand_next(rng, T)"),
        (() -> rand(rng, 3), "rand(rng, T, dims...)", "rand_next(rng, dims...)"),
        (() -> rand(rng, (2, 2)), "rand(rng, T, dims...)", "rand_next(rng, dims...)"),
        (() -> randn(rng), "randn(rng, T)", "randn_next(rng, T)"),
        (() -> randn(rng, 3), "randn(rng, T, dims...)", "randn_next(rng, dims...)"),
        (() -> randn(rng, (2, 2)), "randn(rng, T, dims...)", "randn_next(rng, dims...)"),
        (() -> randexp(rng), "randexp(rng, T)", "randexp_next(rng, T)"),
        (() -> randexp(rng, 3), "randexp(rng, T, dims...)", "randexp_next(rng, dims...)"),
        (
            () -> randexp(rng, (2, 2)),
            "randexp(rng, T, dims...)",
            "randexp_next(rng, dims...)",
        ),
    )
    for (call, held_form, next_form) in cases
        @test_throws ArgumentError call()
        message = _thrown_message(call)
        @test occursin(held_form, message)
        @test occursin(next_form, message)
    end

    # The dims guards take `Integer` and `Dims`, so the typed and range forms
    # must still reach their own methods.
    @test size(rand(rng, Float64, 3)) == (3,)
    @test size(rand(rng, Float64, (2, 2))) == (2, 2)
    @test size(rand(rng, 1:5, 3)) == (3,)
    @test size(randn(rng, Float32, (2, 2))) == (2, 2)
    @test size(randexp(rng, Float64, 3)) == (3,)
    @test size(first(rand_next(rng, 3))) == (3,)
end

@testset "Fill device mismatch names both devices" begin
    rng = Philox4x32(0x6a4)
    message = _thrown_message(() -> rand!(rng, WrongDeviceArray(Vector{UInt32}(undef, 1))))
    @test occursin(string(MLD.CPUDevice), message)
    @test occursin(string(MLD.UnknownDevice), message)
end

@testset "Seed width error names the family and the width" begin
    for (F, family, key_bits) in (
        (Philox2x32, "Philox2x32", 32),
        (Threefry4x64, "Threefry4x64", 256),
        (ChaCha, "ChaCha", 256),
    )
        message = _thrown_message(() -> F(big(1) << key_bits))
        @test occursin(family, message)
        @test occursin("$key_bits bits", message)
    end
end
