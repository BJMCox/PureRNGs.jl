function _thrown_message(call)
    try
        call()
    catch err
        return sprint(showerror, err)
    end
    return ""
end

@testset "threaded fills join failed writers" begin
    destination = TaskWriteProbe(UInt32(1):UInt32(16384))
    failures = try
        rand_next!(Philox4x32(17), destination; threaded = true)
    catch err
        err
    end
    @test all(failure -> istaskfailed(failure.task), failures.exceptions)
    @test Set(failure.task for failure in failures.exceptions) == Set(
        destination.writers[i] for
        i in eachindex(destination) if isassigned(destination.writers, i)
    )
end

@testset "Fill device mismatch names both devices" begin
    rng = Philox4x32(0x6a4)
    message = _thrown_message(
        () -> first(rand_next!(rng, WrongDeviceArray(Vector{UInt32}(undef, 1)))),
    )
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
