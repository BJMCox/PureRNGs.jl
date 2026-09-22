# Public scalar calls inside a kernel must retain the bound generator's stream.
function tandem_scalar_kernel!(integers, floats, states)
    i = Int(CUDA.threadIdx().x)
    if i <= length(states)
        rng = states[i]
        integers[i], rng = PR.rand_next(rng, UInt32)
        floats[i], rng = PR.rand_next(rng, Float64)
        states[i] = rng
    end
    return nothing
end

@testset "Tandem bridge in a CUDA kernel" begin
    device = PR.MLDataDevices.CUDADevice()
    to_cpu = PR.MLDataDevices.CPUDevice()
    for K in (1, 32)
        # The first draw leaves a row or group and forces a cached-state update.
        cpu = [
            TR.Tandem8x32{K}(TR.rngkey(TR.Tandem8x32(seed)), position) for
            (seed, position) in ((42, 992), (99, 32736))
        ]
        states = CuArray(device.(cpu))
        integers = CUDA.zeros(UInt32, 2)
        floats = CUDA.zeros(Float64, 2)
        CUDA.@sync CUDA.@cuda threads=2 tandem_scalar_kernel!(integers, floats, states)
        actual_integers, actual_floats, after =
            Array(integers), Array(floats), Array(states)
        for i in eachindex(cpu)
            x, rng = PR.rand_next(cpu[i], UInt32)
            y, rng = PR.rand_next(rng, Float64)
            @test actual_integers[i] == x
            @test actual_floats[i] == y
            @test to_cpu(after[i]) == rng
        end
    end
end
