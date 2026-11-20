using CUDA
using KernelAbstractions
using MLDataDevices
using PureRNGs
using StaticArrays
using Test

function _collection_pick_kernel!(destination, rng, population)
    destination[1], next_rng = rand_next(rng, population)
    destination[2] = rand_at(rng, population, 2)
    destination[3] = first(rand_next(next_rng, population))
    return nothing
end

@testset "CUDA scalar picks read matching arrays on the host and in kernels" begin
    CUDA.allowscalar(false)
    cpu_rng = Threefry2x32(42)
    gpu_rng = CUDADevice(CUDA.device())(cpu_rng)
    host_population = Int32[11, 22, 33, 44, 55, 66]
    gpu_population = CuArray(host_population)
    @test_throws ArgumentError rand_next(cpu_rng, gpu_population)
    @test_throws ArgumentError rand_at(cpu_rng, gpu_population, 2)
    @test_throws ArgumentError rand_next(gpu_rng, host_population)
    @test_throws ArgumentError rand_at(gpu_rng, host_population, 2)

    output = CuArray{Int32}(undef, 3)
    inline = SVector{6}(host_population)
    for (host, gpu) in (
        (host_population, gpu_population),
        (view(host_population, 2:2:6), view(gpu_population, 2:2:6)),
        (host_population, inline),
    )
        expected, expected_next = rand_next(cpu_rng, host)
        expected_at = rand_at(cpu_rng, host, 2)
        value, next_rng = rand_next(gpu_rng, gpu)
        @test value == expected
        @test next_rng.position == expected_next.position
        @test rand_at(gpu_rng, gpu, 2) == expected_at

        @cuda threads=1 _collection_pick_kernel!(output, gpu_rng, gpu)
        @test Array(output) ==
              [expected, expected_at, first(rand_next(expected_next, host))]
    end

    @test Array(first(randsample_next(gpu_rng, inline, 3))) ==
          first(randsample_next(cpu_rng, host_population, 3))
end
