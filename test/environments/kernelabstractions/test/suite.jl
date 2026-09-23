using KernelAbstractions
using PureRNGs
using Random
using Test

include(joinpath(@__DIR__, "..", "..", "..", "fixtures.jl"))

# The KernelAbstractions CPU backend runs the same kernels a GPU backend launches,
# so the device fill plans can be checked on any machine. A device token routes
# the generator through the device launchers; the destination stays on the host.
_device_rng(rng) = IR._rebuild(rng, rng.position, IR._AMDGPU_BACKEND)

@testset "device fill plans write every element of an offset-axis destination" begin
    n = 1001
    for F in (Philox4x32, Threefry2x32, ChaCha, Philox4x64),
        T in (Float32, Float64, UInt32, Bool),
        plan in (nothing, (Val(:grouped), Val(4)))

        rng = F(7, 3)
        expected, _ = rand_next(rng, T, n)
        destination = ZeroBasedVector(zeros(T, n))
        IR._launch_device_fill!(
            KernelAbstractions.CPU(),
            _device_rng(rng),
            destination,
            T,
            Val(:uniform),
            plan,
        )
        @test destination.data == expected
    end
end
