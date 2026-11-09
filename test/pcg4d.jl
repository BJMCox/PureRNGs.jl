using Random

@testset "PCG4D stream and derivation" begin
    F = PureRNGs.PCG4D
    # Independent C port of Jarzynski and Olano (2020), section 6.1.
    expected = UInt32[
        0x0f02f829,
        0x2d568769,
        0x32b0c43b,
        0xd32548ea,
        0x0b3cefc3,
        0xa31dc1d5,
        0x5ffbe5a5,
        0x623d099e,
    ]
    root = F(0)
    values, next = rand_next(root, UInt32, 8)
    @test values == expected
    @test rngposition(next) == 256
    @test first(rand_next(root, Bool, 256)) == (collect(join(bitstring.(expected))) .== '1')

    _, shifted = rand_next(root, Bool)
    values, next = rand_next(shifted, UInt32, 7)
    @test values == [rand_at(shifted, UInt32, i) for i = 1:7]
    @test rngposition(next) == 225
    @test first(splitrng(root)) == F((0xa8553829, 0x066f8769))
    @test subrng(root, 0) == F((0x21a67829, 0x231e8769))
    @test splitrng(root, 3) == splitrng(next, 3)
    @test subrng(root, typemax(UInt64)) == subrng(next, typemax(UInt64))

    capacity = UInt128(1) << 71
    edge = F((UInt64(1) << 62) - 1, capacity - 128)
    values, exhausted = rand_next(edge, UInt32, 4)
    @test values == UInt32[0xa54d1892, 0x51fa5c67, 0x20321760, 0x52fc2208]
    @test rngposition(exhausted) == capacity
    @test_throws StreamExhausted rand_next(exhausted, Bool)

    bridge = StatefulRNG(F(123))
    Random.seed!(bridge, 456)
    @test rand(bridge, UInt32, 16) == first(rand_next(F(456), UInt32, 16))
    Random.seed!(bridge, Random.RandomDevice())
    @test iszero(rngkey(parent(bridge))[2] >> 30)
    @test rngposition(parent(bridge)) == 0
end
