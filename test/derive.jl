zero_position(rng) = PureRNGs._zero_position(typeof(rng))

@testset "R20 derivation state law" begin
    for F in (Philox2x32, Threefry4x64)
        rng = F(123)
        moved = PureRNGs._reserve(rng, UInt64(7), UInt64(0))
        children = splitrng(rng)
        @test children == splitrng(rng, Val(2))
        @test getfield.(children, :key) == getfield.(splitrng(moved), :key)
        @test subrng(rng, 42).key == subrng(moved, 42).key
        @test all(child -> child.position == zero_position(child), children)
        @test all(child -> child.device == rng.device, children)
        @test subrng(rng, 42).position == zero_position(rng)
        @test subrng(rng, 42).device == rng.device
        @test rng.position == zero_position(rng)
    end

    rng = Philox4x32(123)
    @test subrng(rng, -1) == subrng(rng, typemax(UInt64))
    @test subrng(rng, (UInt128(1) << 64) + UInt128(42)) == subrng(rng, 42)
end

@testset "R13 derivation golden vectors" begin
    # The six generators outside the pinned testbed scope carry revision-9 vectors here.
    # The two pinned-testbed generators and their provenance live in oracle_conformance.jl.
    cases = (
        (
            Philox2x32((0x01234567,)),
            ((0x9830e21c,), (0xb1b55909,), (0xf95ec6ee,), (0x5a2a623f,)),
            (0x4a675769,),
            (0x885a87e9,),
        ),
        (
            Philox2x64((0x0123456789abcdef,)),
            (
                (0x1991b350a85dfec2,),
                (0xf9b187fd214d8989,),
                (0x49674cdc03f67a5d,),
                (0xb84d7c1804ca5d96,),
            ),
            (0x2e81e95f10a09d91,),
            (0x8292b662cef9dd4c,),
        ),
        (
            Philox4x64((0x0123456789abcdef, 0xfedcba9876543210)),
            (
                (0x4a4d442aaf48371c, 0xbdce2c1216f44cd7),
                (0x90b1bfb6cc00768e, 0x4d64843a5c826faa),
                (0xde40be8f1bc7b67c, 0x316c2dd2731e4d9e),
                (0x1d5b5f52c8b8dd0e, 0x64fcc521c7981751),
            ),
            (0x04e6f77d1469f19a, 0x868e4c489c3d9f1a),
            (0x3bedff3a0e54ebae, 0x19b88669e9d9140b),
        ),
        (
            Threefry4x32((0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210)),
            (
                (0xdf4895c4, 0xf9371fc9, 0xc9bc1f01, 0x964be5eb),
                (0xb5f494a9, 0x164793aa, 0x65f0aaf7, 0xbbd97091),
                (0x1048c204, 0x5a37e95f, 0xc00dadf0, 0xa082c1af),
                (0xdd4b9de4, 0x163305a7, 0x03ad0110, 0x909898ea),
            ),
            (0x49881994, 0xa1fe252b, 0x6b3c4feb, 0x3940a3d6),
            (0x016708b5, 0xf3372999, 0x08eb2305, 0x37ff6840),
        ),
        (
            Threefry2x64((0x0123456789abcdef, 0xfedcba9876543210)),
            (
                (0x432bb976501e4c40, 0x004805f499c1473c),
                (0xa710999647f2b740, 0xe8162c41fb93a00e),
                (0xc6dad1d789f14370, 0xe2667b7fa43fb2c9),
                (0xcc56d074e59c2052, 0x52edc9bd44314400),
            ),
            (0x6836648b3c9b496e, 0x7aec512ac0742548),
            (0xe40e74837daf34f3, 0x942102879e7c183f),
        ),
        (
            Threefry4x64((
                0x0123456789abcdef,
                0xfedcba9876543210,
                0x0f1e2d3c4b5a6978,
                0x8877665544332211,
            ),),
            (
                (
                    0xf12b6ab2a01a49f8,
                    0x2b968d7f69cf9674,
                    0xd6152fe12f820576,
                    0x6fc14405552e114c,
                ),
                (
                    0xd34b432ac9f91211,
                    0xed32682c416fc144,
                    0x1ec5d9b70bd45593,
                    0x9e06e378f1f6a5ab,
                ),
                (
                    0x85ff297aa5291ffa,
                    0x7524d2baa55e6835,
                    0x7d2935c16e9626d6,
                    0x0c6cb77bd9065edb,
                ),
                (
                    0xd2bbafe014584b6e,
                    0x3878148982785501,
                    0x54f1ab7799a4bd7c,
                    0x260d7d49295ad0bf,
                ),
            ),
            (
                0xe56d1ea79a0d269f,
                0x8478140215094879,
                0xf159ef490e393296,
                0x4d29ffcd0244b816,
            ),
            (
                0x8bdd86c700a83269,
                0xd93a0a423594628e,
                0xa600e80a472e2294,
                0x6cb0fe23bdf7b67d,
            ),
        ),
    )

    for (rng, split_keys, zero_key, purpose_key) in cases
        @test getfield.(splitrng(rng, Val(4)), :key) == split_keys
        @test subrng(rng, 0).key == zero_key
        @test subrng(rng, 0x0123456789abcdef).key == purpose_key
    end

    @test PureRNGs._derive_child(Philox2x32((0x01234567,)), UInt64(0xfffffffe)).key ==
          (0xc6dcc08a,)
end

@testset "R21 request forms and bounds" begin
    for F in (Philox4x32,)
        rng = F(123)
        @test splitrng(rng, Val(0)) === ()
        @test splitrng(rng, 0) == typeof(rng)[]
        @test splitrng(rng, 3) == collect(splitrng(rng, Val(3)))
        @test splitrng(rng, Int8(3)) == collect(splitrng(rng, Val(3)))
        @test_throws ArgumentError splitrng(rng, -1)
        @test_throws ArgumentError splitrng(rng, big(typemax(Int)) + 1)
        @test_throws ArgumentError splitrng(rng, Val(-1))
        @test_throws ArgumentError splitrng(rng, Val(UInt32(2)))
    end

    for rng in (Philox2x32(123), Threefry2x32(123))
        @test_throws ArgumentError splitrng(rng, UInt64(0x1_0000_0000))
    end
end

@testset "R21 threaded derivation" begin
    for F in (Philox4x32, Threefry2x64, ChaCha)
        rng = F(123)
        # 4096 is the chunk size, so these counts straddle a chunk boundary.
        for count in (0, 1, 4095, 4096, 4097, 100_000)
            @test splitrng(rng, count; threaded = true) ==
                  splitrng(rng, count; threaded = false)
        end
    end
end

# Keep both results live so the optimizer cannot drop either measured allocation.
# A collection inside a measured window skews the counter, so take the minimum.
function serial_split_overhead(rng, count)
    R = typeof(rng)
    reference = Vector{R}(undef, 0)
    children = Vector{R}(undef, 0)
    overhead = typemax(Int)
    for _ = 1:5
        baseline = @allocated reference = Vector{R}(undef, count)
        measured = @allocated children = splitrng(rng, count; threaded = false)
        overhead = min(overhead, measured - baseline)
    end
    return overhead, reference !== children && length(children) == count
end

# Specialize the measurement so Julia 1.10 does not box heterogeneous loop results.
derive_allocations(rng) = (@allocated(splitrng(rng, Val(3))), @allocated(subrng(rng, 42)))

@testset "R30 inference and allocation" begin
    for F in (Philox2x32, Threefry4x64)
        rng = F(123)
        @test @inferred(splitrng(rng, Val(3))) isa NTuple{3,typeof(rng)}
        @test @inferred(subrng(rng, 42)) isa typeof(rng)
        derive_allocations(rng)
        @test derive_allocations(rng) == (0, 0)
    end

    rng = Philox4x32(123)
    for count in (1000, 4096)
        overhead, valid = serial_split_overhead(rng, count)
        # An allocation budget permits lower allocation than the reference.
        @test overhead <= 0 && valid
    end
end
