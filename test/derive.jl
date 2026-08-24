using MLDataDevices

const DERIVE_TAG = UInt32(0xc0ffee00)
const SPLIT_SUBTAG = UInt32(0)
const FOLD_SUBTAG = UInt32(1)
const NARROW_FOLD_INDEX = UInt32(0xffffffff)

zero_position(rng) = PureRNGs._zero_position(typeof(rng))

@testset "R9 derivation constants" begin
    @test PureRNGs._DERIVE_TAG === DERIVE_TAG
    @test PureRNGs._SPLIT_SUBTAG === SPLIT_SUBTAG
    @test PureRNGs._FOLD_SUBTAG === FOLD_SUBTAG
    @test PureRNGs._THREEFRY_FOLD_INDEX === NARROW_FOLD_INDEX
end

@testset "R12 and R17 split counters and packing" begin
    rng = Philox2x32(0x12345678)
    block = PureRNGs._philox2x32((UInt32(0), DERIVE_TAG), rng.key)
    @test splitrng(rng, Val(1))[1].key == (block[1],)

    rng = Philox4x32(0x123456789abcdef0)
    block0 =
        PureRNGs._philox4x32((UInt32(0), UInt32(0), SPLIT_SUBTAG, DERIVE_TAG), rng.key)
    block1 =
        PureRNGs._philox4x32((UInt32(1), UInt32(0), SPLIT_SUBTAG, DERIVE_TAG), rng.key)
    @test getfield.(splitrng(rng, Val(3)), :key) ==
          ((block0[1], block0[2]), (block0[3], block0[4]), (block1[1], block1[2]))

    rng = Philox2x64(0x123456789abcdef0)
    high = (UInt64(DERIVE_TAG) << 32) | UInt64(SPLIT_SUBTAG)
    block0 = PureRNGs._philox2x64((UInt64(0), high), rng.key)
    block1 = PureRNGs._philox2x64((UInt64(1), high), rng.key)
    @test getfield.(splitrng(rng, Val(3)), :key) ==
          ((block0[1],), (block0[2],), (block1[1],))

    rng = Philox4x64(big"0x123456789abcdef00fedcba987654321")
    block = PureRNGs._philox4x64(
        (UInt64(0), UInt64(SPLIT_SUBTAG), UInt64(0), UInt64(DERIVE_TAG)),
        rng.key,
    )
    @test getfield.(splitrng(rng, Val(2)), :key) ==
          ((block[1], block[2]), (block[3], block[4]))

    rng = Threefry2x32(0x123456789abcdef0)
    block = PureRNGs._threefry2x32((UInt32(0), DERIVE_TAG), rng.key)
    @test splitrng(rng, Val(1))[1].key == block

    rng = Threefry4x32(big"0x123456789abcdef00fedcba987654321")
    block = PureRNGs._threefry4x32(
        (UInt32(0), UInt32(0), SPLIT_SUBTAG, DERIVE_TAG),
        rng.key,
    )
    @test splitrng(rng, Val(1))[1].key == block

    rng = Threefry2x64(big"0x123456789abcdef00fedcba987654321")
    block = PureRNGs._threefry2x64((UInt64(0), high), rng.key)
    @test splitrng(rng, Val(1))[1].key == block

    rng = Threefry4x64(
        big"0x123456789abcdef00fedcba987654321112233445566778899aabbccddeeff00",
    )
    block = PureRNGs._threefry4x64(
        (UInt64(0), UInt64(SPLIT_SUBTAG), UInt64(0), UInt64(DERIVE_TAG)),
        rng.key,
    )
    @test splitrng(rng, Val(1))[1].key == block
end

@testset "R12 and R18 subrng counters and packing" begin
    purpose = UInt64(0x123456789abcdef0)

    rng = Philox2x32(0x12345678)
    namespace = PureRNGs._philox2x32((NARROW_FOLD_INDEX, DERIVE_TAG), rng.key)
    block = PureRNGs._philox2x32(
        (purpose % UInt32, (purpose >> 32) % UInt32),
        (namespace[1],),
    )
    @test subrng(rng, purpose).key == (block[1],)

    rng = Threefry2x32(0x123456789abcdef0)
    namespace = PureRNGs._threefry2x32((NARROW_FOLD_INDEX, DERIVE_TAG), rng.key)
    block =
        PureRNGs._threefry2x32((purpose % UInt32, (purpose >> 32) % UInt32), namespace)
    @test subrng(rng, purpose).key == block

    wide_cases = (
        (
            Philox4x32(0x123456789abcdef0),
            (purpose % UInt32, (purpose >> 32) % UInt32, FOLD_SUBTAG, DERIVE_TAG),
            PureRNGs._philox4x32,
            2,
        ),
        (
            Philox2x64(0x123456789abcdef0),
            (purpose, (UInt64(DERIVE_TAG) << 32) | UInt64(FOLD_SUBTAG)),
            PureRNGs._philox2x64,
            1,
        ),
        (
            Philox4x64(big"0x123456789abcdef00fedcba987654321"),
            (purpose, UInt64(FOLD_SUBTAG), UInt64(0), UInt64(DERIVE_TAG)),
            PureRNGs._philox4x64,
            2,
        ),
        (
            Threefry4x32(big"0x123456789abcdef00fedcba987654321"),
            (purpose % UInt32, (purpose >> 32) % UInt32, FOLD_SUBTAG, DERIVE_TAG),
            PureRNGs._threefry4x32,
            4,
        ),
        (
            Threefry2x64(big"0x123456789abcdef00fedcba987654321"),
            (purpose, (UInt64(DERIVE_TAG) << 32) | UInt64(FOLD_SUBTAG)),
            PureRNGs._threefry2x64,
            2,
        ),
        (
            Threefry4x64(
                big"0x123456789abcdef00fedcba987654321112233445566778899aabbccddeeff00",
            ),
            (purpose, UInt64(FOLD_SUBTAG), UInt64(0), UInt64(DERIVE_TAG)),
            PureRNGs._threefry4x64,
            4,
        ),
    )
    for (rng, counter, core, key_words) in wide_cases
        block = core(counter, rng.key)
        @test subrng(rng, purpose).key == ntuple(i -> block[i], key_words)
    end
end

@testset "R20 derivation state law" begin
    for F in FAMILY_TYPES
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
    # Philox4x32 and Threefry2x32 were captured from testbed commit
    # 7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d.
    # The other families are frozen vectors from the revision 9 stream law.
    cases = (
        (
            Philox2x32((0x01234567,)),
            ((0x9830e21c,), (0xb1b55909,), (0xf95ec6ee,), (0x5a2a623f,)),
            (0x4a675769,),
            (0x885a87e9,),
        ),
        (
            Philox4x32((0x01234567, 0x89abcdef)),
            (
                (0x730767c8, 0x34b3bda3),
                (0x36485763, 0x4591f4ae),
                (0x5e5aa077, 0x1242f338),
                (0x1367c2c5, 0xc4916029),
            ),
            (0xaeb2701f, 0xaf683b45),
            (0xcf578767, 0xb4b42840),
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
            Threefry2x32((0x01234567, 0x89abcdef)),
            (
                (0x011ac086, 0x5205f808),
                (0xcb2fd2f2, 0xc0f6498c),
                (0xb8d80ff1, 0xe876ab70),
                (0x62f19539, 0x0e413987),
            ),
            (0x920e1914, 0x6fc51759),
            (0x96551a73, 0x5c4b7b9a),
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
    @test PureRNGs._derive_child(
        Threefry2x32((0x01234567, 0x89abcdef)),
        UInt64(0xfffffffe),
    ).key == (0x4a0bb8c6, 0xc9211a0e)
end

@testset "R21 request forms and bounds" begin
    for F in FAMILY_TYPES
        rng = F(123)
        @test splitrng(rng, Val(0)) === ()
        @test splitrng(rng, 0) == typeof(rng)[]
        @test splitrng(rng, 3) == collect(splitrng(rng, Val(3)))
        @test splitrng(rng, Int8(3)) == collect(splitrng(rng, Val(3)))
        @test_throws ArgumentError splitrng(rng, -1)
        @test_throws ArgumentError splitrng(rng, Val(-1))
        @test_throws ArgumentError splitrng(rng, Val(UInt32(2)))
    end

    for rng in (Philox2x32(123), Threefry2x32(123))
        @test PureRNGs._derive_child(rng, UInt64(0xfffffffe)).position ==
              zero_position(rng)
        @test_throws ArgumentError PureRNGs._derive_child(rng, UInt64(0xffffffff))
        @test_throws ArgumentError splitrng(rng, UInt64(0x1_0000_0000))
    end

    @test !applicable(splitrng, Philox4x32(1), 2.0)
    @test !applicable(subrng, Philox4x32(1), 2.0)
end

@testset "R22 derivation documentation" begin
    docs = Base.Docs.meta(PureRNGs)
    split_doc = string(docs[Base.Docs.Binding(PureRNGs, :splitrng)])
    sub_doc = string(docs[Base.Docs.Binding(PureRNGs, :subrng)])
    philox2x32_doc = string(docs[Base.Docs.Binding(PureRNGs, :Philox2x32)])
    for doc in (split_doc, sub_doc)
        @test occursin("collision", lowercase(doc))
        @test occursin("position", lowercase(doc))
        @test occursin("device", lowercase(doc))
    end
    @test occursin("purpose", lowercase(sub_doc))
    @test occursin("few thousand", lowercase(philox2x32_doc))

    splitting_doc = lowercase(read(joinpath(pkgdir(PureRNGs), "SPLITTING.md"), String))
    @test occursin("n^2 / 2^(k+1)", splitting_doc)
    @test occursin("subrng(root, chunk_id)", splitting_doc)
    @test occursin("subtrees identical", splitting_doc)
    @test occursin("few thousand", splitting_doc)
end

@testset "R30 inference and allocation" begin
    for F in FAMILY_TYPES
        rng = F(123)
        @test @inferred(splitrng(rng, Val(3))) isa NTuple{3,typeof(rng)}
        @test @inferred(subrng(rng, 42)) isa typeof(rng)
        splitrng(rng, Val(3))
        subrng(rng, 42)
        @test @allocated(splitrng(rng, Val(3))) == 0
        @test @allocated(subrng(rng, 42)) == 0
    end
end

@testset "R48 derivation exports" begin
    @test Base.isexported(PureRNGs, :splitrng)
    @test Base.isexported(PureRNGs, :subrng)
    @test !Base.isexported(PureRNGs, :derive_child)
    @test !Base.isexported(PureRNGs, :_derive_child)
end
