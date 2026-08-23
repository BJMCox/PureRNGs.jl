using Random: rand!

const IR = PureRNGs
const KA = PureRNGs.KernelAbstractions
const MLD = PureRNGs.MLDataDevices

const SCALAR_32_FAMILIES = (Philox2x32, Philox4x32, Threefry2x32, Threefry4x32)
const SCALAR_64_FAMILIES = (Philox2x64, Philox4x64, Threefry2x64, Threefry4x64)
const SCALAR_UNIFORM_TYPES = (Bool, UInt32, UInt64, Float32, Float64)

mutable struct BackendProbe{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    lookups::Base.RefValue{Int}
end

Base.size(array::BackendProbe) = size(array.data)
Base.axes(array::BackendProbe) = axes(array.data)
Base.IndexStyle(::Type{<:BackendProbe{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.getindex(array::BackendProbe, indices...) = getindex(array.data, indices...)
Base.setindex!(array::BackendProbe, value, indices...) =
    setindex!(array.data, value, indices...)
MLD.get_device(array::BackendProbe) = MLD.get_device(array.data)
function KA.get_backend(array::BackendProbe)
    array.lookups[] += 1
    return KA.get_backend(array.data)
end

struct WrongDeviceArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end

Base.size(array::WrongDeviceArray) = size(array.data)
Base.axes(array::WrongDeviceArray) = axes(array.data)
Base.IndexStyle(::Type{<:WrongDeviceArray{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.getindex(array::WrongDeviceArray, indices...) = getindex(array.data, indices...)
Base.setindex!(array::WrongDeviceArray, value, indices...) =
    setindex!(array.data, value, indices...)
MLD.get_device(::WrongDeviceArray) = MLD.UnknownDevice()

function scalar_chain(rng, ::Type{T}, count) where {T}
    values = Vector{T}(undef, count)
    cursor = rng
    for index in eachindex(values)
        cursor, values[index] = rand_next(cursor, T)
    end
    return cursor, values
end

sync_cpu() = KA.synchronize(KA.CPU())

@testset "draw block mapping" begin
    @test IR.FAMILY_BITS === UInt32(0)

    family = UInt32(0xa5)
    block = UInt64(0x00123456789abcde)
    block128 = (UInt64(0x0123456789abcdef), UInt64(0xfedcba9876543210))

    p2x32 = Philox2x32((UInt32(0x89abcdef),))
    p4x32 = Philox4x32((UInt32(0x89abcdef), UInt32(0x01234567)))
    p2x64 = Philox2x64((UInt64(0x0123456789abcdef),))
    p4x64 = Philox4x64((UInt64(0x0123456789abcdef), UInt64(0xfedcba9876543210)))
    t2x32 = Threefry2x32((UInt32(0x89abcdef), UInt32(0x01234567)))
    t4x32 = Threefry4x32((
        UInt32(0x89abcdef),
        UInt32(0x01234567),
        UInt32(0xfedcba98),
        UInt32(0x76543210),
    ),)
    t2x64 = Threefry2x64((UInt64(0x0123456789abcdef), UInt64(0xfedcba9876543210)))
    t4x64 = Threefry4x64((
        UInt64(0x0123456789abcdef),
        UInt64(0xfedcba9876543210),
        UInt64(0x13579bdf2468ace0),
        UInt64(0xeca86420fdb97531),
    ),)

    narrow_counter =
        (block % UInt32, (family << 24) | (((block >> 32) & 0x00ffffff) % UInt32))
    wide32_counter = (block % UInt32, (block >> 32) % UInt32, family, UInt32(0))
    wide2x64_counter = (block, UInt64(family))
    wide4x64_counter = (block128..., UInt64(family), UInt64(0))

    @test IR._block(p2x32, family, block) == IR._philox2x32(narrow_counter, p2x32.key)
    @test IR._block(t2x32, family, block) == IR._threefry2x32(narrow_counter, t2x32.key)
    @test IR._block(p4x32, family, block) == IR._philox4x32(wide32_counter, p4x32.key)
    @test IR._block(t4x32, family, block) == IR._threefry4x32(wide32_counter, t4x32.key)
    @test IR._block(p2x64, family, block) == IR._philox2x64(wide2x64_counter, p2x64.key)
    @test IR._block(t2x64, family, block) == IR._threefry2x64(wide2x64_counter, t2x64.key)
    @test IR._block(p4x64, family, block128...) ==
          IR._philox4x64(wide4x64_counter, p4x64.key)
    @test IR._block(t4x64, family, block128...) ==
          IR._threefry4x64(wide4x64_counter, t4x64.key)

    # PureRNGsTestbed.jl 7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d
    @test IR._block(p4x32, family, block) ==
          (UInt32(0xea75de19), UInt32(0xfb84293b), UInt32(0xe16f7d84), UInt32(0xd9b42cde))
    @test IR._block(t2x32, family, block) == (UInt32(0x12830fe3), UInt32(0xd895391c))

    @testset "boundaries" begin
        narrow_max = UInt64(0x00ffffffffffffff)
        narrow_max_counter = (typemax(UInt32), (family << 24) | UInt32(0x00ffffff))
        @test IR._block(p2x32, family, narrow_max) ==
              IR._philox2x32(narrow_max_counter, p2x32.key)
        @test IR._block(t2x32, family, narrow_max) ==
              IR._threefry2x32(narrow_max_counter, t2x32.key)

        block_max = typemax(UInt64)
        wide32_max_counter = (typemax(UInt32), typemax(UInt32), family, UInt32(0))
        @test IR._block(p4x32, family, block_max) ==
              IR._philox4x32(wide32_max_counter, p4x32.key)
        @test IR._block(t4x32, family, block_max) ==
              IR._threefry4x32(wide32_max_counter, t4x32.key)
        @test IR._block(p2x64, family, block_max) ==
              IR._philox2x64((block_max, UInt64(family)), p2x64.key)
        @test IR._block(t2x64, family, block_max) ==
              IR._threefry2x64((block_max, UInt64(family)), t2x64.key)

        block128_max = (typemax(UInt64), typemax(UInt64))
        wide4x64_max_counter = (block128_max..., UInt64(family), UInt64(0))
        @test IR._block(p4x64, family, block128_max...) ==
              IR._philox4x64(wide4x64_max_counter, p4x64.key)
        @test IR._block(t4x64, family, block128_max...) ==
              IR._threefry4x64(wide4x64_max_counter, t4x64.key)
    end

    calls = (
        (p2x32, family, block),
        (p4x32, family, block),
        (p2x64, family, block),
        (p4x64, family, block128...),
        (t2x32, family, block),
        (t4x32, family, block),
        (t2x64, family, block),
        (t4x64, family, block128...),
    )
    for args in calls
        @test @inferred(IR._block(args...)) == IR._block(args...)
        IR._block(args...)
        @test @allocated(IR._block(args...)) == 0
    end
end

@testset "R23 scalar method surface" begin
    rng = Philox4x32(0)
    error = try
        rand(rng)
    catch caught
        caught
    end
    @test error isa ArgumentError
    @test occursin("rand(rng, T)", sprint(showerror, error))
    @test which(rand, (typeof(rng),)).module === IR
    @test which(rand, (typeof(rng), Type{UInt32})).module === IR
    @test which(rand_next, (typeof(rng),)).module === IR
    @test which(rand_next, (typeof(rng), Type{UInt32})).module === IR
    for unsupported in (Union{UInt32,Float32}, Int32)
        @test !applicable(rand, rng, unsupported)
        @test !applicable(rand_next, rng, unsupported)
        @test_throws MethodError rand(rng, unsupported)
        @test_throws MethodError rand_next(rng, unsupported)
    end
end

@testset "R2 and R25 scalar oracle" begin
    # PureRNGsTestbed.jl 7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d
    testbed_vectors = (
        (
            Philox4x32((UInt32(0), UInt32(0))),
            (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8),
            0x3ecc4fd0,
            0x3fd989fa35785a70,
        ),
        (
            Threefry2x32((UInt32(0), UInt32(0))),
            (0x6b200159, 0x99ba4efe),
            0x3ed64002,
            0x3fdac80056666e92,
        ),
    )

    # R13 core-plus-layout vectors for families outside the testbed.
    additional_vectors = (
        (
            Philox2x32((UInt32(0),)),
            (0xff1dae59, 0x6cd10df2),
            0x3f7f1dae,
            0x3fefe3b5cb2d9a21,
        ),
        (
            Threefry4x32(ntuple(_ -> UInt32(0), Val(4))),
            (0x9c6ca96a, 0xe17eae66, 0xfc10ecd4, 0x5256a7d8),
            0x3f1c6ca9,
            0x3fe38d952d5c2fd5,
        ),
    )

    for (rng, words, float32_bits, float64_bits) in
        (testbed_vectors..., additional_vectors...)
        raw64 = (UInt64(words[1]) << 32) | UInt64(words[2])
        @test rand(rng, UInt32) === words[1]
        @test rand(rng, Bool) === isodd(words[1])
        @test rand(rng, UInt64) === raw64
        @test reinterpret(UInt32, rand(rng, Float32)) == float32_bits
        @test reinterpret(UInt64, rand(rng, Float64)) == float64_bits
    end
end

@testset "R25, R27, and R62 native-64 scalar oracle" begin
    # Frozen from the core outputs at family zero, block zero.
    vectors = (
        (
            Philox2x64(0),
            0xca00a045,
            true,
            0xca00a0459843d731,
            0x3f4a00a0,
            0x3fe9401408b3087a,
        ),
        (
            Philox4x64(0),
            0x16554d9e,
            false,
            0x16554d9eca36314c,
            0x3db2aa68,
            0x3fb6554d9eca3630,
        ),
        (
            Threefry2x64(0),
            0xc2b6e3a8,
            false,
            0xc2b6e3a8c2c69865,
            0x3f42b6e3,
            0x3fe856dc751858d3,
        ),
        (
            Threefry4x64(0),
            0x09218ebd,
            true,
            0x09218ebde6c85537,
            0x3d1218e0,
            0x3fa2431d7bcd90a0,
        ),
    )

    for (rng, word32, bit, word64, float32_bits, float64_bits) in vectors
        @test rand(rng, UInt32) === word32
        @test rand(rng, Bool) === bit
        @test rand(rng, UInt64) === word64
        @test reinterpret(UInt32, rand(rng, Float32)) === float32_bits
        @test reinterpret(UInt64, rand(rng, Float64)) === float64_bits
    end
end

@testset "R62 native-64 logical lane order" begin
    vectors = (
        (Philox2x64(0), IR._Position64(7, 1), 0xa5a91b11, true, 0x3f25a91b, 0xad605ce0),
        (Philox4x64(0), IR._Position128(7, 9, 1), 0x9b05a583, true, 0x3f1b05a5, 0x641004f0),
        (Threefry2x64(0), IR._Position64(7, 1), 0xa9653714, false, 0x3f296537, 0xb977e699),
        (
            Threefry4x64(0),
            IR._Position128(7, 9, 1),
            0x3dee0db8,
            false,
            0x3e77b834,
            0xa1e4dde9,
        ),
    )

    for (base, position, low_half, bit, float32_bits, next_high_half) in vectors
        rng = IR._rebuild(base, position, base.device)
        next, first = rand_next(rng, UInt32)
        after, second = rand_next(next, UInt32)
        @test first === low_half
        @test rand(rng, Bool) === bit
        @test reinterpret(UInt32, rand(rng, Float32)) === float32_bits
        @test second === next_high_half
        @test rand(rng, UInt32) === low_half
        @test rng.position === position
        @test after.position.lane == UInt8(3)
    end
end

@testset "R53 native-64 mixed-width continuation" begin
    for F in SCALAR_64_FAMILIES
        base = F(73)
        position =
            base.position isa IR._Position64 ? IR._Position64(5, 1) :
            IR._Position128(5, 7, 1)
        rng = IR._rebuild(base, position, base.device)
        next32, word32 = rand_next(rng, UInt32)
        next64, word64 = rand_next(next32, UInt64)
        aligned, expected_next = IR._reserve_aligned(next32, UInt64(2), UInt64(2))
        @test word32 === rand(rng, UInt32)
        @test word64 === rand(aligned, UInt64)
        @test next64 === expected_next
    end
end

@testset "R53 and R54 native-64 terminal scalar draws" begin
    for F in SCALAR_64_FAMILIES
        base = F(7)
        width = IR._words_per_block(base)
        last_position =
            base.position isa IR._Position64 ? IR._Position64(typemax(UInt64), width - 1) :
            IR._Position128(typemax(UInt64), typemax(UInt64), width - 1)
        last = IR._rebuild(base, last_position, base.device)

        exhausted, value = rand_next(last, UInt32)
        @test value === rand(last, UInt32)
        @test IR._is_exhausted(exhausted.position)
        @test_throws ArgumentError rand(exhausted, UInt32)
        @test_throws ArgumentError rand_next(exhausted, UInt32)
        @test_throws ArgumentError rand(last, UInt64)
        @test_throws ArgumentError rand_next(last, UInt64)

        pair_position =
            base.position isa IR._Position64 ? IR._Position64(typemax(UInt64), width - 2) :
            IR._Position128(typemax(UInt64), typemax(UInt64), width - 2)
        final_pair = IR._rebuild(base, pair_position, base.device)
        pair_end, pair = rand_next(final_pair, UInt64)
        @test pair === rand(final_pair, UInt64)
        @test IR._is_exhausted(pair_end.position)
    end
end

@testset "R29 native-64 addressed scalar draws" begin
    for F in SCALAR_64_FAMILIES, T in SCALAR_UNIFORM_TYPES
        rng = F(123)
        cursor = rng
        for i = 1:9
            cursor, expected = rand_next(cursor, T)
            @test randat(rng, T, i) === expected
            @test rng.position == IR._zero_position(F)
        end
    end

    for F in SCALAR_64_FAMILIES, T in SCALAR_UNIFORM_TYPES
        base = F(47)
        position =
            base.position isa IR._Position64 ? IR._Position64(5, 1) :
            IR._Position128(5, 7, 1)
        rng = IR._rebuild(base, position, base.device)
        cursor = rng
        for i = 1:7
            cursor, expected = rand_next(cursor, T)
            @test randat(rng, T, i) === expected
        end
    end
end

@testset "R29 native-64 addressed bounds" begin
    for F in SCALAR_64_FAMILIES, T in SCALAR_UNIFORM_TYPES
        rng = F(7)
        @test_throws ArgumentError randat(rng, T, 0)
        @test_throws ArgumentError randat(rng, T, -1)
    end

    for F in SCALAR_64_FAMILIES
        base = F(13)
        width = IR._words_per_block(base)
        last_position =
            base.position isa IR._Position64 ? IR._Position64(typemax(UInt64), width - 1) :
            IR._Position128(typemax(UInt64), typemax(UInt64), width - 1)
        last = IR._rebuild(base, last_position, base.device)
        @test randat(last, UInt32, 1) === rand(last, UInt32)
        @test_throws ArgumentError randat(last, UInt32, 2)
        @test_throws ArgumentError randat(last, UInt64, 1)

        pair_position =
            base.position isa IR._Position64 ? IR._Position64(typemax(UInt64), width - 2) :
            IR._Position128(typemax(UInt64), typemax(UInt64), width - 2)
        final_pair = IR._rebuild(base, pair_position, base.device)
        @test randat(final_pair, UInt64, 1) === rand(final_pair, UInt64)
        @test_throws ArgumentError randat(final_pair, UInt64, 2)
    end

    for F in (Philox2x64, Threefry2x64)
        rng = F(19)
        final_index = big(1) << 66
        final_word = IR._block(rng, IR.FAMILY_BITS, typemax(UInt64))[2]
        @test randat(rng, UInt32, final_index) === final_word % UInt32
        @test randat(rng, UInt64, big(1) << 65) === final_word
        @test_throws ArgumentError randat(rng, UInt32, final_index + 1)
        @test_throws ArgumentError randat(rng, UInt64, (big(1) << 65) + 1)
    end

    for F in (Philox4x64, Threefry4x64)
        base = F(17)
        near_end = IR._rebuild(
            base,
            IR._Position128(typemax(UInt64) - 1, typemax(UInt64), 0),
            base.device,
        )
        @test randat(near_end, UInt32, UInt64(16)) isa UInt32
        @test randat(near_end, UInt64, UInt64(8)) isa UInt64
        @test_throws ArgumentError randat(near_end, UInt32, UInt64(17))
        @test_throws ArgumentError randat(near_end, UInt64, UInt64(9))
    end

    for F in (Philox4x64, Threefry4x64)
        rng = F(19)
        final_block = IR._block(rng, IR.FAMILY_BITS, typemax(UInt64), typemax(UInt64))
        @test randat(rng, UInt32, big(1) << 131) === final_block[4] % UInt32
        @test randat(rng, UInt64, big(1) << 130) === final_block[4]
        @test_throws ArgumentError randat(rng, UInt32, (big(1) << 131) + 1)
        @test_throws ArgumentError randat(rng, UInt64, (big(1) << 130) + 1)
    end
end

@testset "R23, R29, and R30 native-64 method surface and performance" begin
    for F in SCALAR_64_FAMILIES
        rng = F(29)
        default_next, default_value = rand_next(rng)
        typed_next, typed_value = rand_next(rng, Float64)
        @test default_next === typed_next
        @test default_value === typed_value
        @test which(rand_next, (typeof(rng),)).module === IR

        for T in SCALAR_UNIFORM_TYPES
            @test which(rand, (typeof(rng), Type{T})).module === IR
            @test which(rand_next, (typeof(rng), Type{T})).module === IR
            int_method = which(randat, (typeof(rng), Type{T}, Int))
            @test int_method.module === IR
            @test which(randat, (typeof(rng), Type{T}, UInt64)) === int_method
            @test which(randat, (typeof(rng), Type{T}, BigInt)) === int_method

            @test @inferred(rand(rng, T)) isa T
            @test @inferred(rand_next(rng, T)) isa Tuple{typeof(rng),T}
            @test @inferred(randat(rng, T, 3)) isa T
            @test @inferred(randat(rng, T, UInt64(3))) isa T
            rand(rng, T)
            rand_next(rng, T)
            randat(rng, T, 3)
            randat(rng, T, UInt64(3))
            @test @allocated(rand(rng, T)) == 0
            @test @allocated(rand_next(rng, T)) == 0
            @test @allocated(randat(rng, T, 3)) == 0
            @test @allocated(randat(rng, T, UInt64(3))) == 0
        end

        for unsupported in (Union{UInt32,Float32}, Int32, Int64, Float16)
            @test !applicable(rand, rng, unsupported)
            @test !applicable(rand_next, rng, unsupported)
            @test !applicable(randat, rng, unsupported, 1)
            @test_throws MethodError rand(rng, unsupported)
            @test_throws MethodError rand_next(rng, unsupported)
            @test_throws MethodError randat(rng, unsupported, 1)
        end
        @test !applicable(randat, rng, UInt32, 1.0)
    end
end

@testset "R5 pure scalar draws" begin
    for F in SCALAR_32_FAMILIES, T in SCALAR_UNIFORM_TYPES
        rng = IR._rebuild(F(123), IR._Position64(7, 1), F(123).device)
        position = rng.position
        first = rand(rng, T)
        @test rand(rng, T) === first
        @test rng.position === position
    end
end

@testset "R24 and R53 scalar continuation" begin
    for F in SCALAR_32_FAMILIES
        rng = F(0)
        next32, value32 = rand_next(rng, UInt32)
        @test value32 === rand(rng, UInt32)
        @test next32.position == IR._Position64(0, 1)

        next64, value64 = rand_next(rng, UInt64)
        @test value64 === rand(rng, UInt64)
        @test next64.position ==
              (IR._words_per_block(rng) == 2 ? IR._Position64(1, 0) : IR._Position64(0, 2))

        default_next, default_value = rand_next(rng)
        typed_next, typed_value = rand_next(rng, Float64)
        @test default_next === typed_next
        @test default_value === typed_value
        @test default_value isa Float64
    end

    p = Philox4x32(0)
    p1, bit = rand_next(p, Bool)
    p2, value = rand_next(p1, UInt64)
    pblock = IR._block(p, IR.FAMILY_BITS, UInt64(0))
    @test bit === isodd(pblock[1])
    @test value === (UInt64(pblock[3]) << 32) | UInt64(pblock[4])
    @test p2.position == IR._Position64(1, 0)

    t = Threefry2x32(0)
    t1, tbit = rand_next(t, Bool)
    t2, tvalue = rand_next(t1, UInt64)
    tblock1 = IR._block(t, IR.FAMILY_BITS, UInt64(1))
    @test tbit === isodd(IR._block(t, IR.FAMILY_BITS, UInt64(0))[1])
    @test tvalue === (UInt64(tblock1[1]) << 32) | UInt64(tblock1[2])
    @test t2.position == IR._Position64(2, 0)
end

@testset "R27 and R53 aligned scalar words" begin
    for F in (Philox4x32, Threefry4x32)
        base = F(42)
        rng = IR._rebuild(base, IR._Position64(3, 3), base.device)
        block4 = IR._block(rng, IR.FAMILY_BITS, UInt64(4))
        next, value = rand_next(rng, UInt64)
        @test value === (UInt64(block4[1]) << 32) | UInt64(block4[2])
        @test rand(rng, UInt64) === value
        @test rng.position == IR._Position64(3, 3)
        @test next.position == IR._Position64(4, 2)
    end

    for F in (Philox2x32, Threefry2x32)
        base = F(42)
        rng = IR._rebuild(base, IR._Position64(3, 1), base.device)
        block4 = IR._block(rng, IR.FAMILY_BITS, UInt64(4))
        next, value = rand_next(rng, UInt64)
        @test value === (UInt64(block4[1]) << 32) | UInt64(block4[2])
        @test rand(rng, UInt64) === value
        @test rng.position == IR._Position64(3, 1)
        @test next.position == IR._Position64(5, 0)
    end
end

@testset "R53 and R54 terminal scalar draws" begin
    for F in SCALAR_32_FAMILIES
        base = F(7)
        width = IR._words_per_block(base)
        maximum = IR._max_block(base)
        last = IR._rebuild(base, IR._Position64(maximum, width - 1), base.device)
        last_word = IR._block(last, IR.FAMILY_BITS, maximum)[Int(width)]

        @test rand(last, UInt32) === last_word
        exhausted, value = rand_next(last, UInt32)
        @test value === last_word
        @test IR._is_exhausted(exhausted.position)
        @test_throws ArgumentError rand(exhausted, UInt32)
        @test_throws ArgumentError rand_next(exhausted, UInt32)
        @test_throws ArgumentError rand(last, UInt64)
        @test_throws ArgumentError rand_next(last, UInt64)

        if width == 4
            before_pair = IR._rebuild(base, IR._Position64(maximum, width - 3), base.device)
            pair_end, pair = rand_next(before_pair, UInt64)
            words = IR._block(before_pair, IR.FAMILY_BITS, maximum)
            @test pair === (UInt64(words[3]) << 32) | UInt64(words[4])
            @test IR._is_exhausted(pair_end.position)
        end

        final_pair = IR._rebuild(base, IR._Position64(maximum, width - 2), base.device)
        pair_end, pair = rand_next(final_pair, UInt64)
        words = IR._block(final_pair, IR.FAMILY_BITS, maximum)
        @test pair === (UInt64(words[Int(width)-1]) << 32) | UInt64(words[Int(width)])
        @test IR._is_exhausted(pair_end.position)
    end
end

@testset "R4 and R30 scalar inference and allocation" begin
    for F in SCALAR_32_FAMILIES, T in SCALAR_UNIFORM_TYPES
        rng = F(123)
        @test @inferred(rand(rng, T)) isa T
        @test @inferred(rand_next(rng, T)) isa Tuple{typeof(rng),T}
        rand(rng, T)
        rand_next(rng, T)
        @test @allocated(rand(rng, T)) == 0
        @test @allocated(rand_next(rng, T)) == 0
    end
end

@testset "R29 addressed scalar draws" begin
    for F in SCALAR_32_FAMILIES, T in SCALAR_UNIFORM_TYPES
        rng = F(123)
        cursor = rng
        for i = 1:9
            cursor, expected = rand_next(cursor, T)
            @test randat(rng, T, i) === expected
            @test rng.position == IR._Position64(0, 0)
        end
    end

    for F in SCALAR_32_FAMILIES, T in SCALAR_UNIFORM_TYPES
        base = F(47)
        rng = IR._rebuild(base, IR._Position64(5, 1), base.device)
        cursor = rng
        for i = 1:7
            cursor, expected = rand_next(cursor, T)
            @test randat(rng, T, i) === expected
        end
    end

    for F in SCALAR_32_FAMILIES
        base = F(53)
        rng = IR._rebuild(base, IR._Position64(5, 1), base.device)
        aligned, _ = IR._reserve_aligned(rng, UInt64(2), UInt64(2))
        @test randat(rng, UInt64, 1) === rand(aligned, UInt64)
        @test randat(rng, Float64, 1) === rand(aligned, Float64)
    end

    for F in SCALAR_32_FAMILIES
        base = F(91)
        width = IR._words_per_block(base)
        rng = IR._rebuild(base, IR._Position64(3, width - 1), base.device)
        for T in SCALAR_UNIFORM_TYPES
            @test randat(rng, T, 1) === rand(rng, T)
        end
    end
end

@testset "R29 addressed bounds" begin
    for F in SCALAR_32_FAMILIES, T in SCALAR_UNIFORM_TYPES
        rng = F(7)
        @test_throws ArgumentError randat(rng, T, 0)
        @test_throws ArgumentError randat(rng, T, -1)
    end

    for F in SCALAR_32_FAMILIES
        base = F(13)
        width = IR._words_per_block(base)
        maximum = IR._max_block(base)
        last = IR._rebuild(base, IR._Position64(maximum, width - 1), base.device)
        @test randat(last, UInt32, 1) === rand(last, UInt32)
        @test_throws ArgumentError randat(last, UInt32, 2)
        @test_throws ArgumentError randat(last, UInt64, 1)

        if width == 4
            before_pair = IR._rebuild(base, IR._Position64(maximum, width - 3), base.device)
            @test randat(before_pair, UInt64, 1) === rand(before_pair, UInt64)
            @test_throws ArgumentError randat(before_pair, UInt64, 2)
        end

        exhausted = IR._reserve(last, UInt64(1))
        @test_throws ArgumentError randat(exhausted, UInt32, 1)
    end

    for F in (Philox2x32, Threefry2x32)
        rng = F(19)
        final_index = UInt64(1) << 57
        final_block = IR._block(rng, IR.FAMILY_BITS, IR._max_block(rng))
        @test randat(rng, UInt32, final_index) === final_block[2]
        @test_throws ArgumentError randat(rng, UInt32, final_index + 1)
        @test_throws ArgumentError randat(rng, UInt32, big(final_index) + 1)
    end
end

@testset "R29 large wide addresses" begin
    index = big(typemax(UInt64)) + 2
    offset = index - 1
    for F in (Philox4x32, Threefry4x32)
        rng = F(23)
        width = Int(IR._words_per_block(rng))
        block, lane = divrem(offset, width)
        expected = IR._block(rng, IR.FAMILY_BITS, UInt64(block))[Int(lane)+1]
        @test randat(rng, UInt32, index) === expected

        pair_block, pair_slot = divrem(offset, width ÷ 2)
        pair_words = IR._block(rng, IR.FAMILY_BITS, UInt64(pair_block))
        first_lane = 2 * Int(pair_slot) + 1
        expected64 =
            (UInt64(pair_words[first_lane]) << 32) | UInt64(pair_words[first_lane+1])
        @test randat(rng, UInt64, index) === expected64
    end
end

@testset "R29 addressed method surface and performance" begin
    rng = Philox4x32(29)
    for T in SCALAR_UNIFORM_TYPES
        int_method = which(randat, (typeof(rng), Type{T}, Int))
        @test int_method.module === IR
        @test which(randat, (typeof(rng), Type{T}, UInt64)) === int_method
        @test which(randat, (typeof(rng), Type{T}, BigInt)) === int_method
        @test @inferred(randat(rng, T, 3)) isa T
        @test @inferred(randat(rng, T, UInt64(3))) isa T
        randat(rng, T, 3)
        randat(rng, T, UInt64(3))
        @test @allocated(randat(rng, T, 3)) == 0
        @test @allocated(randat(rng, T, UInt64(3))) == 0
    end

    for unsupported in (Union{UInt32,Float32}, Int32, Int64, Float16)
        @test !applicable(randat, rng, unsupported, 1)
        @test_throws MethodError randat(rng, unsupported, 1)
    end
    @test !applicable(randat, rng, UInt32, 1.0)
end

@testset "R24, R26, and R27 CPU fills" begin
    for F in SCALAR_32_FAMILIES, T in SCALAR_UNIFORM_TYPES
        width = Int(IR._words_per_block(F(0)))
        for lane = 0:(width-1), count in (0, 1, width - lane, width - lane + 1, 9)
            base = F(123)
            rng = IR._rebuild(base, IR._Position64(7, lane), base.device)
            expected_rng, expected = scalar_chain(rng, T, count)

            destination = Vector{T}(undef, count)
            @test rand!(rng, destination) === destination
            sync_cpu()
            @test destination == expected
            @test rng.position == IR._Position64(7, lane)

            replay = Vector{T}(undef, count)
            next_rng, returned = rand_next!(rng, replay)
            @test returned === replay
            sync_cpu()
            @test replay == expected
            @test next_rng === expected_rng
        end
    end
end

@testset "R26 and R53 aligned CPU fills" begin
    for F in (Philox4x32, Threefry4x32)
        base = F(31)
        rng = IR._rebuild(base, IR._Position64(8, 1), base.device)
        destination = Vector{UInt64}(undef, 1)
        next_rng, _ = rand_next!(rng, destination)
        sync_cpu()
        words = IR._block(rng, IR.FAMILY_BITS, UInt64(8))
        @test destination[1] == (UInt64(words[3]) << 32) | UInt64(words[4])
        @test next_rng.position == IR._Position64(9, 0)
    end

    for F in (Philox2x32, Threefry2x32)
        base = F(31)
        rng = IR._rebuild(base, IR._Position64(8, 1), base.device)
        destination = Vector{UInt64}(undef, 1)
        next_rng, _ = rand_next!(rng, destination)
        sync_cpu()
        words = IR._block(rng, IR.FAMILY_BITS, UInt64(9))
        @test destination[1] == (UInt64(words[1]) << 32) | UInt64(words[2])
        @test next_rng.position == IR._Position64(10, 0)
    end
end

@testset "R25 Bool CPU fills" begin
    for F in SCALAR_32_FAMILIES, count in (0, 1, 2, 3, 7, 9, 65, 67)
        base = F(91)
        rng = IR._rebuild(base, IR._Position64(4, 1), base.device)
        next_rng, expected = scalar_chain(rng, Bool, count)

        plain = Array{Bool}(undef, count)
        packed = BitArray(undef, count)
        @test rand!(rng, plain) === plain
        @test rand!(rng, packed) === packed
        sync_cpu()
        @test plain == expected
        @test packed == expected

        packed_next, returned = rand_next!(rng, packed)
        sync_cpu()
        @test returned === packed
        @test packed == expected
        @test packed_next === next_rng
    end
end

@testset "R26 CPU fill shapes and views" begin
    for F in SCALAR_32_FAMILIES, T in SCALAR_UNIFORM_TYPES
        rng = F(22)
        expected_rng, expected = scalar_chain(rng, T, 12)

        matrix = Matrix{T}(undef, 3, 4)
        next_rng, returned = rand_next!(rng, matrix)
        sync_cpu()
        @test returned === matrix
        @test vec(matrix) == expected
        @test next_rng === expected_rng

        storage = fill(zero(T), 24)
        destination = @view storage[2:2:24]
        @test rand!(rng, destination) === destination
        sync_cpu()
        @test collect(destination) == expected
        @test all(iszero, @view storage[1:2:23])
    end
end

@testset "R26 parallel dense CPU fills" begin
    for T in SCALAR_UNIFORM_TYPES
        chunk = IR._dense_fill_chunk_elements(T)
        @test chunk * Int(IR._draw_words(T)) == Int(IR._CPU_FILL_CHUNK_WORDS)
        @test IR._dense_fill_workitems(chunk, T) == 1
        @test IR._dense_fill_workitems(2 * chunk + 1, T) == 3

        parallel_start = (IR._CPU_FILL_MIN_WORKITEMS - 1) * chunk + 1
        @test !IR._use_parallel_dense_fill(IR._dense_fill_workitems(parallel_start - 1, T))
        @test IR._use_parallel_dense_fill(IR._dense_fill_workitems(parallel_start, T))

        maximum_count = typemax(Int)
        final_workitem = IR._dense_fill_workitems(maximum_count, T)
        first, last = IR._dense_fill_bounds(final_workitem, maximum_count, chunk)
        @test first <= last
        @test last == maximum_count
        @test last - first + 1 <= chunk

        previous_first, previous_last =
            IR._dense_fill_bounds(final_workitem - 1, maximum_count, chunk)
        @test previous_first <= previous_last
        @test previous_last + 1 == first

        counts = (
            chunk - 1,
            chunk,
            chunk + 1,
            parallel_start - 1,
            parallel_start,
            parallel_start + chunk,
        )
        for F in SCALAR_32_FAMILIES, count in counts
            base = F(0x531)
            rng = IR._rebuild(base, IR._Position64(11, 1), base.device)
            expected_rng, expected = scalar_chain(rng, T, count)

            dense = Vector{T}(undef, count)
            next_rng, returned = rand_next!(rng, dense)
            sync_cpu()
            @test returned === dense
            @test dense == expected
            @test next_rng === expected_rng

            fallback_data = Vector{T}(undef, count)
            fallback = BackendProbe(fallback_data, Ref(0))
            fallback_rng, returned_fallback = rand_next!(rng, fallback)
            sync_cpu()
            @test returned_fallback === fallback
            @test fallback_data == dense
            @test fallback_rng === next_rng
        end
    end
end

@testset "R39 CPU fill placement" begin
    rng = Philox4x32(7)
    for T in SCALAR_UNIFORM_TYPES
        wrong = WrongDeviceArray(Vector{T}(undef, 1))
        @test_throws ArgumentError rand!(rng, wrong)
        @test_throws ArgumentError rand_next!(rng, wrong)

        empty_wrong = WrongDeviceArray(Vector{T}(undef, 0))
        @test_throws ArgumentError rand!(rng, empty_wrong)
        @test_throws ArgumentError rand_next!(rng, empty_wrong)
    end

    adapted_rng = MLD.CPUDevice{Nothing}()(rng)
    @test adapted_rng.device != MLD.get_device(UInt32[])
    empty = UInt32[]
    @test rand!(adapted_rng, empty) === empty
    destination = Vector{UInt32}(undef, 3)
    @test rand!(adapted_rng, destination) === destination
    sync_cpu()
    @test destination == scalar_chain(adapted_rng, UInt32, 3)[2]
end

@testset "R40 KernelAbstractions launch path" begin
    rng = Philox4x32(13)
    lookups = Ref(0)
    destination = BackendProbe(Vector{UInt32}(undef, 5), lookups)
    expected_rng, expected = scalar_chain(rng, UInt32, length(destination))

    next_rng, returned = rand_next!(rng, destination)
    sync_cpu()
    @test returned === destination
    @test destination.data == expected
    @test next_rng === expected_rng
    @test lookups[] == 1

    width = IR._words_per_block(rng)
    maximum = IR._max_block(rng)
    last = IR._rebuild(rng, IR._Position64(maximum, width - 1), rng.device)
    exhausted = IR._reserve(last, UInt64(1))
    empty_lookups = Ref(0)
    empty = BackendProbe(UInt32[], empty_lookups)
    @test rand!(exhausted, empty) === empty
    next_empty, returned_empty = rand_next!(exhausted, empty)
    @test next_empty === exhausted
    @test returned_empty === empty
    @test empty_lookups[] == 0

    empty_bits = BitArray(undef, 0)
    @test rand!(exhausted, empty_bits) === empty_bits
    bit_next, returned_bits = rand_next!(exhausted, empty_bits)
    @test bit_next === exhausted
    @test returned_bits === empty_bits
end

@testset "R53 CPU fill preflight" begin
    maximum_count = typemax(Int)
    @test IR._fill_word_count(maximum_count, UInt64(2)) ÷ UInt64(2) == UInt64(maximum_count)

    for F in SCALAR_32_FAMILIES, T in SCALAR_UNIFORM_TYPES
        rng = F(17)
        width = IR._words_per_block(rng)
        maximum = IR._max_block(rng)
        span = IR._draw_words(T)
        position = IR._Position64(maximum, width - span)
        last = IR._rebuild(rng, position, rng.device)
        destination = fill(convert(T, T === Bool ? true : 1), 2)
        before = copy(destination)
        @test_throws ArgumentError rand!(last, destination)
        @test destination == before
        @test_throws ArgumentError rand_next!(last, destination)
        @test destination == before
    end
end

@testset "R23 CPU fill method surface" begin
    rng = Philox4x32(29)
    for T in SCALAR_UNIFORM_TYPES
        destination = Vector{T}(undef, 1)
        @test which(rand!, (typeof(rng), typeof(destination))).module === IR
        @test which(rand_next!, (typeof(rng), typeof(destination))).module === IR
        @test @inferred(rand!(rng, destination)) === destination
        @test @inferred(rand_next!(rng, destination)) isa
              Tuple{typeof(rng),typeof(destination)}
        sync_cpu()
    end

    for T in (Int32, Int64, Float16)
        destination = Vector{T}(undef, 1)
        @test !applicable(rand!, rng, destination)
        @test !applicable(rand_next!, rng, destination)
        @test_throws MethodError rand!(rng, destination)
        @test_throws MethodError rand_next!(rng, destination)
    end
end
