# A pick from a collection is one unweighted sample, so every form is fixed by
# `randsample_next` at the same position.

const PICK_POPULATIONS = Any[
    [10, 20, 30, 40],
    reshape(1:6, 2, 3) .* 1.5,
    view([1, 2, 3, 4, 5], 2:4),
    1.0:0.5:3.0,
    big(1):(big(10)^18),
    (1, :a, "x"),
    "héllo",
    Dict(1 => 2, 3 => 4, 5 => 6),
    Set([7, 8, 9]),
    BitSet([1, 5, 9]),
]

@testset "a pick is one unweighted sample of the collection" begin
    for F in (Philox2x32, Philox4x32, Threefry4x64, ChaCha),
        bit in (0, 3),
        pop in PICK_POPULATIONS

        rng = F(0x7d1, bit)
        value, next_rng = rand_next(rng, pop)
        sample, sample_next = randsample_next(rng, pop, 1)
        @test value == only(sample)
        @test next_rng == sample_next
        @test rand(rng, pop) == value
    end
end

@testset "addressed picks equal the chained picks" begin
    for F in (Philox4x32, Threefry2x64), pop in PICK_POPULATIONS
        rng = F(0x7d2, 5)
        current = rng
        for i = 1:4
            value, current = rand_next(current, pop)
            @test rand_at(rng, pop, i) == value
        end
    end
end

@testset "pick arrays and fills equal the unweighted sample" begin
    for F in (Philox2x32, ChaCha), pop in PICK_POPULATIONS, threaded in (false, true)
        rng = F(0x7d3, 1)
        values, next_rng = rand_next(rng, pop, 2, 3; threaded)
        sample, sample_next = randsample_next(rng, pop, 6)
        @test size(values) == (2, 3)
        @test vec(values) == sample
        @test next_rng == sample_next
        @test rand(rng, pop, (2, 3); threaded) == values
        destination = similar(sample)
        @test rand!(rng, destination, pop; threaded) === destination
        @test destination == sample
        @test last(rand_next!(rng, similar(sample), pop; threaded)) == sample_next
    end
end

@testset "a tuple of Int is a shape, and an empty collection is an error" begin
    rng = Philox4x32(0x7d4)
    @test size(first(rand_next(rng, (2, 3)))) == (2, 3)
    @test_throws ArgumentError rand(rng, (2, 3))
    @test rand(rng, (2, 3.0)) in (2, 3.0)
    for empty in (Int[], (), "", Dict{Int,Int}(), Set{Int}())
        empty === () && continue
        @test_throws ArgumentError rand_next(rng, empty)
        @test_throws ArgumentError rand_at(rng, empty, 1)
    end
end

@testset "Char draws are the Unicode scalar at a range offset" begin
    for F in (Philox2x32, Philox4x32, Threefry4x64, ChaCha), bit in (0, 7)
        rng = F(0x7d5, bit)
        offset, offset_next = rand_next(rng, 0x00000000:0x0010f7ff)
        value, next_rng = rand_next(rng, Char)
        @test value === (offset < 0xd800 ? Char(offset) : Char(offset + 0x800))
        @test next_rng == offset_next
        @test rand(rng, Char) === value
    end
    @test PureRNGs._unicode_scalar(UInt64(0xd7ff)) === '퟿'
    @test PureRNGs._unicode_scalar(UInt64(0xd800)) === ''
    @test PureRNGs._unicode_scalar(UInt64(0x10f7ff)) === '\U10ffff'
    @test all(isvalid, rand(Philox4x32(0x7d6), Char, 100_000))
end

@testset "Char fills and addressed draws equal the chained draws" begin
    for F in (Philox4x32, Threefry4x64), threaded in (false, true)
        rng = F(0x7d7, 3)
        chained = Char[]
        current = rng
        for _ = 1:37
            value, current = rand_next(current, Char)
            push!(chained, value)
        end
        values, next_rng = rand_next(rng, Char, 37; threaded)
        @test values == chained
        @test next_rng == current
        @test rand!(rng, Vector{Char}(undef, 37); threaded) == chained
        @test rand_at(rng, Char, 5:9) == chained[5:9]
        @test rand_at(rng, Char, 6) === chained[6]
    end
end

@testset "the bridge picks from every collection as the pure API does" begin
    rng = Philox4x32(0x7d8)
    for pop in vcat(PICK_POPULATIONS, Any[(5,), (5, 6), "abc"])
        bridge = StatefulRNG(rng)
        value, next_rng = PureRNGs._rand_next_pick(rng, pop)
        @test rand(bridge, pop) == value
        @test parent(bridge) == next_rng
        values, after = rand_next(next_rng, pop, 3)
        @test rand(bridge, pop, 3) == values
        @test parent(bridge) == after
    end
    bridge = StatefulRNG(rng)
    @test rand(bridge, Char, 5) == first(rand_next(rng, Char, 5))
end
