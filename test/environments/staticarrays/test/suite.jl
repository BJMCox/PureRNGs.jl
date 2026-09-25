using PureRNGs
using Random
using StaticArrays
using Test

# A static array of N elements is the next N scalar draws, so it equals a
# length-N fill.
@testset "a static draw is the next N scalar draws" begin
    for F in (Philox4x32, Threefry2x64, ChaCha),
        (SA, next) in (
            (SVector{3,Float64}, rand_next),
            (SMatrix{2,3,Int32,6}, rand_next),
            (MVector{2,UInt8}, rand_next),
            (SMatrix{2,2,Float32,4}, randn_next),
            (SVector{4,Float64}, randexp_next),
        )

        rng = F(0x5a1, 3)
        T = eltype(SA)
        values, after = next(rng, T, length(SA))
        draw, next_rng = next(rng, SA)
        @test draw isa SA
        @test draw == SA(values)
        @test next_rng == after
    end
    rng = Philox4x32(0x5a2)
    @test rand(rng, SVector{2,Float32}) == first(rand_next(rng, SVector{2,Float32}))
    @test randn(rng, SVector{2,Float64}) == first(randn_next(rng, SVector{2,Float64}))
    @test randexp(rng, SVector{2,Float64}) == first(randexp_next(rng, SVector{2,Float64}))
    static_bytes(rng) = @allocated rand_next(rng, SVector{3,Float64})
    static_bytes(rng)
    @test static_bytes(rng) == 0
end

# Arrays of static arrays and addressed draws are the chained static draws.
@testset "static arrays, fills, and addresses equal the chained draws" begin
    SA = SVector{3,Float64}
    for (held, fill!, next, next!, at) in (
            (rand, rand!, rand_next, rand_next!, rand_at),
            (randn, randn!, randn_next, randn_next!, randn_at),
        ),
        threaded in (false, true)

        rng = Philox4x32(0x5a3, 5)
        chained = SA[]
        current = rng
        for _ = 1:1000
            value, current = next(current, SA)
            push!(chained, value)
        end
        values, next_rng = next(rng, SA, 10, 100; threaded)
        @test values isa Matrix{SA}
        @test vec(values) == chained
        @test next_rng == current
        @test vec(held(rng, SA, (10, 100); threaded)) == chained
        @test fill!(rng, Vector{SA}(undef, 1000); threaded) == chained
        @test last(next!(rng, Vector{SA}(undef, 1000); threaded)) == current
        @test at(rng, SA, 7) == chained[7]
        @test at(rng, SA, 5:900; threaded) == chained[5:900]
    end
end

@testset "a static pick is N chained picks" begin
    rng = Philox4x32(0x5a4)
    picks, after = rand_next(rng, 1:6, 4)
    draw, next_rng = rand_next(rng, 1:6, SVector{4})
    @test draw === SVector{4,Int}(picks)
    @test next_rng == after
    @test rand(rng, [:a, :b], SVector{3}) == SVector{3}(first(rand_next(rng, [:a, :b], 3)))
end
