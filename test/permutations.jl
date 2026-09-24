# A permutation orders the uniform `UInt64` draws at the held position, ties by
# index; cycles, shuffles, and samples without replacement follow from it.

function _is_one_cycle(cycle)
    length(cycle) <= 1 && return cycle == 1:length(cycle)
    index, steps = 1, 0
    while true
        index = cycle[index]
        steps += 1
        index == 1 && return steps == length(cycle)
        steps > length(cycle) && return false
    end
end

_chi_square(counts, expected) = sum((count - expected)^2 / expected for count in counts)

@testset "a permutation orders the uniform keys at the held position" begin
    for F in (Philox2x32, Philox4x32, Threefry4x64, ChaCha), bit in (0, 3)
        for n in (0, 1, 2, 17, 1000, 4097)
            rng = F(0x7e1, bit)
            keys, keys_next = rand_next(rng, UInt64, n)
            permutation, next_rng = randperm_next(rng, n)
            @test permutation == sortperm(keys)
            @test next_rng == keys_next
            @test randperm(rng, n) == permutation
            @test first(randperm_next(rng, n; threaded = true)) == permutation
            destination = Matrix{Int}(undef, 1, n)
            @test first(randperm_next!(rng, destination)) === destination
            @test vec(destination) == permutation
        end
    end
    @test eltype(randperm(Philox4x32(1), Int32(5))) === Int32
    @test_throws ArgumentError randperm_next(Philox4x32(1), -1)
end

@testset "equal keys are shuffled within their run" begin
    keys = UInt64[5, 3, 5, 1, 3, 5, 9, 1]
    permutation = sortperm(keys)
    IR._resolve_key_ties!(permutation, keys, Philox4x32(0x7e2))
    @test isperm(permutation)
    @test issorted(keys[permutation])
    # Three equal keys: all six orders, each about equally often.
    counts = Dict{Vector{Int},Int}()
    for seed = 1:12_000
        order = [1, 2, 3]
        IR._resolve_key_ties!(order, UInt64[7, 7, 7], Philox4x32(seed))
        counts[order] = get(counts, order, 0) + 1
    end
    @test length(counts) == 6
    @test _chi_square(values(counts), 2000) < 20.5
end

@testset "the key order is the stable sort order" begin
    for n in (0, 1, 2, 3, 100, 70_000)
        keys = first(rand_next(Philox4x32(n), UInt64, n))
        @test IR._order_keys!(Vector{Int}(undef, n), keys) == sortperm(keys)
    end
    keys = UInt64[3, 1, 3, UInt64(1)<<63, 3, 0, typemax(UInt64), 1]
    @test IR._order_keys!(zeros(Int, length(keys)), keys) == sortperm(keys)
end

@testset "permutations of four are uniform" begin
    counts = Dict{Vector{Int},Int}()
    for seed = 1:24_000
        permutation = randperm(Philox4x32(seed), 4)
        counts[permutation] = get(counts, permutation, 0) + 1
    end
    @test length(counts) == 24
    @test _chi_square(values(counts), 1000) < 49.7
end

@testset "a cycle steps through the permutation" begin
    for F in (Philox4x32, Threefry2x64), n in (0, 1, 2, 5, 1000)
        rng = F(0x7e3, 1)
        permutation, permutation_next = randperm_next(rng, n)
        cycle, next_rng = randcycle_next(rng, n)
        @test next_rng == permutation_next
        @test isperm(cycle)
        @test _is_one_cycle(cycle)
        @test all(cycle[permutation[i]] == permutation[mod1(i + 1, n)] for i = 1:n)
        @test randcycle(rng, n) == cycle
        @test first(randcycle_next!(rng, Vector{Int}(undef, n))) == cycle
    end
end

@testset "a shuffle moves elements in linear order by the permutation" begin
    rng = Philox4x32(0x7e4, 5)
    for values in
        ([10, 20, 30, 40, 50], reshape(1.0:12.0, 3, 4), BitVector([1, 0, 0, 1, 1]))
        permutation, permutation_next = randperm_next(rng, length(values))
        shuffled, next_rng = shuffle_next(rng, values)
        @test vec(shuffled) == vec(values)[permutation]
        @test size(shuffled) == size(values)
        @test next_rng == permutation_next
        @test shuffle(rng, values) == shuffled
        in_place = copy(values)
        @test first(shuffle_next!(rng, in_place)) === in_place
        @test in_place == shuffled
        @test shuffle!(rng, copy(values)) == shuffled
    end
end

@testset "a sample without replacement is the shuffled prefix" begin
    for population in ([:a, :b, :c, :d, :e], 1:1000, "héxlo")
        rng = Philox4x32(0x7e5, 2)
        shuffled, shuffle_rng = shuffle_next(rng, collect(population))
        for count in (0, 1, 3)
            sample, next_rng = randsample_next(rng, population, count; replace = false)
            @test sample == shuffled[1:count]
            @test next_rng == shuffle_rng
            @test randsample(rng, population, count; replace = false) == sample
            destination = similar(shuffled, count)
            @test randsample!(rng, population, destination; replace = false) == sample
            @test last(randsample_next!(rng, population, destination; replace = false)) ==
                  shuffle_rng
        end
        @test first(randsample_next(rng, population; replace = false)) == shuffled
        @test allunique(randsample(rng, population, length(shuffled); replace = false))
        @test_throws ArgumentError randsample(
            rng,
            population,
            length(shuffled) + 1;
            replace = false,
        )
    end
end

@testset "the bridge permutes owned arrays by the pure law" begin
    rng = Philox4x32(0x7e6)
    values = collect(1:20)
    for (bridge_call, pure_call) in (
        (bridge -> randperm(bridge, 20), rng -> randperm_next(rng, 20)),
        (bridge -> randcycle(bridge, 20), rng -> randcycle_next(rng, 20)),
        (bridge -> shuffle(bridge, values), rng -> shuffle_next(rng, values)),
        (bridge -> shuffle!(bridge, copy(values)), rng -> shuffle_next(rng, values)),
    )
        bridge = StatefulRNG(rng)
        expected, next_rng = pure_call(rng)
        @test bridge_call(bridge) == expected
        @test parent(bridge) == next_rng
    end
end
