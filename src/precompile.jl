using PrecompileTools: @compile_workload, @setup_workload

# The generated 128-draw Float64 group and the fill schedulers dominate first-call
# latency, so the workload runs a real fill for every public draw kind rather than
# listing `precompile` signatures. The continuations destructure their result
# because that is how callers write them, and the destructuring itself compiles.
@setup_workload begin
    seed = 20250918
    draws = 128
    population = 1:10
    weights = collect(1.0:10.0)

    @compile_workload begin
        table = WeightTable(weights)
        for rng in (Philox4x32(seed), Threefry4x64(seed), ChaCha(seed))
            for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
                scalar, _ = rand_next(rng, T)
                array, _ = rand_next(rng, T, draws)
                rand(rng, T, draws)
                randat(rng, T, 1)
                destination = Vector{T}(undef, draws)
                # A call that passes no keyword compiles apart from the two that do.
                rand_next!(rng, destination)
                rand_next!(rng, destination; threaded = false)
                rand_next!(rng, destination; threaded = true)
            end

            for T in (Float32, Float64)
                normal, _ = randn_next(rng, T)
                normals, _ = randn_next(rng, T, draws)
                exponential, _ = randexp_next(rng, T)
                exponentials, _ = randexp_next(rng, T, draws)
                randnat(rng, T, 1)
                randexpat(rng, T, 1)
                destination = Vector{T}(undef, draws)
                randn_next!(rng, destination)
                randexp_next!(rng, destination)
            end

            dice = Vector{Int}(undef, draws)
            rand_next!(rng, dice, 1:6)
            rand_next!(rng, dice, 1:(2^40))
            rolls, _ = rand_next(rng, 1:6, draws)

            splitrng(rng, 4)
            subrng(rng, 1)

            randsample(rng, population, 3)
            randsample(rng, population, weights, 3)
            randsample(rng, population, table, 3)

            bridge = StatefulRNG(rng)
            rand(bridge, Float64)
            rand(bridge, Float64, draws)
        end
    end
end
