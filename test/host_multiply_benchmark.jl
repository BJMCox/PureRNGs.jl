# Temporary diagnostic branch. Do not merge this script.
using PureRNGs, BenchmarkTools, Test, InteractiveUtils

const MULTIPLY_MODE = ENV["PURERNGS_MULTIPLY_BENCHMARK"]
if MULTIPLY_MODE == "portable"
    @eval PureRNGs @inline _word_mulhilo(
        ::_HostWordOps,
        ::Val{64},
        a::UInt64,
        b::UInt64,
    ) = _mulhilo64(a, b)
end

versioninfo()
rng = Philox4x64(0x62a)
_, successor = Base.invokelatest(rand_next, rng, UInt64, 12)
value, _ = Base.invokelatest(rand_next, successor, UInt64)
println(
    "[MULTIPLY] mode=", MULTIPLY_MODE, " next=", value,
    " correct=", value == 0x88d25590be71246e,
)
if MULTIPLY_MODE == "portable"
    @test value == 0x88d25590be71246e
    @test Base.invokelatest(rand, Philox2x64(0), UInt64, 2) ==
          [0xca00a0459843d731, 0x66c24222c9a845b5]
end

function scalar_chain(rng, ::Type{T}, count) where {T}
    total = zero(T)
    for _ = 1:count
        value, rng = rand_next(rng, T)
        total += value
    end
    return total, rng
end

const BENCHMARK_DIR = joinpath(dirname(@__DIR__), "diagnostics")
mkpath(BENCHMARK_DIR)
open(joinpath(BENCHMARK_DIR, "$MULTIPLY_MODE.csv"), "w") do io
    println(io, "mode,family,case,min_ns,median_ns,memory,allocs,wall_seconds,samples")
    for F in (Philox2x64, Philox4x64)
        rng = F(0x62a)
        destination = Vector{UInt64}(undef, 262144)
        cases = (
            ("chain_u64", @benchmarkable(scalar_chain($rng, UInt64, 4096)), 100),
            ("chain_f64", @benchmarkable(scalar_chain($rng, Float64, 4096)), 100),
            ("batch12", @benchmarkable(rand_next($rng, UInt64, 12)), 10000),
            ("fill_u64", @benchmarkable(rand_next!($rng, $destination; threaded = false)), 10),
        )
        for (name, bench, evals) in cases
            BenchmarkTools.run(bench; seconds = 0.2, samples = 2, evals = 1)
            started = time_ns()
            trial = BenchmarkTools.run(bench; seconds = 10, samples = 1000000, evals)
            elapsed = (time_ns() - started) / 1.0e9
            best, middle = minimum(trial), BenchmarkTools.median(trial)
            row = join(
                (MULTIPLY_MODE, F, name, best.time, middle.time,
                 best.memory, best.allocs, elapsed, length(trial)),
                ',',
            )
            println(io, row)
            flush(io)
            println("[MULTIPLY] ", row)
        end
    end
end

if MULTIPLY_MODE == "portable"
    for file in (
        "philox.jl", "threefry.jl", "chacha.jl", "generators.jl",
        "oracle_conformance.jl", "derive.jl", "bits.jl", "uniform.jl",
        "uniform_allocating.jl",
    )
        include(file)
    end
end
