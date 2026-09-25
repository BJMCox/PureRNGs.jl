# Pinned throughput check. From the repository root:
#
#     julia --project=benchmark benchmark/regression.jl a100
#     julia --project=benchmark benchmark/regression.jl a100 --update
#
# Each case's minimum time must stay within the baseline's tolerance of its pin;
# a slower case fails the run with exit code 1. `--update` rewrites the pins,
# the host, and the Julia version from this run.
using PureRNGs
using BenchmarkTools
using Printf
using TOML
import KernelAbstractions
import MLDataDevices

length(ARGS) >= 1 || error("usage: regression.jl <baseline> [--update]")
const BASELINE_PATH = joinpath(@__DIR__, "baselines", ARGS[1] * ".toml")
const UPDATE = "--update" in ARGS
const BASELINE = TOML.parsefile(BASELINE_PATH)

BASELINE["device"] == "cuda" && @eval import CUDA
const TARGET =
    BASELINE["device"] == "cuda" ? MLDataDevices.CUDADevice() : MLDataDevices.CPUDevice()
const TYPES = Dict(
    string(T) => T for
    T in (Bool, UInt8, UInt16, UInt32, UInt64, Int32, Int64, Float16, Float32, Float64)
)
const FILLS = Dict("rand" => rand_next!, "randn" => randn_next!, "randexp" => randexp_next!)

function case_name(case)
    parts = [case["kind"], case["family"], get(case, "type", ""), string(case["elements"])]
    get(case, "threaded", false) && push!(parts, "threaded")
    haskey(case, "count") &&
        push!(parts, "count=$(case["count"]) replace=$(case["replace"])")
    return join(filter(!isempty, parts), " ")
end

# The operation a case times, with its output element size for fills.
function case_operation(case)
    rng = TARGET(getfield(PureRNGs, Symbol(case["family"]))(12345))
    n = case["elements"]
    threaded = get(case, "threaded", false)
    kind = case["kind"]
    if haskey(FILLS, kind)
        values = rand(rng, TYPES[case["type"]], n)
        return () -> FILLS[kind](rng, values; threaded), values
    elseif kind == "randperm"
        return () -> randperm_next(rng, n; threaded), nothing
    elseif kind == "weighted"
        population = TARGET(collect(Int32(1):Int32(n)))
        weights = TARGET([mod(7index, 11) + 0.5 for index = 1:n])
        count, replace = case["count"], case["replace"]
        return () -> randsample(rng, population, weights, count; replace, threaded), nothing
    end
    error("unknown case kind $kind")
end

function measure(case)
    operation, values = case_operation(case)
    backend = KernelAbstractions.get_backend(TARGET(zeros(Float32, 1)))
    trial = @benchmark begin
        $operation()
        KernelAbstractions.synchronize($backend)
    end seconds = get(BASELINE, "seconds", 3.0)
    # The minimum resists garbage-collection noise in the allocating cases,
    # which moves their medians by more than 10% between identical runs.
    minimum_ms = minimum(trial).time / 1.0e6
    gib = values === nothing ? NaN : sizeof(values) / 2.0^30 / (minimum_ms / 1.0e3)
    return minimum_ms, gib
end

tolerance = BASELINE["tolerance"]
get(BASELINE, "threads", Threads.nthreads()) == Threads.nthreads() ||
    @warn "the pins used $(BASELINE["threads"]) threads; this run has $(Threads.nthreads())"
regressions = String[]
@printf("%-58s %10s %10s %7s %9s\n", "case", "pinned ms", "minimum ms", "ratio", "GiB/s")
for case in BASELINE["case"]
    minimum_ms, gib = measure(case)
    pinned = get(case, "minimum_ms", NaN)
    ratio = minimum_ms / pinned
    status = ratio > 1 + tolerance ? "SLOWER" : ratio < 1 - tolerance ? "faster" : ""
    status == "SLOWER" && push!(regressions, case_name(case))
    @printf(
        "%-58s %10.3f %10.3f %7.3f %9.1f %s\n",
        case_name(case),
        pinned,
        minimum_ms,
        ratio,
        gib,
        status
    )
    UPDATE && (case["minimum_ms"] = round(minimum_ms; sigdigits = 4))
end

if UPDATE
    BASELINE["host"] = gethostname()
    BASELINE["julia"] = string(VERSION)
    BASELINE["threads"] = Threads.nthreads()
    open(io -> TOML.print(io, BASELINE; sorted = true), BASELINE_PATH, "w")
    println("updated ", BASELINE_PATH)
elseif !isempty(regressions)
    println("slower than the pins by more than ", tolerance, ": ", join(regressions, ", "))
    exit(1)
end
