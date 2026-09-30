# Include from a benchmark REPL, then run_regression("apple-cpu") or "a100".
# Load CUDA before a CUDA run. The command-line forms remain supported.
using PureRNGs
using BenchmarkTools
using Distributions
using LinearAlgebra
using Printf
using TOML
import KernelAbstractions
import MLDataDevices

const TYPES = Dict(
    string(T) => T for
    T in (Bool, UInt8, UInt16, UInt32, UInt64, Int32, Int64, Float16, Float32, Float64)
)
const FILLS = Dict("rand" => rand_next!, "randn" => randn_next!, "randexp" => randexp_next!)

function case_name(case)
    parts = [case["kind"], case["family"], get(case, "type", ""), string(case["elements"])]
    for key in ("shape", "components", "storage", "cached", "threaded")
        haskey(case, key) && push!(parts, "$key=$(case[key])")
    end
    haskey(case, "count") &&
        push!(parts, "count=$(case["count"]) replace=$(case["replace"])")
    return join(filter(!isempty, parts), " ")
end

function case_distribution(case, ::Type{T}) where {T}
    kind = case["kind"]
    kind == "gamma" && return Gamma(T(case["shape"]), one(T))
    kind == "beta" && return Beta(T(0.3), T(0.4))
    n = case["components"]
    kind == "dirichlet" && return Dirichlet(fill(T(0.3), n))
    covariance =
        get(case, "storage", "diagonal") == "dense" ?
        Matrix{T}(I, n, n) + fill(T(0.1), n, n) : Diagonal(ones(T, n))
    return MvNormal(zeros(T, n), covariance)
end

# Build fixtures before timing. Distribution cases time an in-place public call,
# including any parameter validation or device preparation that call performs.
function case_operation(case, target)
    rng = target(getfield(PureRNGs, Symbol(case["family"]))(12345))
    n = case["elements"]
    threaded = get(case, "threaded", false)
    kind = case["kind"]
    if haskey(FILLS, kind)
        values =
            get(case, "storage", "array") == "packed" ? falses(n) :
            rand(rng, TYPES[case["type"]], n)
        fill! = FILLS[kind]
        operation = let values = values
            () -> fill!(rng, values; threaded)
        end
        return operation, values
    elseif kind == "randperm"
        return () -> randperm_next(rng, n; threaded), nothing
    elseif kind == "weighted"
        population = target(collect(Int32(1):Int32(n)))
        raw_weights = target([mod(7index, 11) + 0.5 for index = 1:n])
        weights = get(case, "cached", false) ? WeightTable(raw_weights) : raw_weights
        count, replace = case["count"], case["replace"]
        return () -> randsample(rng, population, weights, count; replace, threaded), nothing
    elseif kind in ("gamma", "beta", "dirichlet", "mvnormal")
        T = TYPES[case["type"]]
        d = case_distribution(case, T)
        dims = haskey(case, "components") ? (case["components"], n) : (n,)
        values = rand(rng, T, dims...)
        operation = let values = values
            () -> rand_next!(rng, d, values; threaded)
        end
        return operation, values
    end
    error("unknown case kind $kind")
end

function measure(case, target; seconds = 10.0)
    seconds >= 10 || throw(ArgumentError("use at least 10 seconds per case"))
    operation, values = case_operation(case, target)
    backend = KernelAbstractions.get_backend(target(zeros(Float32, 1)))
    operation()
    KernelAbstractions.synchronize(backend)
    trial = @benchmark begin
        $operation()
        KernelAbstractions.synchronize($backend)
    end seconds = seconds evals = 1 samples = 100_000_000
    minimum_ns = minimum(trial).time
    output_bytes = values === nothing ? 0 : sizeof(values)
    return Dict{String,Any}(
        "minimum_ms" => minimum_ns / 1e6,
        "median_ms" => median(trial).time / 1e6,
        "host_bytes" => trial.memory,
        "host_allocations" => trial.allocs,
        "samples" => length(trial),
        "output_gib_per_second" => output_bytes / 2.0^30 / (minimum_ns / 1e9),
        "values_per_second" =>
            values === nothing ? 0.0 : length(values) / (minimum_ns / 1e9),
    )
end

# A short distribution sweep, not a family-by-type Cartesian product. These
# unpinned cases collect evidence; only measured, reviewed results become pins.
function distribution_cases(; elements = 2^18)
    cases = [Dict{String,Any}("kind" => "gamma", "shape" => a) for a in (0.3, 1.0, 2.5)]
    push!(cases, Dict("kind" => "beta"))
    for n in (3, 128)
        push!(cases, Dict("kind" => "dirichlet", "components" => n))
    end
    for storage in ("diagonal", "dense")
        push!(cases, Dict("kind" => "mvnormal", "components" => 16, "storage" => storage))
    end
    for case in cases
        merge!(
            case,
            Dict(
                "family" => "Philox4x32",
                "type" => "Float64",
                "elements" => max(1, elements ÷ get(case, "components", 1)),
            ),
        )
    end
    return cases
end

function benchmark_metadata(device)
    root = dirname(@__DIR__)
    modules = (PureRNGs, BenchmarkTools, Distributions, KernelAbstractions, MLDataDevices)
    packages = Dict(string(nameof(m)) => string(Base.pkgversion(m)) for m in modules)
    data = Dict{String,Any}(
        "source_sha" => strip(read(`git -C $root rev-parse HEAD`, String)),
        "timestamp_unix" => time(),
        "dirty" => !isempty(read(`git -C $root status --porcelain`, String)),
        "julia" => string(VERSION),
        "host" => gethostname(),
        "device" => device,
        "cpu" => first(Sys.cpu_info()).model,
        "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "load_average" => Sys.loadavg(),
        "packages" => packages,
    )
    if device == "cuda"
        data["gpu"] = CUDA.name(CUDA.device())
        device = CUDA.NVML.Device(CUDA.uuid(CUDA.device()))
        data["gpu_load"] = CUDA.NVML.utilization_rates(device).compute
        data["gpu_memory_bytes"] = CUDA.NVML.memory_info(device).used
        packages["CUDA"] = string(Base.pkgversion(CUDA))
    end
    return data
end

function run_regression(name; update = false, extra_cases = [], report_path = nothing)
    baseline_path = joinpath(@__DIR__, "baselines", name * ".toml")
    baseline = TOML.parsefile(baseline_path)
    cuda = baseline["device"] == "cuda"
    cuda && !isdefined(@__MODULE__, :CUDA) && error("import CUDA before the CUDA benchmark")
    target = cuda ? MLDataDevices.CUDADevice() : MLDataDevices.CPUDevice()
    metadata = benchmark_metadata(baseline["device"])
    TOML.print(stdout, metadata; sorted = true)
    get(baseline, "threads", Threads.nthreads()) == Threads.nthreads() ||
        @warn "thread count differs from the pin" pinned = baseline["threads"]
    rows, regressions = [], String[]
    for case in vcat(baseline["case"], extra_cases)
        load_before = Sys.loadavg()
        result = measure(case, target; seconds = max(10.0, get(baseline, "seconds", 10.0)))
        ratio = result["minimum_ms"] / get(case, "minimum_ms", NaN)
        ratio > 1 + baseline["tolerance"] && push!(regressions, case_name(case))
        push!(
            rows,
            merge(
                copy(case),
                result,
                Dict("load_before" => load_before, "load_after" => Sys.loadavg()),
            ),
        )
        @printf(
            "%s: %.4f ms min, %.4f ms median, %d host bytes, %d allocations, %.3fx pin\n",
            case_name(case),
            result["minimum_ms"],
            result["median_ms"],
            result["host_bytes"],
            result["host_allocations"],
            ratio
        )
        update && merge!(case, result)
    end
    report = Dict("metadata" => metadata, "case" => rows, "regressions" => regressions)
    report_path === nothing ||
        open(io -> TOML.print(io, report; sorted = true), report_path, "w")
    if update
        merge!(baseline, metadata)
        baseline["seconds"] = 10.0
        open(io -> TOML.print(io, baseline; sorted = true), baseline_path, "w")
    end
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: regression.jl <baseline> [--update]")
    path = joinpath(@__DIR__, "baselines", ARGS[1] * ".toml")
    TOML.parsefile(path)["device"] == "cuda" && @eval import CUDA
    report = run_regression(ARGS[1]; update = "--update" in ARGS)
    isempty(report["regressions"]) || "--update" in ARGS || exit(1)
end
