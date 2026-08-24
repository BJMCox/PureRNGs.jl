module PureRNGsTestU01

import PureRNGs
import Libdl

const DRIVER_SCHEMA = 2
const TESTU01_VERSION = v"1.2.3"
const ROOT_SEED = 12345
const CHILD_COUNT = 8
# TestU01 1.2.3 battery summaries list values outside this interval, then state
# that all remaining tests passed.
const DIAGNOSTIC_MIN = 0.001
const DIAGNOSTIC_MAX = 0.999
const R50_CASE_COUNT = 32
const R50_P_VALUES_PER_CASE = 15
const R50_P_VALUE_COUNT = R50_CASE_COUNT * R50_P_VALUES_PER_CASE
const RELEASE_ALPHA = 0.001 / R50_P_VALUE_COUNT
const RELEASE_MAX = 1.0 - RELEASE_ALPHA
const FAMILY_TYPES = (
    PureRNGs.Philox2x32,
    PureRNGs.Philox4x32,
    PureRNGs.Philox2x64,
    PureRNGs.Philox4x64,
    PureRNGs.Threefry2x32,
    PureRNGs.Threefry4x32,
    PureRNGs.Threefry2x64,
    PureRNGs.Threefry4x64,
)
const STREAMS = (:bits, :uniform)
const SCHEDULES = (:sequential, :interleaved)
const BATTERIES = (:SmallCrush, :Crush, :BigCrush)

abstract type AbstractDriverState end

mutable struct SequentialState{R} <: AbstractDriverState
    rng::R
end

mutable struct InterleavedState{R} <: AbstractDriverState
    children::Vector{R}
    child::Int
end

@inline function _next_value!(state::SequentialState, ::Type{T}) where {T}
    next_rng, value = PureRNGs.rand_next(state.rng, T)
    state.rng = next_rng
    return value
end

@inline function _next_value!(state::InterleavedState, ::Type{T}) where {T}
    child = state.child
    next_rng, value = PureRNGs.rand_next(@inbounds(state.children[child]), T)
    @inbounds state.children[child] = next_rng
    state.child = child == length(state.children) ? 1 : child + 1
    return value
end

const ACTIVE_STATE = Ref{Any}(nothing)

function _next_bits()::Cuint
    state = ACTIVE_STATE[]::AbstractDriverState
    return Cuint(_next_value!(state, UInt32))
end

function _next_uniform()::Cdouble
    state = ACTIVE_STATE[]::AbstractDriverState
    return Cdouble(_next_value!(state, Float64))
end

const BITS_CALLBACK = @cfunction(_next_bits, Cuint, ())
const UNIFORM_CALLBACK = @cfunction(_next_uniform, Cdouble, ())

struct TestU01API
    handle::Ptr{Cvoid}
end

TestU01API(library::AbstractString) = TestU01API(Libdl.dlopen(library))
@inline _symbol(api::TestU01API, name::Symbol) = Libdl.dlsym(api.handle, name)

@inline _family_name(F) = String(nameof(F))

function _family_type(name::AbstractString)
    for F in FAMILY_TYPES
        name == _family_name(F) && return F
    end
    throw(ArgumentError("unknown family: $name"))
end

function _driver_state(F, schedule::Symbol)
    root = F(ROOT_SEED)
    schedule === :sequential && return SequentialState(root)
    schedule === :interleaved &&
        return InterleavedState(PureRNGs.splitrng(root, CHILD_COUNT), 1)
    throw(ArgumentError("unknown schedule: $schedule"))
end

function _matrix()
    return [
        (F, stream, schedule) for F in FAMILY_TYPES for stream in STREAMS for
        schedule in SCHEDULES
    ]
end

@inline _has_version(bytes) =
    findfirst(codeunits("TestU01 $(TESTU01_VERSION)"), bytes) !== nothing

function _check_library_version(library::AbstractString)
    _has_version(read(library)) || throw(
        ArgumentError("TestU01 library does not identify as version $(TESTU01_VERSION)"),
    )
    return nothing
end

@inline _is_finite(p_value::Float64) = isfinite(p_value)
@inline _within_diagnostic(p_value::Float64) =
    _is_finite(p_value) && DIAGNOSTIC_MIN <= p_value <= DIAGNOSTIC_MAX
@inline _within_release(p_value::Float64) =
    _is_finite(p_value) && RELEASE_ALPHA <= p_value <= RELEASE_MAX
@inline _is_suspect(p_value::Float64) =
    _within_release(p_value) && !_within_diagnostic(p_value)

@inline function _battery_pointer(api::TestU01API, battery::Symbol)
    battery in BATTERIES || throw(ArgumentError("unknown battery: $battery"))
    return _symbol(api, Symbol("bbattery_", battery))
end

@inline _field(value) = replace(string(value), '\t' => ' ', '\n' => ' ', '\r' => ' ')

function _write_metadata(io::IO, battery::Symbol, identities)
    metadata = (
        :schema => DRIVER_SCHEMA,
        :driver_package => "PureRNGs",
        :driver_version => pkgversion(PureRNGs),
        :kernel_abstractions_version => pkgversion(PureRNGs.KernelAbstractions),
        :mldata_devices_version => pkgversion(PureRNGs.MLDataDevices),
        :julia_version => VERSION,
        :testu01_version => TESTU01_VERSION,
        :battery => battery,
        :architecture => Sys.ARCH,
        :kernel => Sys.KERNEL,
        :cpu => Sys.CPU_NAME,
        :root_seed => ROOT_SEED,
        :child_count => CHILD_COUNT,
        :interleave => "round-robin, one value per child in child order",
        :bits_api => "unif01_CreateExternGenBits(UInt32)",
        :uniform_api => "unif01_CreateExternGen01(Float64)",
        :diagnostic_interval => "[$DIAGNOSTIC_MIN, $DIAGNOSTIC_MAX]",
        :release_alpha => RELEASE_ALPHA,
        :release_interval => "[$RELEASE_ALPHA, $RELEASE_MAX]",
        :expected_cases => R50_CASE_COUNT,
        :expected_p_values_per_case => R50_P_VALUES_PER_CASE,
        :expected_p_values => R50_P_VALUE_COUNT,
    )
    for (key, value) in metadata
        println(io, "# ", key, '\t', _field(value))
    end
    for identity in identities
        println(
            io,
            "# file_sha256\t",
            _field(identity.role),
            '\t',
            _field(identity.path),
            '\t',
            identity.sha256,
        )
    end
    println(
        io,
        "battery\tfamily\tstream\tschedule\tstatistic_index\tstatistic_name\tp_value\tp_value_bits\tfinite\twithin_diagnostic_interval\twithin_release_interval\tsuspect",
    )
    return nothing
end

function _write_result(
    io::IO,
    battery::Symbol,
    family,
    stream::Symbol,
    schedule::Symbol,
    index::Int,
    name::AbstractString,
    p_value::Float64,
)
    bits = string(reinterpret(UInt64, p_value); base = 16, pad = 16)
    println(
        io,
        join(
            (
                battery,
                _family_name(family),
                stream,
                schedule,
                index,
                _field(name),
                repr(p_value),
                bits,
                _is_finite(p_value),
                _within_diagnostic(p_value),
                _within_release(p_value),
                _is_suspect(p_value),
            ),
            '\t',
        ),
    )
    return nothing
end

function _write_case_count(
    io::IO,
    battery::Symbol,
    family,
    stream::Symbol,
    schedule::Symbol,
    count::Int,
)
    println(
        io,
        "# case_p_value_count\t",
        join((battery, _family_name(family), stream, schedule, count), '\t'),
    )
    return nothing
end

function _write_completion(io::IO, status)
    println(io, "# completed\ttrue")
    println(io, "# completed_cases\t", status.completed_cases)
    println(io, "# completed_p_values\t", status.completed_p_values)
    println(io, "# all_p_values_finite\t", status.all_finite)
    println(io, "# all_within_diagnostic_interval\t", status.all_diagnostic)
    println(io, "# all_within_release_interval\t", status.all_release)
    println(io, "# all_case_counts_valid\t", status.counts_valid)
    println(io, "# release_applicable\t", status.release_applicable)
    println(io, "# matrix_complete\t", status.matrix_complete)
    println(io, "# r50_release_passed\t", status.release_passed)
    println(io, "# diagnostic_run_passed\t", status.diagnostic_passed)
    return nothing
end

function _create_generator(api::TestU01API, stream::Symbol, name::String)
    if stream === :bits
        symbol, callback = :unif01_CreateExternGenBits, BITS_CALLBACK
    elseif stream === :uniform
        symbol, callback = :unif01_CreateExternGen01, UNIFORM_CALLBACK
    else
        throw(ArgumentError("unknown stream: $stream"))
    end
    return ccall(_symbol(api, symbol), Ptr{Cvoid}, (Cstring, Ptr{Cvoid}), name, callback)
end

function _delete_generator(api::TestU01API, stream::Symbol, generator::Ptr{Cvoid})
    symbol = stream === :bits ? :unif01_DeleteExternGenBits : :unif01_DeleteExternGen01
    ccall(_symbol(api, symbol), Cvoid, (Ptr{Cvoid},), generator)
    return nothing
end

function _run_case!(
    io::IO,
    api::TestU01API,
    battery::Symbol,
    F,
    stream::Symbol,
    schedule::Symbol,
)
    ACTIVE_STATE[] = _driver_state(F, schedule)
    name = join((battery, _family_name(F), stream, schedule), '/')
    GC.@preserve name begin
        generator = C_NULL
        try
            generator = _create_generator(api, stream, name)
            generator == C_NULL && error("TestU01 failed to create generator $name")
            ccall(_battery_pointer(api, battery), Cvoid, (Ptr{Cvoid},), generator)
        finally
            generator == C_NULL || _delete_generator(api, stream, generator)
            ACTIVE_STATE[] = nothing
        end
    end

    test_count = Int(unsafe_load(Ptr{Cint}(_symbol(api, :bbattery_NTests))))
    test_names = Ptr{Ptr{UInt8}}(_symbol(api, :bbattery_TestNames))
    p_values = Ptr{Cdouble}(_symbol(api, :bbattery_pVal))
    all_finite = true
    all_diagnostic = true
    all_release = true
    for index = 1:max(test_count, 0)
        name_pointer = unsafe_load(test_names, index)
        test_name = name_pointer == C_NULL ? "" : unsafe_string(name_pointer)
        p_value = unsafe_load(p_values, index)
        _write_result(io, battery, F, stream, schedule, index, test_name, p_value)
        all_finite &= _is_finite(p_value)
        all_diagnostic &= _within_diagnostic(p_value)
        all_release &= _within_release(p_value)
    end
    _write_case_count(io, battery, F, stream, schedule, test_count)
    flush(io)
    return (
        p_values = test_count,
        complete = true,
        all_finite = all_finite,
        all_diagnostic = all_diagnostic,
        all_release = all_release,
    )
end

function _r50_status(battery::Symbol, cases, results)
    completed_cases = count(result -> result.complete, results)
    completed_p_values = sum(result -> result.p_values, results; init = 0)
    all_finite = all(result -> result.all_finite, results)
    all_diagnostic = all(result -> result.all_diagnostic, results)
    all_release = all(result -> result.all_release, results)
    counts_valid = if battery === :SmallCrush
        all(result -> result.p_values == R50_P_VALUES_PER_CASE, results)
    else
        all(result -> result.p_values >= 0, results)
    end
    release_applicable = battery === :SmallCrush && cases == _matrix()
    matrix_complete =
        release_applicable &&
        completed_cases == R50_CASE_COUNT &&
        completed_p_values == R50_P_VALUE_COUNT
    release_passed = matrix_complete && counts_valid && all_finite && all_release
    diagnostic_passed =
        completed_cases == length(cases) && counts_valid && all_finite && all_diagnostic
    return (;
        completed_cases,
        completed_p_values,
        all_finite,
        all_diagnostic,
        all_release,
        counts_valid,
        release_applicable,
        matrix_complete,
        release_passed,
        diagnostic_passed,
    )
end

function _battery(name::AbstractString)
    battery = Symbol(name)
    battery in BATTERIES || throw(ArgumentError("unknown battery: $name"))
    return battery
end

function _parse_run(args::Vector{String})
    length(args) == 1 && return args[1], :SmallCrush, _matrix()
    length(args) == 2 && return args[1], _battery(args[2]), _matrix()
    length(args) in (4, 5) || throw(
        ArgumentError(
            "usage: testu01.jl OUTPUT [BATTERY] [FAMILY {bits|uniform} {sequential|interleaved}]",
        ),
    )
    offset = length(args) == 4 ? 0 : 1
    battery = iszero(offset) ? :SmallCrush : _battery(args[2])
    F = _family_type(args[2+offset])
    stream_name = args[3+offset]
    stream = Symbol(stream_name)
    stream in STREAMS || throw(ArgumentError("unknown stream: $stream_name"))
    schedule_name = args[4+offset]
    schedule = Symbol(schedule_name)
    schedule in SCHEDULES || throw(ArgumentError("unknown schedule: $schedule_name"))
    return args[1], battery, [(F, stream, schedule)]
end

function _library_path()
    haskey(ENV, "TESTU01_LIBRARY") && return ENV["TESTU01_LIBRARY"]
    haskey(ENV, "TESTU01_PREFIX") ||
        throw(ArgumentError("set TESTU01_PREFIX or TESTU01_LIBRARY"))
    return joinpath(ENV["TESTU01_PREFIX"], "lib", "libtestu01.$(Libdl.dlext)")
end

function _sha256(path::AbstractString)
    output = read(`sha256sum -- $path`, String)
    digest = first(split(output))
    occursin(r"^[0-9a-f]{64}$", digest) || error("invalid sha256sum output for $path")
    return digest
end

function _identity(role, path, root = nothing)
    resolved = realpath(path)
    shown = root === nothing ? resolved : relpath(resolved, root)
    return (role = role, path = shown, sha256 = _sha256(resolved))
end

@inline _source_file(path::AbstractString) =
    endswith(path, ".jl") && !startswith(basename(path), "._")

function _identities(library::AbstractString)
    package_root = realpath(pkgdir(PureRNGs))
    library_directory = dirname(realpath(library))
    identities = [
        _identity(:driver, @__FILE__, package_root),
        _identity(:testu01, library),
        _identity(:probdist, joinpath(library_directory, "libprobdist.$(Libdl.dlext)")),
        _identity(:mylib, joinpath(library_directory, "libmylib.$(Libdl.dlext)")),
        _identity(:source, joinpath(package_root, "Project.toml"), package_root),
    ]
    for directory in ("src", "ext")
        root = joinpath(package_root, directory)
        isdir(root) || continue
        for path in sort(filter(_source_file, readdir(root; join = true)))
            push!(identities, _identity(:source, path, package_root))
        end
    end
    return identities
end

function main(args::Vector{String})
    Sys.islinux() && Sys.ARCH === :x86_64 ||
        throw(ArgumentError("R50 requires an x86-64 Linux runner"))
    output, battery, cases = _parse_run(args)
    ispath(output) && throw(ArgumentError("output path already exists: $output"))
    library = _library_path()
    isfile(library) || throw(ArgumentError("TestU01 library not found: $library"))
    _check_library_version(library)
    api = TestU01API(library)
    passed = try
        final_output = abspath(output)
        temporary, io = mktemp(dirname(final_output); cleanup = false)
        try
            _write_metadata(io, battery, _identities(library))
            results = map(cases) do case
                _run_case!(io, api, battery, case...)
            end
            status = _r50_status(battery, cases, results)
            _write_completion(io, status)
            close(io)
            mv(temporary, final_output)
            status.release_applicable ? status.release_passed : status.diagnostic_passed
        catch
            isopen(io) && close(io)
            rethrow()
        end
    finally
        Libdl.dlclose(api.handle)
    end
    return passed ? 0 : 1
end

end # module PureRNGsTestU01

if abspath(PROGRAM_FILE) == @__FILE__
    exit(PureRNGsTestU01.main(ARGS))
end
