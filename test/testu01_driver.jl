include("statistical/testu01.jl")

const TestU01Driver = PureRNGsTestU01

@testset "TestU01 SmallCrush matrix is the pinned R50 matrix" begin
    matrix = TestU01Driver._matrix()
    @test length(matrix) == 32
    @test length(unique(matrix)) == 32
    @test Set(first.(matrix)) == Set(TestU01Driver.FAMILY_TYPES)
    @test Set(getindex.(matrix, 2)) == Set(TestU01Driver.STREAMS)
    @test Set(last.(matrix)) == Set(TestU01Driver.SCHEDULES)
    @test TestU01Driver.BATTERIES === (:SmallCrush, :Crush, :BigCrush)
    @test TestU01Driver.R50_CASE_COUNT == 32
    @test TestU01Driver.R50_P_VALUES_PER_CASE == 15
    @test TestU01Driver.R50_P_VALUE_COUNT == 480
    @test TestU01Driver.RELEASE_ALPHA === 0.001 / 480
end

@testset "TestU01 sequential state follows continuation draws" begin
    for F in TestU01Driver.FAMILY_TYPES, T in (UInt32, Float64)
        state = TestU01Driver._driver_state(F, :sequential)
        reference = F(TestU01Driver.ROOT_SEED)
        for _ = 1:24
            reference, expected = rand_next(reference, T)
            @test TestU01Driver._next_value!(state, T) === expected
            @test state.rng == reference
        end
    end
end

@testset "TestU01 interleaved state follows children in child order" begin
    for F in TestU01Driver.FAMILY_TYPES, T in (UInt32, Float64)
        state = TestU01Driver._driver_state(F, :interleaved)
        children = splitrng(F(TestU01Driver.ROOT_SEED), TestU01Driver.CHILD_COUNT)
        for draw = 1:24
            child = mod1(draw, TestU01Driver.CHILD_COUNT)
            children[child], expected = rand_next(children[child], T)
            @test TestU01Driver._next_value!(state, T) === expected
            @test state.child == mod1(draw + 1, TestU01Driver.CHILD_COUNT)
            @test state.children == children
        end
    end
end

@testset "TestU01 callbacks use the active typed state" begin
    bits_state = TestU01Driver._driver_state(Philox4x32, :sequential)
    TestU01Driver.ACTIVE_STATE[] = bits_state
    _, expected_bits = rand_next(Philox4x32(TestU01Driver.ROOT_SEED), UInt32)
    @test TestU01Driver._next_bits() === expected_bits

    uniform_state = TestU01Driver._driver_state(Threefry4x64, :interleaved)
    TestU01Driver.ACTIVE_STATE[] = uniform_state
    children = splitrng(Threefry4x64(TestU01Driver.ROOT_SEED), 8)
    _, first_child = rand_next(children[1], Float64)
    @test TestU01Driver._next_uniform() === first_child
    TestU01Driver.ACTIVE_STATE[] = nothing
end

@testset "TestU01 metadata and result schema are complete" begin
    io = IOBuffer()
    identities = [(role = :driver, path = "test/statistical/testu01.jl", sha256 = "a"^64)]
    TestU01Driver._write_metadata(io, :SmallCrush, identities)
    metadata = String(take!(io))
    for field in (
        "schema",
        "driver_package",
        "driver_version",
        "kernel_abstractions_version",
        "mldata_devices_version",
        "julia_version",
        "testu01_version",
        "battery",
        "architecture",
        "kernel",
        "cpu",
        "root_seed",
        "child_count",
        "interleave",
        "bits_api",
        "uniform_api",
        "diagnostic_interval",
        "release_alpha",
        "release_interval",
        "expected_cases",
        "expected_p_values_per_case",
        "expected_p_values",
    )
        @test occursin("# $field\t", metadata)
    end
    @test occursin("# schema\t2\n", metadata)
    @test endswith(
        metadata,
        "battery\tfamily\tstream\tschedule\tstatistic_index\tstatistic_name\tp_value\tp_value_bits\tfinite\twithin_diagnostic_interval\twithin_release_interval\tsuspect\n",
    )
    @test occursin(
        "# file_sha256\tdriver\ttest/statistical/testu01.jl\t$("a"^64)\n",
        metadata,
    )

    TestU01Driver._write_result(
        io,
        :SmallCrush,
        Philox2x32,
        :bits,
        :sequential,
        3,
        "name\twith\nspace",
        0.5,
    )
    row = String(take!(io))
    @test row ==
          "SmallCrush\tPhilox2x32\tbits\tsequential\t3\tname with space\t0.5\t3fe0000000000000\ttrue\ttrue\ttrue\tfalse\n"

    TestU01Driver._write_case_count(io, :SmallCrush, Philox2x32, :bits, :sequential, 15)
    @test String(take!(io)) ==
          "# case_p_value_count\tSmallCrush\tPhilox2x32\tbits\tsequential\t15\n"

    status = (
        completed_cases = 32,
        completed_p_values = 480,
        all_finite = true,
        all_diagnostic = false,
        all_release = true,
        counts_valid = true,
        release_applicable = true,
        matrix_complete = true,
        release_passed = true,
        diagnostic_passed = true,
    )
    TestU01Driver._write_completion(io, status)
    @test String(take!(io)) ==
          "# completed\ttrue\n# completed_cases\t32\n# completed_p_values\t480\n# all_p_values_finite\ttrue\n# all_within_diagnostic_interval\tfalse\n# all_within_release_interval\ttrue\n# all_case_counts_valid\ttrue\n# release_applicable\ttrue\n# matrix_complete\ttrue\n# r50_release_passed\ttrue\n# diagnostic_run_passed\ttrue\n"
end

@testset "TestU01 diagnostic and release intervals are distinct" begin
    alpha = TestU01Driver.RELEASE_ALPHA
    release_max = TestU01Driver.RELEASE_MAX
    @test TestU01Driver._within_diagnostic(0.001)
    @test TestU01Driver._within_diagnostic(0.999)
    @test !TestU01Driver._within_diagnostic(prevfloat(0.001))
    @test !TestU01Driver._within_diagnostic(nextfloat(0.999))
    @test TestU01Driver._within_release(alpha)
    @test TestU01Driver._within_release(release_max)
    @test !TestU01Driver._within_release(prevfloat(alpha))
    @test !TestU01Driver._within_release(nextfloat(release_max))
    @test TestU01Driver._is_suspect(prevfloat(0.001))
    @test TestU01Driver._is_suspect(nextfloat(0.999))
    @test TestU01Driver._is_suspect(0.9993672821429762)
    for value in (NaN, Inf, -Inf)
        @test !TestU01Driver._is_finite(value)
        @test !TestU01Driver._within_diagnostic(value)
        @test !TestU01Driver._within_release(value)
        @test !TestU01Driver._is_suspect(value)
    end
end

@testset "TestU01 release requires the complete R50 matrix" begin
    matrix = TestU01Driver._matrix()
    good = (
        p_values = 15,
        complete = true,
        all_finite = true,
        all_diagnostic = true,
        all_release = true,
    )
    status = TestU01Driver._r50_status(:SmallCrush, matrix, fill(good, 32))
    @test status.completed_cases == 32
    @test status.completed_p_values == 480
    @test status.counts_valid
    @test status.release_applicable
    @test status.matrix_complete
    @test status.release_passed
    @test status.diagnostic_passed

    shard = TestU01Driver._r50_status(:SmallCrush, matrix[1:1], [good])
    @test !shard.release_applicable
    @test !shard.matrix_complete
    @test !shard.release_passed
    @test shard.diagnostic_passed

    suspect = merge(good, (all_diagnostic = false,))
    shard = TestU01Driver._r50_status(:SmallCrush, matrix[1:1], [suspect])
    @test !shard.release_applicable
    @test !shard.diagnostic_passed

    wrong_count = merge(good, (p_values = 14,))
    status = TestU01Driver._r50_status(:SmallCrush, matrix, [fill(good, 31); wrong_count])
    @test !status.counts_valid
    @test !status.matrix_complete
    @test !status.release_passed
    @test !status.diagnostic_passed

    nonfinite = merge(good, (all_finite = false, all_release = false))
    status = TestU01Driver._r50_status(:SmallCrush, matrix, [fill(good, 31); nonfinite])
    @test !status.all_finite
    @test !status.all_release
    @test status.matrix_complete
    @test !status.release_passed
    @test !status.diagnostic_passed

    incomplete = merge(good, (complete = false,))
    status = TestU01Driver._r50_status(:SmallCrush, matrix, [fill(good, 31); incomplete])
    @test status.completed_cases == 31
    @test !status.matrix_complete
    @test !status.release_passed
    @test !status.diagnostic_passed

    crush = TestU01Driver._r50_status(:Crush, matrix[1:1], [good])
    @test !crush.release_applicable
    @test !crush.matrix_complete
    @test !crush.release_passed
    @test crush.diagnostic_passed
end

@testset "TestU01 version and case validation need no library" begin
    needle = collect(codeunits("TestU01 1.2.3"))
    @test TestU01Driver._has_version(vcat(UInt8[0x00], needle, UInt8[0xff]))
    @test !TestU01Driver._has_version(UInt8[0x01, 0x02])
    @test TestU01Driver._source_file("src/uniform.jl")
    @test !TestU01Driver._source_file("src/._uniform.jl")
    @test !TestU01Driver._source_file("src/uniform.c")

    output, battery, cases = TestU01Driver._parse_run(["result.tsv"])
    @test output == "result.tsv"
    @test battery === :SmallCrush
    @test cases == TestU01Driver._matrix()
    for expected_battery in TestU01Driver.BATTERIES
        output, battery, cases =
            TestU01Driver._parse_run(["result.tsv", string(expected_battery)])
        @test (output, battery, cases) ==
              ("result.tsv", expected_battery, TestU01Driver._matrix())
    end
    output, battery, cases =
        TestU01Driver._parse_run(["result.tsv", "Philox4x32", "bits", "sequential"])
    @test (output, battery, cases) ==
          ("result.tsv", :SmallCrush, [(Philox4x32, :bits, :sequential)])
    output, battery, cases = TestU01Driver._parse_run([
        "result.tsv",
        "BigCrush",
        "Philox4x32",
        "uniform",
        "interleaved",
    ])
    @test output == "result.tsv"
    @test battery === :BigCrush
    @test cases == [(Philox4x32, :uniform, :interleaved)]
    @test_throws ArgumentError TestU01Driver._parse_run(String[])
    @test_throws ArgumentError TestU01Driver._parse_run(["x", "NotCrush"])
    @test_throws ArgumentError TestU01Driver._parse_run(["x", "Bad", "bits", "sequential"])
    @test_throws ArgumentError TestU01Driver._parse_run([
        "x",
        "Philox4x32",
        "bad",
        "sequential",
    ])
    @test_throws ArgumentError TestU01Driver._parse_run(["x", "Philox4x32", "bits", "bad"])
end

@testset "Committed SmallCrush log is complete and current" begin
    log_path = joinpath(@__DIR__, "statistical", "smallcrush.tsv")
    lines = readlines(log_path)
    metadata = Dict{String,Vector{Vector{SubString{String}}}}()
    rows = Vector{Vector{SubString{String}}}()
    header = "battery\tfamily\tstream\tschedule\tstatistic_index\tstatistic_name\tp_value\tp_value_bits\tfinite\twithin_diagnostic_interval\twithin_release_interval\tsuspect"
    saw_header = false
    for line in lines
        if startswith(line, "# ")
            fields = split(line[3:end], '\t')
            push!(
                get!(Vector{Vector{SubString{String}}}, metadata, String(first(fields))),
                fields[2:end],
            )
        elseif line == header
            @test !saw_header
            saw_header = true
        else
            @test saw_header
            push!(rows, split(line, '\t'))
        end
    end

    scalar(key) = only(only(metadata[key]))
    @test scalar("schema") == string(TestU01Driver.DRIVER_SCHEMA)
    @test scalar("driver_package") == "PureRNGs"
    @test scalar("driver_version") == string(pkgversion(PureRNGs))
    @test scalar("testu01_version") == string(TestU01Driver.TESTU01_VERSION)
    @test scalar("battery") == "SmallCrush"
    @test scalar("architecture") == "x86_64"
    @test scalar("kernel") == "Linux"
    @test !isempty(scalar("cpu"))
    @test scalar("root_seed") == string(TestU01Driver.ROOT_SEED)
    @test scalar("child_count") == string(TestU01Driver.CHILD_COUNT)
    @test scalar("interleave") == "round-robin, one value per child in child order"
    @test scalar("bits_api") == "unif01_CreateExternGenBits(UInt32)"
    @test scalar("uniform_api") == "unif01_CreateExternGen01(Float64)"
    @test scalar("diagnostic_interval") ==
          "[$(TestU01Driver.DIAGNOSTIC_MIN), $(TestU01Driver.DIAGNOSTIC_MAX)]"
    @test parse(Float64, scalar("release_alpha")) === TestU01Driver.RELEASE_ALPHA
    @test scalar("release_interval") ==
          "[$(TestU01Driver.RELEASE_ALPHA), $(TestU01Driver.RELEASE_MAX)]"
    @test scalar("expected_cases") == string(TestU01Driver.R50_CASE_COUNT)
    @test scalar("expected_p_values_per_case") ==
          string(TestU01Driver.R50_P_VALUES_PER_CASE)
    @test scalar("expected_p_values") == string(TestU01Driver.R50_P_VALUE_COUNT)

    package_root = pkgdir(PureRNGs)
    source_paths = ["Project.toml"]
    for directory in ("src", "ext")
        append!(
            source_paths,
            relpath.(
                sort(
                    filter(
                        TestU01Driver._source_file,
                        readdir(joinpath(package_root, directory); join = true),
                    ),
                ),
                package_root,
            ),
        )
    end
    identities = metadata["file_sha256"]
    source_hashes =
        Dict(fields[2] => fields[3] for fields in identities if fields[1] == "source")
    @test Set(keys(source_hashes)) == Set(source_paths)
    for path in source_paths
        @test source_hashes[path] == TestU01Driver._sha256(joinpath(package_root, path))
    end
    driver = only(filter(fields -> fields[1] == "driver", identities))
    @test driver[2] == "test/statistical/testu01.jl"
    @test driver[3] == TestU01Driver._sha256(joinpath(package_root, driver[2]))
    for role in ("testu01", "probdist", "mylib")
        identity = only(filter(fields -> fields[1] == role, identities))
        @test occursin(r"^[0-9a-f]{64}$", identity[3])
    end

    @test saw_header
    @test length(rows) == TestU01Driver.R50_P_VALUE_COUNT
    case_indices = Dict{Tuple{String,String,String},Vector{Int}}()
    all_diagnostic = true
    all_release = true
    for row in rows
        @test length(row) == 12
        @test row[1] == "SmallCrush"
        F = TestU01Driver._family_type(row[2])
        stream = Symbol(row[3])
        schedule = Symbol(row[4])
        @test (F, stream, schedule) in TestU01Driver._matrix()
        index = parse(Int, row[5])
        push!(get!(Vector{Int}, case_indices, (row[2], row[3], row[4])), index)
        p_value = parse(Float64, row[7])
        @test parse(UInt64, row[8]; base = 16) == reinterpret(UInt64, p_value)
        @test parse(Bool, row[9]) == TestU01Driver._is_finite(p_value)
        diagnostic = TestU01Driver._within_diagnostic(p_value)
        release = TestU01Driver._within_release(p_value)
        @test parse(Bool, row[10]) == diagnostic
        @test parse(Bool, row[11]) == release
        @test parse(Bool, row[12]) == TestU01Driver._is_suspect(p_value)
        all_diagnostic &= diagnostic
        all_release &= release
    end
    @test length(case_indices) == TestU01Driver.R50_CASE_COUNT
    @test all(
        indices == collect(1:TestU01Driver.R50_P_VALUES_PER_CASE) for
        indices in values(case_indices)
    )

    case_counts = metadata["case_p_value_count"]
    @test length(case_counts) == TestU01Driver.R50_CASE_COUNT
    expected_cases = Set(
        (string(nameof(F)), string(stream), string(schedule)) for
        (F, stream, schedule) in TestU01Driver._matrix()
    )
    @test Set(Tuple(String.(fields[2:4])) for fields in case_counts) == expected_cases
    @test all(
        fields ->
            fields[1] == "SmallCrush" &&
            parse(Int, fields[5]) == TestU01Driver.R50_P_VALUES_PER_CASE,
        case_counts,
    )
    @test scalar("completed") == "true"
    @test scalar("completed_cases") == string(TestU01Driver.R50_CASE_COUNT)
    @test scalar("completed_p_values") == string(TestU01Driver.R50_P_VALUE_COUNT)
    @test parse(Bool, scalar("all_p_values_finite"))
    @test parse(Bool, scalar("all_within_diagnostic_interval")) == all_diagnostic
    @test parse(Bool, scalar("all_within_release_interval")) == all_release
    @test parse(Bool, scalar("all_case_counts_valid"))
    @test parse(Bool, scalar("release_applicable"))
    @test parse(Bool, scalar("matrix_complete"))
    @test parse(Bool, scalar("r50_release_passed"))
    @test parse(Bool, scalar("diagnostic_run_passed")) == all_diagnostic
    @test all_release
end
