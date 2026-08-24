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
