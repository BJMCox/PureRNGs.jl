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
        "testu01_summary_interval",
    )
        @test occursin("# $field\t", metadata)
    end
    @test endswith(
        metadata,
        "battery\tfamily\tstream\tschedule\tstatistic_index\tstatistic_name\tp_value\tp_value_bits\twithin_summary_interval\n",
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
          "SmallCrush\tPhilox2x32\tbits\tsequential\t3\tname with space\t0.5\t3fe0000000000000\ttrue\n"
    @test TestU01Driver._passes(0.001)
    @test TestU01Driver._passes(0.999)
    @test !TestU01Driver._passes(prevfloat(0.001))
    @test !TestU01Driver._passes(nextfloat(0.999))
    @test !TestU01Driver._passes(NaN)
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
