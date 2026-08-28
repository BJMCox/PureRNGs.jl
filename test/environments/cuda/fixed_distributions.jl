const CUDA_FIXED_DISTRIBUTIONS = (
    Normal{Float32}(0.75f0, 1.25f0),
    Uniform{Float32}(-1.5f0, 2.75f0),
    Distributions.Exponential{Float32}(1.5f0),
    Bernoulli{Float32}(0.375f0),
    Normal{Float64}(0.75, 1.25),
    Uniform{Float64}(-1.5, 2.75),
    Distributions.Exponential{Float64}(1.5),
    Bernoulli{Float64}(0.375),
    DiscreteUniform(-11, 17),
)

@inline function _write_distribution_probe!(destination, offset, rng, distribution)
    next_rng, continued = rand_next(rng, distribution)
    @inbounds begin
        destination[offset+1] = rand(rng, distribution)
        destination[offset+2] = continued
        destination[offset+3] = randat(rng, distribution, 1)
        destination[offset+4] = rand(next_rng, distribution)
        destination[offset+5] = randat(rng, distribution, 2)
    end
    return nothing
end

function _fixed_distribution_kernel!(values32, values64, bools, integers, rng, dists)
    if CUDA.threadIdx().x == 1
        _write_distribution_probe!(values32, 0, rng, dists[1])
        _write_distribution_probe!(values32, 5, rng, dists[2])
        _write_distribution_probe!(values32, 10, rng, dists[3])
        _write_distribution_probe!(bools, 0, rng, dists[4])
        _write_distribution_probe!(values64, 0, rng, dists[5])
        _write_distribution_probe!(values64, 5, rng, dists[6])
        _write_distribution_probe!(values64, 10, rng, dists[7])
        _write_distribution_probe!(bools, 5, rng, dists[8])
        _write_distribution_probe!(integers, 0, rng, dists[9])
    end
    return
end

function _cuda_copy_sizes(profile)
    h2d = [
        profile.device.size[index] for index in eachindex(profile.device.name) if
        profile.device.name[index] == "[copy pageable to device memory]"
    ]
    d2h = [
        profile.device.size[index] for index in eachindex(profile.device.name) if
        profile.device.name[index] == "[copy device to pageable memory]"
    ]
    return h2d, d2h
end

function _distribution_fill_functions(distribution)
    pure =
        (rng, destination; threaded = true) ->
            rand!(rng, distribution, destination; threaded)
    continued =
        (rng, destination; threaded = true) ->
            rand_next!(rng, distribution, destination; threaded)
    return pure, continued
end

@testset "CUDA fixed-distribution scalar kernel probes" begin
    rng = device(Philox4x32(0x64c0))
    values32 = CUDA.CuArray{Float32}(undef, 15)
    values64 = CUDA.CuArray{Float64}(undef, 15)
    bools = CUDA.CuArray{Bool}(undef, 10)
    integers = CUDA.CuArray{Int}(undef, 5)
    args = (values32, values64, bools, integers, rng, CUDA_FIXED_DISTRIBUTIONS)

    CUDA.@sync CUDA.@cuda threads = 1 blocks = 1 _fixed_distribution_kernel!(args...)
    signature = Tuple{map(typeof, args)...}
    typed_text = sprint(show, CUDA.code_typed(_fixed_distribution_kernel!, signature))
    llvm_text = sprint(io -> CUDA.code_llvm(io, _fixed_distribution_kernel!, signature))
    for forbidden in ("StatefulRNG", "Task", "RefValue", "BigInt", "UInt128")
        @test !occursin(forbidden, typed_text)
    end
    @test !occursin(r"\bi128\b", llvm_text)

    observed = (
        Array(values32)[1:5],
        Array(values32)[6:10],
        Array(values32)[11:15],
        Array(bools)[1:5],
        Array(values64)[1:5],
        Array(values64)[6:10],
        Array(values64)[11:15],
        Array(bools)[6:10],
        Array(integers),
    )
    for (distribution, values) in zip(CUDA_FIXED_DISTRIBUTIONS, observed)
        expected = Array(rand(rng, distribution, 2))
        @test isequal(
            values,
            [expected[1], expected[1], expected[1], expected[2], expected[2]],
        )
    end
end

@testset "CUDA fixed-distribution arrays and fills" begin
    extension = Base.get_extension(IR, :PureRNGsDistributionsExt)
    @test extension !== nothing
    for F in FAMILIES, distribution in CUDA_FIXED_DISTRIBUTIONS
        cpu_rng = F(0x64c1)
        gpu_rng = device(cpu_rng)
        result_type = extension._result_type(distribution)
        fills = (_distribution_fill_functions(distribution),)
        exact = !(distribution isa Union{Normal,Distributions.Exponential})
        _check_array_draw(
            cpu_rng,
            gpu_rng,
            distribution,
            result_type,
            rand,
            rand_next;
            fills,
            cpu_parity = exact,
        )
    end
end

@testset "CUDA fixed-distribution capacity and empty fills" begin
    extension = Base.get_extension(IR, :PureRNGsDistributionsExt)
    rng = device(Philox4x32(0x64c2))
    for distribution in CUDA_FIXED_DISTRIBUTIONS
        T = extension._result_type(distribution)
        width = extension._distribution_span(distribution)
        last_rng = _last_draw_rng(rng, width)
        terminal, _ = rand_next(last_rng, distribution)
        @test terminal.position == _terminal(rng)

        array_terminal, values = rand_next(last_rng, distribution, 1)
        destination = similar(values)
        fill_terminal, returned = rand_next!(last_rng, distribution, destination)
        @test returned === destination
        @test isequal(Array(destination), Array(values))
        @test fill_terminal.position == array_terminal.position == terminal.position

        sentinel = T === Bool ? true : T(-1)
        failed = CUDA.fill(sentinel, 2)
        @test_throws ArgumentError rand_next!(last_rng, distribution, failed)
        @test Array(failed) == fill(sentinel, 2)
        @test_throws ArgumentError rand(terminal, distribution)

        empty = CUDA.CuArray{T}(undef, 0)
        profile = CUDA.@profile raw = true rand_next!(terminal, distribution, empty)
        @test count(value -> !ismissing(value), profile.device.grid) == 0
        empty_next, returned_empty = rand_next!(terminal, distribution, empty)
        @test returned_empty === empty
        @test empty_next.position == terminal.position
    end
end

@testset "CUDA fixed-distribution fills do not stage through the host" begin
    rng = device(Philox4x32(0x64c3))
    for distribution in CUDA_FIXED_DISTRIBUTIONS
        destination = rand(rng, distribution, 4096)
        rand_next!(rng, distribution, destination)
        CUDA.synchronize()
        profile = CUDA.Profile.profile_internally(; concurrent = false, trace = true) do
            rand_next!(rng, distribution, destination)
            CUDA.synchronize()
        end
        h2d, d2h = _cuda_copy_sizes(profile)
        @test all(==(8), h2d)
        @test isempty(d2h)
        @test _device_id(destination) == CUDA.deviceid(primary)
    end
end
