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

function _fixed_distribution_compile_kernel!(destination, rng, distribution)
    if CUDA.threadIdx().x == 1
        @inbounds destination[1] = rand(rng, distribution)
    end
    return
end

@inline _primitive(rng, ::Normal{T}) where {T} = randn(rng, T)
@inline _primitive(rng, ::Uniform{T}) where {T} = rand(rng, T)
@inline _primitive(rng, ::Distributions.Exponential{T}) where {T} = randexp(rng, T)
@inline _primitive(rng, ::Bernoulli{T}) where {T} = rand(rng, T)
@inline _primitive(rng, distribution::DiscreteUniform) =
    rand(rng, distribution.a:distribution.b)

@inline _primitive_next(rng, ::Normal{T}) where {T} = randn_next(rng, T)
@inline _primitive_next(rng, ::Uniform{T}) where {T} = rand_next(rng, T)
@inline _primitive_next(rng, ::Distributions.Exponential{T}) where {T} =
    randexp_next(rng, T)
@inline _primitive_next(rng, ::Bernoulli{T}) where {T} = rand_next(rng, T)
@inline _primitive_next(rng, distribution::DiscreteUniform) =
    rand_next(rng, distribution.a:distribution.b)

@inline _primitive_at(rng, ::Normal{T}, index) where {T} = randnat(rng, T, index)
@inline _primitive_at(rng, ::Uniform{T}, index) where {T} = randat(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Exponential{T}, index) where {T} =
    randexpat(rng, T, index)
@inline _primitive_at(rng, ::Bernoulli{T}, index) where {T} = randat(rng, T, index)
@inline function _primitive_at(rng, distribution::DiscreteUniform, index)
    span = (distribution.b % UInt64 - distribution.a % UInt64) + UInt64(1)
    addressed = IR._addressed_rng(rng, IR._range_bits(span), index)
    return rand(addressed, distribution.a:distribution.b)
end

@inline _primitive_array(rng, ::Normal{T}, count) where {T} = randn(rng, T, count)
@inline _primitive_array(rng, ::Uniform{T}, count) where {T} = rand(rng, T, count)
@inline _primitive_array(rng, ::Distributions.Exponential{T}, count) where {T} =
    randexp(rng, T, count)
@inline _primitive_array(rng, ::Bernoulli{T}, count) where {T} = rand(rng, T, count)
@inline _primitive_array(rng, distribution::DiscreteUniform, count) =
    rand(rng, distribution.a:distribution.b, count)

@noinline _distribution_oracle(distribution::Normal, value) =
    fma(distribution.σ, value, distribution.μ)
@noinline function _distribution_oracle(distribution::Uniform, value)
    width = distribution.b - distribution.a
    scaled = width * value
    return distribution.a + scaled
end
@noinline _distribution_oracle(distribution::Distributions.Exponential, value) =
    distribution.θ * value
@noinline _distribution_oracle(distribution::Bernoulli, value) = value < distribution.p
@noinline _distribution_oracle(::DiscreteUniform, value) = value

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

@inline function _write_primitive_probe!(destination, offset, rng, distribution)
    next_rng, continued = _primitive_next(rng, distribution)
    @inbounds begin
        destination[offset+1] = _primitive(rng, distribution)
        destination[offset+2] = continued
        destination[offset+3] = _primitive_at(rng, distribution, 1)
        destination[offset+4] = _primitive(next_rng, distribution)
        destination[offset+5] = _primitive_at(rng, distribution, 2)
    end
    return nothing
end

function _fixed_distribution_kernel!(
    values32,
    values64,
    bools,
    integers,
    primitives32,
    primitives64,
    primitive_integers,
    rng,
    dists,
)
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
        _write_primitive_probe!(primitives32, 0, rng, dists[1])
        _write_primitive_probe!(primitives32, 5, rng, dists[2])
        _write_primitive_probe!(primitives32, 10, rng, dists[3])
        _write_primitive_probe!(primitives32, 15, rng, dists[4])
        _write_primitive_probe!(primitives64, 0, rng, dists[5])
        _write_primitive_probe!(primitives64, 5, rng, dists[6])
        _write_primitive_probe!(primitives64, 10, rng, dists[7])
        _write_primitive_probe!(primitives64, 15, rng, dists[8])
        _write_primitive_probe!(primitive_integers, 0, rng, dists[9])
    end
    return
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

@testset "CUDA fixed-distribution validation compiles" begin
    extension = Base.get_extension(IR, :PureRNGsDistributionsExt)
    rng = device(Philox4x32(0x64c0))
    for distribution in CUDA_FIXED_DISTRIBUTIONS
        destination = CUDA.CuArray{extension._result_type(distribution)}(undef, 1)
        signature = Tuple{typeof(destination),typeof(rng),typeof(distribution)}
        llvm_text =
            sprint(io -> CUDA.code_llvm(io, _fixed_distribution_compile_kernel!, signature))
        @test !isempty(llvm_text)
    end
end

@testset "CUDA fixed-distribution scalar kernel probes" begin
    rng = device(Philox4x32(0x64c0))
    values32 = CUDA.CuArray{Float32}(undef, 15)
    values64 = CUDA.CuArray{Float64}(undef, 15)
    bools = CUDA.CuArray{Bool}(undef, 10)
    integers = CUDA.CuArray{Int}(undef, 5)
    primitives32 = CUDA.CuArray{Float32}(undef, 20)
    primitives64 = CUDA.CuArray{Float64}(undef, 20)
    primitive_integers = CUDA.CuArray{Int}(undef, 5)
    args = (
        values32,
        values64,
        bools,
        integers,
        primitives32,
        primitives64,
        primitive_integers,
        rng,
        CUDA_FIXED_DISTRIBUTIONS,
    )

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
    primitive = (
        Array(primitives32)[1:5],
        Array(primitives32)[6:10],
        Array(primitives32)[11:15],
        Array(primitives32)[16:20],
        Array(primitives64)[1:5],
        Array(primitives64)[6:10],
        Array(primitives64)[11:15],
        Array(primitives64)[16:20],
        Array(primitive_integers),
    )
    for (distribution, values, raw) in zip(CUDA_FIXED_DISTRIBUTIONS, observed, primitive)
        expected = map(value -> _distribution_oracle(distribution, value), raw)
        @test isequal(values, expected)
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
        raw = Array(_primitive_array(gpu_rng, distribution, 19))
        expected = map(value -> _distribution_oracle(distribution, value), raw)
        @test isequal(Array(rand(gpu_rng, distribution, 19)), expected)
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

@testset "CUDA fixed-distribution validation order" begin
    extension = Base.get_extension(IR, :PureRNGsDistributionsExt)
    rng = device(Philox4x32(0x64c4))
    invalid = (
        (Normal(Inf, 1.0; check_args = false), "Normal"),
        (Uniform(1.0, 1.0; check_args = false), "Uniform"),
        (Distributions.Exponential(0.0; check_args = false), "Exponential"),
        (Bernoulli(NaN; check_args = false), "Bernoulli"),
        (DiscreteUniform(2, 1; check_args = false), "DiscreteUniform"),
    )

    for (distribution, name) in invalid
        T = extension._result_type(distribution)
        expected_error = "ArgumentError: invalid $name parameters"
        wrong_device = Vector{T}(undef, 0)

        @test_throws TypeError rand_next!(rng, distribution, wrong_device; threaded = 1)
        device_error = try
            rand_next!(rng, distribution, wrong_device)
            nothing
        catch caught
            caught
        end
        @test sprint(showerror, device_error) ==
              "ArgumentError: destination device differs from the generator device"

        empty = CUDA.CuArray{T}(undef, 0)
        caught = Ref{Any}()
        profile = CUDA.Profile.profile_internally(; concurrent = false, trace = true) do
            caught[] = try
                rand_next!(rng, distribution, empty)
                nothing
            catch error
                error
            end
            CUDA.synchronize()
        end
        events = _cuda_profile_events(profile)
        @test sprint(showerror, caught[]) == expected_error
        @test isempty(events.kernels)
        @test isempty(events.copies)
        @test isempty(events.memsets)

        sentinel = T === Bool ? true : T(-1)
        destination = CUDA.fill(sentinel, 8)
        mutation_error = try
            rand_next!(rng, distribution, destination)
            nothing
        catch caught
            caught
        end
        @test sprint(showerror, mutation_error) == expected_error
        @test Array(destination) == fill(sentinel, 8)

        size_error = try
            rand(rng, distribution, -1)
            nothing
        catch caught
            caught
        end
        @test sprint(showerror, size_error) == expected_error
    end

    allocating_forms = (
        () -> rand(rng, Float32, -1),
        () -> rand_next(rng, Float32, -1),
        () -> randn(rng, Float32, -1),
        () -> randn_next(rng, Float32, -1),
        () -> randexp(rng, Float32, -1),
        () -> randexp_next(rng, Float32, -1),
        () -> rand(rng, UInt32(3):UInt32(7), -1),
        () -> rand_next(rng, UInt32(3):UInt32(7), -1),
        () -> rand(rng, Normal(), -1),
        () -> rand_next(rng, Normal(), -1),
    )
    for draw in allocating_forms
        @test_throws ArgumentError draw()
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
        events = _cuda_profile_events(profile)
        @test !isempty(events.kernels)
        @test isempty(events.host_to_device)
        @test isempty(events.device_to_host)
        @test _device_id(destination) == CUDA.deviceid(primary)
    end
end
