include(joinpath(@__DIR__, "..", "..", "..", "distribution_transform_cases.jl"))

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
const CUDA_EXPANDED_CONTINUOUS_DISTRIBUTIONS = (
    LogNormal{Float32}(0.25f0, 0.75f0),
    LogNormal{Float64}(0.25, 0.75),
    Weibull{Float32}(1.75f0, 0.75f0),
    Weibull{Float64}(1.75, 0.75),
    Rayleigh{Float32}(0.75f0),
    Rayleigh{Float64}(0.75),
    Laplace{Float32}(0.25f0, 0.75f0),
    Laplace{Float64}(0.25, 0.75),
    six_transform_distributions(Float32)...,
    six_transform_distributions(Float64)...,
)

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

@inline _primitive_at(rng, ::Normal{T}, index) where {T} = randn_at(rng, T, index)
@inline _primitive_at(rng, ::Uniform{T}, index) where {T} = rand_at(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Exponential{T}, index) where {T} =
    randexp_at(rng, T, index)
@inline _primitive_at(rng, ::Bernoulli{T}, index) where {T} = rand_at(rng, T, index)
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
    continued, next_rng = rand_next(rng, distribution)
    @inbounds begin
        destination[offset+1] = rand(rng, distribution)
        destination[offset+2] = continued
        destination[offset+3] = rand_at(rng, distribution, 1)
        destination[offset+4] = rand(next_rng, distribution)
        destination[offset+5] = rand_at(rng, distribution, 2)
    end
    return nothing
end

@inline function _write_primitive_probe!(destination, offset, rng, distribution)
    continued, next_rng = _primitive_next(rng, distribution)
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

@inline _expanded_result_type(::LogNormal{T}) where {T} = T
@inline _expanded_result_type(::Weibull{T}) where {T} = T
@inline _expanded_result_type(::Rayleigh{T}) where {T} = T
@inline _expanded_result_type(::Laplace{T}) where {T} = T
@inline _expanded_result_type(d::SIX_TRANSFORM_TYPES) = six_transform_result_type(d)
@inline _expanded_primitive_next(rng, d::SIX_TRANSFORM_TYPES) =
    six_transform_input_next(rng, d)
@inline _expanded_formula(d::SIX_TRANSFORM_TYPES, primitive) =
    six_transform_formula(d, primitive)

@inline _expanded_primitive_next(rng, ::LogNormal{T}) where {T} = randn_next(rng, T)
@inline _expanded_primitive_next(rng, ::Weibull{T}) where {T} = randexp_next(rng, T)
@inline _expanded_primitive_next(rng, ::Rayleigh{T}) where {T} = randexp_next(rng, T)
@inline function _expanded_primitive_next(rng, ::Laplace{T}) where {T}
    magnitude, after_magnitude = randexp_next(rng, T)
    positive, next_rng = rand_next(after_magnitude, Bool)
    return (magnitude, positive), next_rng
end

@inline _expanded_formula(d::LogNormal, primitive) = exp(fma(d.σ, primitive, d.μ))
@inline _expanded_formula(d::Weibull, primitive) = d.θ * primitive^inv(d.α)
@inline _expanded_formula(d::Rayleigh{T}, primitive) where {T} =
    d.σ * sqrt(T(2) * primitive)
@inline _expanded_formula(d::Laplace, primitive) =
    fma(ifelse(primitive[2], d.θ, -d.θ), primitive[1], d.μ)

function _expanded_chain(rng, distribution, count)
    values = Vector{_expanded_result_type(distribution)}(undef, count)
    cursor = rng
    for index in eachindex(values)
        primitive, cursor = _expanded_primitive_next(cursor, distribution)
        values[index] = _expanded_formula(distribution, primitive)
    end
    return cursor, values
end

function _expanded_formula_kernel!(destination, rng, distribution)
    if CUDA.threadIdx().x == 1
        cursor = rng
        @inbounds for index in eachindex(destination)
            primitive, cursor = _expanded_primitive_next(cursor, distribution)
            destination[index] = _expanded_formula(distribution, primitive)
        end
    end
    return nothing
end

function _expanded_scalar_probe_kernel!(values, expected, rng, distribution)
    if CUDA.threadIdx().x == 1
        first_primitive, after_first = _expanded_primitive_next(rng, distribution)
        second_primitive, _ = _expanded_primitive_next(after_first, distribution)
        continued, next_rng = rand_next(rng, distribution)
        first_value = _expanded_formula(distribution, first_primitive)
        second_value = _expanded_formula(distribution, second_primitive)
        @inbounds begin
            values[1] = rand(rng, distribution)
            values[2] = continued
            values[3] = rand_at(rng, distribution, 1)
            values[4] = rand(next_rng, distribution)
            values[5] = rand_at(rng, distribution, 2)
            expected[1] = first_value
            expected[2] = first_value
            expected[3] = first_value
            expected[4] = second_value
            expected[5] = second_value
        end
    end
    return nothing
end

function _expanded_addressed_kernel!(destination, rng, distribution)
    if CUDA.threadIdx().x == 1
        @inbounds for index in eachindex(destination)
            destination[index] = rand_at(rng, distribution, index)
        end
    end
    return nothing
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

@testset "CUDA expanded continuous mapped forms" begin
    rng = device(Philox4x32(0x64c6))
    for distribution in CUDA_EXPANDED_CONTINUOUS_DISTRIBUTIONS
        @testset "$distribution" begin
            T = _expanded_result_type(distribution)
            scalar_values = CUDA.CuArray{T}(undef, 5)
            scalar_expected = similar(scalar_values)
            formula_values = CUDA.CuArray{T}(undef, 2048)
            addressed = similar(formula_values)

            CUDA.@sync CUDA.@cuda threads = 1 blocks = 1 _expanded_scalar_probe_kernel!(
                scalar_values,
                scalar_expected,
                rng,
                distribution,
            )
            @test isequal(Array(scalar_values), Array(scalar_expected))

            CUDA.@sync CUDA.@cuda threads = 1 blocks = 1 _expanded_formula_kernel!(
                formula_values,
                rng,
                distribution,
            )
            CUDA.@sync CUDA.@cuda threads = 1 blocks = 1 _expanded_addressed_kernel!(
                addressed,
                rng,
                distribution,
            )
            @test isequal(Array(addressed), Array(formula_values))

            pure_values = rand(rng, distribution, length(formula_values))
            values, next_rng = rand_next(rng, distribution, length(formula_values))
            expected_next, _ = _expanded_chain(rng, distribution, length(formula_values))
            @test isequal(Array(pure_values), Array(formula_values))
            @test isequal(Array(values), Array(formula_values))
            @test next_rng.position == expected_next.position
            @test next_rng.device == expected_next.device

            destination = similar(formula_values)
            @test rand!(rng, distribution, destination) === destination
            @test isequal(Array(destination), Array(formula_values))

            returned, fill_next = rand_next!(rng, distribution, destination)
            @test returned === destination
            @test isequal(Array(destination), Array(formula_values))
            @test fill_next.position == expected_next.position
            @test fill_next.device == expected_next.device
        end
    end
end

@testset "CUDA Categorical allocating and fill forms" begin
    rng = device(Philox4x32(0x64c7))
    probabilities = CUDA.CuArray(Float64[0, 1, 0, 3])
    distribution = Categorical(probabilities; check_args = false)
    expected, expected_next =
        randsample_next(rng, 1:length(probabilities), probabilities, 33)

    values = rand(rng, distribution, 33)
    @test values isa CUDA.CuArray{Int,1}
    @test Array(values) == Array(expected)

    continued, continued_next = rand_next(rng, distribution, 33)
    @test Array(continued) == Array(expected)
    @test continued_next.position == expected_next.position
    @test continued_next.device == expected_next.device

    destination = similar(values)
    @test rand!(rng, distribution, destination) === destination
    @test Array(destination) == Array(expected)
    returned, filled_next = rand_next!(rng, distribution, destination)
    @test returned === destination
    @test Array(destination) == Array(expected)
    @test filled_next.position == expected_next.position
    @test filled_next.device == expected_next.device

    cpu_probabilities = Categorical(Float64[0, 1]; check_args = false)
    @test_throws ArgumentError rand(rng, cpu_probabilities, 0)
    @test_throws ArgumentError rand!(rng, cpu_probabilities, CUDA.CuArray{Int}(undef, 0))
end

@testset "CUDA fixed-distribution arrays and fills" begin
    extension = Base.get_extension(IR, :PureRNGsDistributionsExt)
    # One exact distribution per primitive draw type for the other generators.
    exact_distributions = (
        CUDA_FIXED_DISTRIBUTIONS[4],
        CUDA_FIXED_DISTRIBUTIONS[6],
        CUDA_FIXED_DISTRIBUTIONS[9],
    )
    cases = (
        ((Philox4x32, distribution) for distribution in CUDA_FIXED_DISTRIBUTIONS)...,
        (
            (F, distribution) for F in GENERATOR_TYPES for
            distribution in exact_distributions if F !== Philox4x32
        )...,
    )
    for (F, distribution) in cases
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
        if !exact
            raw = Array(_primitive_array(gpu_rng, distribution, 19))
            expected = map(value -> _distribution_oracle(distribution, value), raw)
            @test isequal(Array(rand(gpu_rng, distribution, 19)), expected)
        end
    end
end

@testset "CUDA fixed packed and offset fills agree" begin
    for F in GENERATOR_TYPES, T in (Float32, Float64)
        rng = device(F(0x64c5))
        count = T === Float32 ? 2048 : 1024
        aligned = CUDA.CuArray{T}(undef, count)
        offset_storage = CUDA.CuArray{T}(undef, count + 1)
        offset = @view offset_storage[2:end]

        for distribution in (Uniform(T(-1.5), T(2.75)), Distributions.Exponential(T(1.5)))
            _, aligned_next = rand_next!(rng, distribution, aligned)
            _, offset_next = rand_next!(rng, distribution, offset)
            @test isequal(Array(aligned), Array(offset))
            @test aligned_next.position == offset_next.position
        end
    end
end

@testset "CUDA fixed-distribution capacity and empty fills" begin
    extension = Base.get_extension(IR, :PureRNGsDistributionsExt)
    rng = device(Philox4x32(0x64c2))
    for distribution in CUDA_FIXED_DISTRIBUTIONS
        T = extension._result_type(distribution)
        width = extension._distribution_span(distribution)
        last_rng = _last_draw_rng(rng, width)
        _, terminal = rand_next(last_rng, distribution)
        @test terminal.position == _terminal(rng)

        values, array_terminal = rand_next(last_rng, distribution, 1)
        destination = similar(values)
        returned, fill_terminal = rand_next!(last_rng, distribution, destination)
        @test returned === destination
        @test isequal(Array(destination), Array(values))
        @test fill_terminal.position == array_terminal.position == terminal.position

        sentinel = T === Bool ? true : T(-1)
        failed = CUDA.fill(sentinel, 2)
        @test_throws StreamExhausted rand_next!(last_rng, distribution, failed)
        @test Array(failed) == fill(sentinel, 2)
        @test_throws StreamExhausted rand(terminal, distribution)

        empty = CUDA.CuArray{T}(undef, 0)
        profile = CUDA.@profile raw = true rand_next!(terminal, distribution, empty)
        @test count(value -> !ismissing(value), profile.device.grid) == 0
        returned_empty, empty_next = rand_next!(terminal, distribution, empty)
        @test returned_empty === empty
        @test empty_next.position == terminal.position
    end
end

@testset "CUDA fixed-distribution validation order" begin
    extension = Base.get_extension(IR, :PureRNGsDistributionsExt)
    rng = device(Philox4x32(0x64c4))
    invalid = (
        Normal(Inf, 1.0; check_args = false),
        Uniform(1.0, 1.0; check_args = false),
        Distributions.Exponential(0.0; check_args = false),
        Bernoulli(NaN; check_args = false),
        DiscreteUniform(2, 1; check_args = false),
    )

    for distribution in invalid
        T = extension._result_type(distribution)
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
        @test caught[] isa ArgumentError
        @test isempty(events.kernels)
        @test isempty(events.copies)
        @test isempty(events.memsets)
    end

    distribution = first(invalid)
    T = extension._result_type(distribution)
    wrong_device = Vector{T}(undef, 0)
    @test_throws ArgumentError rand_next!(rng, distribution, wrong_device)

    sentinel = T === Bool ? true : T(-1)
    destination = CUDA.fill(sentinel, 8)
    @test_throws ArgumentError rand_next!(rng, distribution, destination)
    @test Array(destination) == fill(sentinel, 8)

    @test_throws ArgumentError rand(rng, distribution, -1)

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

# A device Gamma fill equals the device's addressed draws. Device normals differ
# from CPU normals in the last ulp, which moves values by rounding and flips an
# acceptance decision only at the boundary, so nearly all values match the CPU.
@testset "CUDA Gamma fills equal device draws and track the CPU" begin
    for F in GENERATOR_TYPES, d in (Gamma(2.5, 2.0), Gamma(0.3f0, 1.0f0))
        T = partype(d)
        cpu_rng = F(0x792, 1)
        gpu_rng = device(cpu_rng)
        values, next_rng = rand_next(gpu_rng, d, 1000)
        @test values isa CuArray{T,1}
        @test next_rng.position == last(rand_next(cpu_rng, d, 1000)).position
        addressed = CuArray{T}(undef, 1000)
        CUDA.@sync CUDA.@cuda threads = 1 _expanded_addressed_kernel!(addressed, gpu_rng, d)
        @test Array(values) == Array(addressed)
        host = rand(cpu_rng, d, 1000)
        @test count(isapprox.(Array(values), host; rtol = 100eps(T))) >= 998
        # One candidate sends about 5% of shape-one draws down the child stream.
        codec = IR._GammaCodec(T(1), T(1), gpu_rng.device, 1)
        forced = CuArray{T}(undef, 10_000)
        IR._fill_prevalidated!(gpu_rng, forced, false, codec)
        host_forced = zeros(T, 10_000)
        IR._fill_prevalidated!(
            cpu_rng,
            host_forced,
            false,
            IR._GammaCodec(T(1), T(1), cpu_rng.device, 1),
        )
        @test count(isapprox.(Array(forced), host_forced; rtol = 100eps(T))) >= 9_980
    end
end

# Beta subtracts two log-gammas and exponentiates, so device rounding reaches
# further than for Gamma; the tolerance allows it.
@testset "CUDA Gamma family fills equal device draws and track the CPU" begin
    for F in GENERATOR_TYPES,
        d in (Chisq(3.0), InverseGamma(2.5f0, 1.5f0), Beta(0.3, 0.4), TDist(3.0f0))

        T = partype(d)
        cpu_rng = F(0x793, 1)
        gpu_rng = device(cpu_rng)
        values = rand(gpu_rng, d, 1000)
        @test values isa CuArray{T,1}
        addressed = CuArray{T}(undef, 1000)
        CUDA.@sync CUDA.@cuda threads = 1 _expanded_addressed_kernel!(addressed, gpu_rng, d)
        @test Array(values) == Array(addressed)
        @test count(isapprox.(Array(values), rand(cpu_rng, d, 1000); rtol = 1000eps(T))) >=
              998
    end
end

# A Dirichlet or MvNormal draw fills a column per draw on the device. Device
# rounding can flip a Gamma acceptance at the boundary, so nearly all values
# match the CPU, and a fill returns nothing to the host.
@testset "CUDA Dirichlet and MvNormal draws track the CPU" begin
    PDMats = Distributions.PDMats
    covariance = [2.0 0.5 0.1; 0.5 1.0 0.2; 0.1 0.2 3.0]
    for F in GENERATOR_TYPES,
        d in (
            Dirichlet([0.3, 2.0, 5.0]),
            Dirichlet(fill(0.05f0, 7)),
            MvNormal([1.0, 2.0, 3.0], covariance),
            MvNormal(Float32[1, 2], PDMats.PDiagMat(Float32[2, 3])),
            MvNormal(zeros(2), PDMats.ScalMat(2, 4.0)),
        )

        T = eltype(d)
        cpu_rng = F(0x794, 1)
        gpu_rng = device(cpu_rng)
        values, next_rng = rand_next(gpu_rng, d, 1000)
        expected, expected_next = rand_next(cpu_rng, d, 1000)
        @test values isa CuArray{T,2}
        @test next_rng.position == expected_next.position
        @test count(
            isapprox.(Array(values), expected; rtol = 1000eps(T), atol = 10eps(T)),
        ) >= 0.998 * length(expected)
        @test Array(rand_at(gpu_rng, d, 7)) ≈ rand_at(cpu_rng, d, 7)
        destination = CuArray{T}(undef, length(d), 5)
        rand!(gpu_rng, d, destination)
        @test Array(destination) ≈ rand(cpu_rng, d, 5)
        @test isempty(_device_events(() -> rand(gpu_rng, d, 1000)).device_to_host)
    end
end

