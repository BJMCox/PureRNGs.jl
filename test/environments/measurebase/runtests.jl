using PureRNGs
using MeasureBase
using Random
using Test

const ROOT = Philox4x32(12345)
const STANDARD_MEASURES = (StdUniform(), StdNormal(), StdLogistic(), StdExponential())

function primitive_draw(rng, ::Type{T}, ::StdUniform) where {T}
    return rand(rng, T)
end

function primitive_draw(rng, ::Type{T}, ::StdNormal) where {T}
    return randn(rng, T)
end

function primitive_draw(rng, ::Type{T}, ::StdLogistic) where {T}
    return MeasureBase.logit(rand(rng, T))
end

function primitive_draw(rng, ::Type{T}, ::StdExponential) where {T}
    return randexp(rng, T)
end

function check_replay(::Type{T}, measure) where {T}
    actual_rng = StatefulRNG(ROOT)
    replay_rng = StatefulRNG(ROOT)
    expected_rng = StatefulRNG(ROOT)

    actual = rand(actual_rng, T, measure)
    replay = rand(replay_rng, T, measure)
    expected = primitive_draw(expected_rng, T, measure)

    @test isequal(actual, replay)
    @test isequal(actual, expected)
    @test parent(actual_rng) === parent(replay_rng)
    @test parent(actual_rng) === parent(expected_rng)
    @test parent(actual_rng) !== ROOT
end

@testset "R66 MeasureBase environment" begin
    @test Base.pkgversion(MeasureBase) == v"0.14.13"

    @testset "standard measures" begin
        for T in (Float32, Float64), measure in STANDARD_MEASURES
            check_replay(T, measure)
        end
    end

    @testset "power measure" begin
        for T in (Float32, Float64)
            actual_rng = StatefulRNG(ROOT)
            replay_rng = StatefulRNG(ROOT)
            expected_rng = StatefulRNG(ROOT)
            measure = StdNormal()^4

            actual = rand(actual_rng, T, measure)
            replay = rand(replay_rng, T, measure)
            expected = [randn(expected_rng, T) for _ = 1:4]

            @test isequal(actual, replay)
            @test isequal(actual, expected)
            @test parent(actual_rng) === parent(replay_rng)
            @test parent(actual_rng) === parent(expected_rng)
        end
    end

    @testset "product measure" begin
        for T in (Float32, Float64)
            actual_rng = StatefulRNG(ROOT)
            replay_rng = StatefulRNG(ROOT)
            expected_rng = StatefulRNG(ROOT)
            measure = productmeasure(STANDARD_MEASURES)

            actual = rand(actual_rng, T, measure)
            replay = rand(replay_rng, T, measure)
            expected = map(STANDARD_MEASURES) do component
                primitive_draw(expected_rng, T, component)
            end

            @test isequal(actual, replay)
            @test isequal(actual, expected)
            @test parent(actual_rng) === parent(replay_rng)
            @test parent(actual_rng) === parent(expected_rng)
        end
    end

    @testset "bridge lifecycle" begin
        bridge = StatefulRNG(ROOT)
        rand(bridge, Float64, StdUniform())
        replay = copy(bridge)
        @test replay !== bridge
        @test parent(replay) === parent(bridge)
        @test rand(replay, Float64, StdNormal()) === rand(bridge, Float64, StdNormal())
        @test parent(replay) === parent(bridge)

        @test Random.seed!(bridge, 12345) === bridge
        @test parent(bridge) === ROOT
    end

    @testset "loaded-package ambiguities" begin
        ambiguities =
            Test.detect_ambiguities(PureRNGs, Random, MeasureBase; recursive = true)
        package_ambiguities = filter(ambiguities) do pair
            any(pair) do method
                method.module === PureRNGs ||
                    parentmodule(method.module) === PureRNGs
            end
        end
        @test isempty(package_ambiguities)
    end
end
