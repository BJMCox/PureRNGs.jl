using AbstractMCMC
using Distributions
using PureRNGs
using Random
using Test
using Turing

const ROOT = Philox4x32(12345)
const CHAINS = 4
const SAMPLERS =
    (Prior = (() -> Prior(), 32), MH = (() -> MH(), 32), NUTS = (() -> NUTS(16, 0.65), 16))

@model function gaussian_model(y)
    μ ~ Normal(0.0, 1.0)
    y ~ Normal(μ, 1.0)
end

mutable struct StatefulCallback
    calls::Threads.Atomic{Int}
end

StatefulCallback() = StatefulCallback(Threads.Atomic{Int}(0))

function (callback::StatefulCallback)(
    rng,
    model,
    sampler,
    transition,
    state,
    iteration;
    kwargs...,
)
    rng isa StatefulRNG || error("Turing callback received $(typeof(rng))")
    Threads.atomic_add!(callback.calls, 1)
    return nothing
end

function sample_once(ensemble, sampler_factory, count)
    rng = StatefulRNG(ROOT)
    callback = StatefulCallback()
    chain = Turing.sample(
        rng,
        gaussian_model(0.25),
        sampler_factory(),
        ensemble,
        count,
        CHAINS;
        callback,
        progress = false,
        verbose = false,
    )
    return Array(chain), parent(rng), callback.calls[]
end

function check_replay(ensemble, sampler_factory, count)
    first_values, first_parent, first_calls = sample_once(ensemble, sampler_factory, count)
    second_values, second_parent, second_calls =
        sample_once(ensemble, sampler_factory, count)

    @test size(first_values) == (count, CHAINS, 1)
    @test isequal(first_values, second_values)
    @test first_parent === second_parent
    @test first_parent !== ROOT
    @test first_calls > 0
    @test first_calls == second_calls
end

@testset "R66 Turing environment" begin
    @test Base.pkgversion(Turing) == v"0.46.1"
    @test Base.pkgversion(AbstractMCMC) == v"5.16.0"
    @test Base.pkgversion(Distributions) == v"0.25.131"
    @test Threads.nthreads() >= 4

    @testset "bridge lifecycle" begin
        bridge = StatefulRNG(ROOT)
        rand(bridge, UInt64)
        replay = copy(bridge)
        @test replay !== bridge
        @test parent(replay) === parent(bridge)
        @test rand(replay, UInt64) === rand(bridge, UInt64)
        @test parent(replay) === parent(bridge)

        @test Random.seed!(bridge, 12345) === bridge
        @test parent(bridge) === ROOT
    end

    @testset "serial replay" begin
        for (name, (sampler_factory, count)) in pairs(SAMPLERS)
            @testset "$name" begin
                check_replay(MCMCSerial(), sampler_factory, count)
            end
        end
    end

    @testset "threaded replay" begin
        for (name, (sampler_factory, count)) in pairs(SAMPLERS)
            @testset "$name" begin
                check_replay(MCMCThreads(), sampler_factory, count)
            end
        end
    end

    @testset "loaded-package ambiguities" begin
        ambiguities = Test.detect_ambiguities(
            PureRNGs,
            Random,
            Distributions,
            AbstractMCMC,
            Turing;
            recursive = true,
        )
        package_ambiguities = filter(ambiguities) do pair
            any(pair) do method
                method.module === PureRNGs ||
                    parentmodule(method.module) === PureRNGs
            end
        end
        @test isempty(package_ambiguities)
    end
end
