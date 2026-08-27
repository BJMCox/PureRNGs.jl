using MLDataDevices
using Random

const StatefulIR = PureRNGs

struct BridgeVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.size(vector::BridgeVector) = size(vector.data)
Base.getindex(vector::BridgeVector, index::Int) = vector.data[index]
Base.setindex!(vector::BridgeVector, value, index::Int) =
    setindex!(vector.data, value, index)

function _bridge_last(rng, width::UInt16)
    bit = StatefulIR._block_bits(rng) - width
    position = if rng.position isa StatefulIR._Position64
        StatefulIR._Position64(StatefulIR._max_block(rng), bit)
    else
        StatefulIR._Position128(typemax(UInt64), typemax(UInt64), bit)
    end
    return StatefulIR._rebuild(rng, position, rng.device)
end

function _bridge_methods(function_)
    return Set(filter(methods(function_)) do method
        signature = Base.unwrap_unionall(method.sig)
        length(signature.parameters) >= 2 || return false
        owner = signature.parameters[2]
        owner isa Type || owner isa UnionAll || return false
        return owner <: StatefulRNG
    end)
end

function _bridge_allocations()
    mutable_rng = StatefulRNG(Philox4x32(0x800))
    destination = Vector{UInt64}(undef, 32)
    rand(mutable_rng, UInt64)
    randn(mutable_rng, Float64)
    rand(mutable_rng, UInt16(1):UInt16(17))
    rand!(mutable_rng, destination)
    return (
        @allocated(rand(mutable_rng, UInt64)),
        @allocated(randn(mutable_rng, Float64)),
        @allocated(rand(mutable_rng, UInt16(1):UInt16(17))),
        @allocated(rand!(mutable_rng, destination)),
    )
end

function _parent_allocation(mutable_rng)
    parent(mutable_rng)
    return @allocated parent(mutable_rng)
end

@testset "R32-R35 StatefulRNG scalar bridge" begin
    source = StatefulIR._reserve(Philox4x32(0x801), UInt64(17), UInt64(0))
    device_source = MLDataDevices.CUDADevice(:discarded)(source)
    mutable_rng = StatefulRNG(device_source)

    @test supertype(typeof(mutable_rng)) === Random.AbstractRNG
    @test ismutabletype(typeof(mutable_rng))
    @test fieldcount(typeof(mutable_rng)) == 1
    @test isconcretetype(fieldtype(typeof(mutable_rng), 1))
    @test mutable_rng.rng.key == source.key
    @test mutable_rng.rng.position == source.position
    @test mutable_rng.rng.device === StatefulIR._CPU_BACKEND

    @test @inferred(parent(mutable_rng)) === mutable_rng.rng
    @test _parent_allocation(mutable_rng) == 0

    for F in FAMILY_TYPES
        cursor = F(0x802)
        mutable_rng = StatefulRNG(cursor)
        for T in SCALAR_UNIFORM_TYPES
            cursor, expected = rand_next(cursor, T)
            @test rand(mutable_rng, T) === expected
            @test mutable_rng.rng === cursor
        end
        for T in NORMAL_TYPES
            cursor, expected = randn_next(cursor, T)
            @test randn(mutable_rng, T) === expected
            @test mutable_rng.rng === cursor
        end
        for T in RANGE_INTS
            range = T(2):T(3):T(20)
            cursor, expected = rand_next(cursor, range)
            @test rand(mutable_rng, range) === expected
            @test mutable_rng.rng === cursor
        end
    end

    root = Philox4x32(0x803)
    mutable_rng = StatefulRNG(root)
    next_rng, expected = rand_next(root, Float64)
    @test rand(mutable_rng) === expected
    @test mutable_rng.rng === next_rng
    normal_next, normal_expected = randn_next(next_rng, Float64)
    @test randn(mutable_rng) === normal_expected
    @test mutable_rng.rng === normal_next
    @test _bridge_allocations() == (0, 0, 0, 0)
end

@testset "R35 StatefulRNG parent" begin
    root = Philox4x32(0x812)
    rebound = StatefulRNG(MLDataDevices.CUDADevice(:discarded)(root))
    @test parent(rebound) === root

    first_bridge = StatefulRNG(root)
    second_bridge = StatefulRNG(root)
    @test rand(first_bridge, UInt64) === rand(second_bridge, UInt64)
    @test parent(first_bridge) === parent(second_bridge)

    snapshot = parent(first_bridge)
    replay = copy(first_bridge)
    @test parent(replay) === snapshot
    rand(replay, UInt32)
    @test parent(first_bridge) === snapshot
    @test parent(replay) !== snapshot

    @test Random.seed!(first_bridge, 0x813) === first_bridge
    @test parent(first_bridge) === Philox4x32(0x813)

    before_failure = parent(first_bridge)
    @test_throws ArgumentError Random.seed!(first_bridge, -1)
    @test parent(first_bridge) === before_failure

    exhausted = StatefulIR._reserve(
        _bridge_last(Philox2x32(0x814), UInt16(1)),
        UInt64(1),
        UInt64(0),
    )
    exhausted_bridge = StatefulRNG(exhausted)
    @test_throws ArgumentError rand(exhausted_bridge, Bool)
    @test parent(exhausted_bridge) === exhausted
end

@testset "R34 range sampler dispatch" begin
    mutable_rng = StatefulRNG(Philox4x32(0x804))
    unit = UInt16(2):UInt16(17)
    stepped = UInt16(2):UInt16(3):UInt16(17)

    unit_sampler = Random.Sampler(typeof(mutable_rng), unit, Val(1))
    stepped_sampler = Random.Sampler(typeof(mutable_rng), stepped, Val(1))
    @test unit_sampler isa StatefulIR._StatefulRangeSampler
    @test stepped_sampler isa StatefulIR._StatefulRangeSampler
    @test which(Random.Sampler, Tuple{Type{typeof(mutable_rng)},typeof(unit),Val{1}}).module ===
          StatefulIR
    @test which(Random.Sampler, Tuple{Type{typeof(mutable_rng)},typeof(stepped),Val{1}}).module ===
          StatefulIR

    for range in (unit, stepped)
        root = Philox4x32(0x805)
        expected_rng, expected = rand_next(root, range)
        mutable_rng = StatefulRNG(root)
        @test rand(mutable_rng, range) === expected
        @test mutable_rng.rng === expected_rng
        @test rand(StatefulRNG(root), range, 4) == last(rand_next(root, range, 4))
    end
end

@testset "R34 and R54 owned bridge fills" begin
    for T in SCALAR_UNIFORM_TYPES
        root = Philox4x32(0x806)
        expected_rng, expected = rand_next(root, T, 19)
        destination = Vector{T}(undef, 19)
        mutable_rng = StatefulRNG(root)
        @test rand!(mutable_rng, destination) === destination
        @test destination == expected
        @test mutable_rng.rng === expected_rng
    end

    for T in NORMAL_TYPES
        root = Philox4x32(0x807)
        expected_rng, expected = randn_next(root, T, 19)
        destination = Vector{T}(undef, 19)
        mutable_rng = StatefulRNG(root)
        @test randn!(mutable_rng, destination) === destination
        @test destination == expected
        @test mutable_rng.rng === expected_rng
    end

    root = Philox4x32(0x808)
    expected_rng, expected = rand_next(root, Bool, 67)
    bits = BitArray(undef, 67)
    mutable_rng = StatefulRNG(root)
    @test rand!(mutable_rng, bits) === bits
    @test collect(bits) == expected
    @test mutable_rng.rng === expected_rng

    uniform_last = _bridge_last(Philox2x32(0x809), UInt16(64))
    uniform_destination = fill(UInt64(0xdeadbeef), 2)
    uniform_mutable = StatefulRNG(uniform_last)
    @test_throws ArgumentError rand!(uniform_mutable, uniform_destination)
    @test uniform_destination == fill(UInt64(0xdeadbeef), 2)
    @test uniform_mutable.rng === uniform_last

    bool_last = _bridge_last(Philox2x32(0x80a), UInt16(1))
    bit_destination = trues(2)
    bool_mutable = StatefulRNG(bool_last)
    @test_throws ArgumentError rand!(bool_mutable, bit_destination)
    @test bit_destination == trues(2)
    @test bool_mutable.rng === bool_last

    normal_last = _bridge_last(Philox2x32(0x80b), UInt16(52))
    normal_destination = fill(1.0, 2)
    normal_mutable = StatefulRNG(normal_last)
    @test_throws ArgumentError randn!(normal_mutable, normal_destination)
    @test normal_destination == fill(1.0, 2)
    @test normal_mutable.rng === normal_last

    exhausted = StatefulIR._reserve(bool_last, UInt64(1), UInt64(0))
    exhausted_mutable = StatefulRNG(exhausted)
    @test rand!(exhausted_mutable, UInt32[]) == UInt32[]
    @test rand!(exhausted_mutable, BitArray(undef, 0)) == BitArray(undef, 0)
    @test randn!(exhausted_mutable, Float32[]) == Float32[]
    @test exhausted_mutable.rng === exhausted
    @test_throws ArgumentError rand(exhausted_mutable, Bool)
    @test exhausted_mutable.rng === exhausted
end

@testset "R34 and R54 foreign bridge fills" begin
    uniform_root = _bridge_last(Philox2x32(0x80c), UInt16(64))
    expected_uniform, first_uniform = rand_next(uniform_root, UInt64)
    uniform_destination = BridgeVector(fill(UInt64(0xdeadbeef), 2))
    uniform_mutable = StatefulRNG(uniform_root)
    @test_throws ArgumentError rand!(uniform_mutable, uniform_destination)
    @test uniform_destination.data == [first_uniform, UInt64(0xdeadbeef)]
    @test uniform_mutable.rng === expected_uniform

    normal_root = _bridge_last(Philox2x32(0x80d), UInt16(52))
    expected_normal, first_normal = randn_next(normal_root, Float64)
    normal_destination = BridgeVector(fill(1.0, 2))
    normal_mutable = StatefulRNG(normal_root)
    @test_throws ArgumentError randn!(normal_mutable, normal_destination)
    @test normal_destination.data == [first_normal, 1.0]
    @test normal_mutable.rng === expected_normal
end

@testset "R34 seed and R35 copy" begin
    for F in FAMILY_TYPES
        mutable_rng = StatefulRNG(F(0x80e))
        rand(mutable_rng, UInt32)
        @test Random.seed!(mutable_rng, 0x80f) === mutable_rng
        @test mutable_rng.rng === F(0x80f)
    end

    mutable_rng = StatefulRNG(Philox2x32(0x810))
    rand(mutable_rng, UInt32)
    before = mutable_rng.rng
    @test_throws ArgumentError Random.seed!(mutable_rng, -1)
    @test mutable_rng.rng === before
    @test_throws ArgumentError Random.seed!(mutable_rng, big(1) << 32)
    @test mutable_rng.rng === before

    replay = copy(mutable_rng)
    @test replay !== mutable_rng
    @test replay.rng === mutable_rng.rng
    @test rand(replay, UInt64) === rand(mutable_rng, UInt64)
    @test replay.rng === mutable_rng.rng
end

@testset "R34 and R52 closed bridge method surface" begin
    mutable_rng = StatefulRNG(Philox4x32(0x811))
    M = typeof(mutable_rng)
    required = Dict(
        function_ => Set{Method}() for function_ in (
            Random.rand,
            Random.rand!,
            Random.randn,
            Random.randn!,
            Random.seed!,
            copy,
            parent,
        )
    )
    require = function (function_, signature)
        method = which(function_, signature)
        @test method.module === StatefulIR
        push!(required[function_], method)
        return nothing
    end

    for sampler in (
        Random.SamplerType{Bool},
        Random.SamplerType{UInt32},
        Random.SamplerType{UInt64},
        Random.SamplerTrivial{Random.CloseOpen01{Float32}},
        Random.SamplerTrivial{Random.CloseOpen01{Float64}},
    )
        require(Random.rand, Tuple{M,sampler})
    end
    require(
        Random.rand,
        Tuple{M,StatefulIR._StatefulRangeSampler{UInt16,typeof(UInt16(1):UInt16(2))}},
    )
    require(Random.randn, Tuple{M})
    require(Random.randn, Tuple{M,Type{Float32}})
    require(Random.randn, Tuple{M,Type{Float64}})
    for T in SCALAR_UNIFORM_TYPES
        require(Random.rand!, Tuple{M,Vector{T}})
    end
    require(Random.rand!, Tuple{M,BitArray})
    for T in NORMAL_TYPES
        require(Random.randn!, Tuple{M,Vector{T}})
    end
    require(Random.seed!, Tuple{M,Int})
    require(copy, Tuple{M})
    require(parent, Tuple{M})

    for (function_, methods_) in required
        @test _bridge_methods(function_) == methods_
    end

    sampler_methods =
        Set(method for method in methods(Random.Sampler) if method.module === StatefulIR)
    unit = UInt16(1):UInt16(2)
    stepped = UInt16(1):UInt16(2):UInt16(5)
    required_samplers = Set((
        which(Random.Sampler, Tuple{Type{M},typeof(unit),Val{1}}),
        which(Random.Sampler, Tuple{Type{M},typeof(stepped),Val{1}}),
    ))
    @test sampler_methods == required_samplers

    ambiguities =
        filter(Test.detect_ambiguities(StatefulIR, Random; recursive = true)) do pair
            any(method -> method.module === StatefulIR, pair)
        end
    @test isempty(ambiguities)

    docs = string(Base.Docs.meta(StatefulIR)[Base.Docs.Binding(StatefulIR, :StatefulRNG)])
    @test occursin("parent(bridge)", docs)
    @test occursin("partially written", docs)
    @test occursin("counter exhaustion", docs)
end
