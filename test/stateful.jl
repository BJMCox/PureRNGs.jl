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
        return (owner isa Type || owner isa UnionAll) && owner <: StatefulRNG
    end)
end

function _bridge_allocations()
    mutable_rng = StatefulRNG(Philox4x32(0x800))
    destination = Vector{UInt64}(undef, 32)
    exponential_destination = Vector{Float64}(undef, 32)
    rand(mutable_rng, Int64)
    rand(mutable_rng, UInt64)
    randn(mutable_rng, Float64)
    randexp(mutable_rng, Float64)
    rand(mutable_rng, UInt16(1):UInt16(17))
    rand!(mutable_rng, destination)
    randexp!(mutable_rng, exponential_destination)
    return (
        @allocated(rand(mutable_rng, Int64)),
        @allocated(rand(mutable_rng, UInt64)),
        @allocated(randn(mutable_rng, Float64)),
        @allocated(randexp(mutable_rng, Float64)),
        @allocated(rand(mutable_rng, UInt16(1):UInt16(17))),
        @allocated(rand!(mutable_rng, destination)),
        @allocated(randexp!(mutable_rng, exponential_destination)),
    )
end

function _parent_allocations(mutable_rng)
    parent(mutable_rng)
    return @allocated parent(mutable_rng)
end

@testset "R32-R35 StatefulRNG scalar bridge" begin
    source = StatefulIR._reserve(Philox4x32(0x801), UInt64(17), UInt64(0))
    device_source = MLDataDevices.CUDADevice(:discarded)(source)
    mutable_rng = StatefulRNG(device_source)

    @test mutable_rng.rng.key == source.key
    @test mutable_rng.rng.position == source.position
    @test mutable_rng.rng.device === StatefulIR._CPU_BACKEND
    @test @inferred(parent(mutable_rng)) === mutable_rng.rng
    @test _parent_allocations(mutable_rng) == 0

    for F in (Philox2x32, Threefry4x64)
        cursor = F(0x802)
        mutable_rng = StatefulRNG(cursor)
        for T in PURE_UNIFORM_TYPES
            cursor, expected = rand_next(cursor, T)
            @test rand(mutable_rng, T) === expected
            @test mutable_rng.rng === cursor
        end
        for T in NORMAL_TYPES
            cursor, expected = randn_next(cursor, T)
            @test randn(mutable_rng, T) === expected
            @test mutable_rng.rng === cursor
        end
        for T in EXPONENTIAL_TYPES
            cursor, expected = randexp_next(cursor, T)
            @test randexp(mutable_rng, T) === expected
            @test mutable_rng.rng === cursor
        end
        for T in (UInt16,)
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
    exponential_next, exponential_expected = randexp_next(normal_next, Float64)
    @test randexp(mutable_rng) === exponential_expected
    @test mutable_rng.rng === exponential_next
    @test _bridge_allocations() == (0, 0, 0, 0, 0, 0, 0)
end

@testset "R35 StatefulRNG parent" begin
    root = Philox4x32(0x812)
    rebound = StatefulRNG(MLDataDevices.CUDADevice(:discarded)(root))
    @test parent(rebound) === root

    exhausted = StatefulIR._reserve(
        _bridge_last(Philox2x32(0x814), UInt16(1)),
        UInt64(1),
        UInt64(0),
    )
    exhausted_bridge = StatefulRNG(exhausted)
    @test_throws ArgumentError rand(exhausted_bridge, Bool)
    @test parent(exhausted_bridge) === exhausted
end

@testset "R33 bridge failure stability" begin
    for (T, width, draw) in (
        (Int32, UInt16(32), rand),
        (Int64, UInt16(64), rand),
        (Float32, UInt16(24), randexp),
        (Float64, UInt16(53), randexp),
    )
        last = _bridge_last(Philox2x32(0x818), width)
        expected_rng, expected = draw === rand ? rand_next(last, T) : randexp_next(last, T)
        mutable_rng = StatefulRNG(last)

        @test draw(mutable_rng, T) === expected
        @test parent(mutable_rng) === expected_rng
        @test_throws ArgumentError draw(mutable_rng, T)
        @test parent(mutable_rng) === expected_rng
    end
end

@testset "R34 range sampler dispatch" begin
    unit = UInt16(2):UInt16(17)
    stepped = UInt16(2):UInt16(3):UInt16(17)

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
    root = Philox4x32(0x806)
    expected_rng, expected = rand_next(root, UInt64, 19)
    destination = Vector{UInt64}(undef, 19)
    mutable_rng = StatefulRNG(root)
    @test rand!(mutable_rng, destination) === destination
    @test destination == expected
    @test mutable_rng.rng === expected_rng

    root = Philox4x32(0x815)
    expected_rng, expected = randexp_next(root, Float64, 19)
    destination = Vector{Float64}(undef, 19)
    mutable_rng = StatefulRNG(root)
    @test randexp!(mutable_rng, destination) === destination
    @test destination == expected
    @test mutable_rng.rng === expected_rng

    allocating_rng = StatefulRNG(root)
    @test randexp(allocating_rng, Float64, 19) == expected
    @test allocating_rng.rng === expected_rng

    root = Philox4x32(0x807)
    expected_rng, expected = randn_next(root, Float64, 19)
    destination = Vector{Float64}(undef, 19)
    mutable_rng = StatefulRNG(root)
    @test randn!(mutable_rng, destination) === destination
    @test destination == expected
    @test mutable_rng.rng === expected_rng

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

    exponential_last = _bridge_last(Philox2x32(0x816), UInt16(53))
    exponential_destination = fill(1.0, 2)
    exponential_mutable = StatefulRNG(exponential_last)
    @test_throws ArgumentError randexp!(exponential_mutable, exponential_destination)
    @test exponential_destination == fill(1.0, 2)
    @test exponential_mutable.rng === exponential_last

    exhausted = StatefulIR._reserve(bool_last, UInt64(1), UInt64(0))
    exhausted_mutable = StatefulRNG(exhausted)
    @test rand!(exhausted_mutable, UInt32[]) == UInt32[]
    @test rand!(exhausted_mutable, BitArray(undef, 0)) == BitArray(undef, 0)
    @test randn!(exhausted_mutable, Float32[]) == Float32[]
    @test randexp!(exhausted_mutable, Float32[]) == Float32[]
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

    exponential_root = _bridge_last(Philox2x32(0x817), UInt16(53))
    expected_exponential, first_exponential = randexp_next(exponential_root, Float64)
    exponential_destination = BridgeVector(fill(1.0, 2))
    exponential_mutable = StatefulRNG(exponential_root)
    @test_throws ArgumentError randexp!(exponential_mutable, exponential_destination)
    @test exponential_destination.data == [first_exponential, 1.0]
    @test exponential_mutable.rng === expected_exponential
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
    replay = copy(mutable_rng)
    @test replay !== mutable_rng
    @test replay.rng === mutable_rng.rng
    @test rand(replay, UInt64) === rand(mutable_rng, UInt64)
    @test replay.rng === mutable_rng.rng
end

@testset "R34 and R52 closed bridge method surface" begin
    M = StatefulRNG{typeof(Philox4x32(0x811))}
    required = Dict(
        function_ => Set{Method}() for function_ in (
            Random.rand,
            Random.rand!,
            Random.randn,
            Random.randn!,
            Random.randexp,
            Random.randexp!,
            Random.seed!,
            copy,
            parent,
        )
    )
    require = function (function_, signature)
        method = which(function_, signature)
        @test method.module === StatefulIR
        push!(required[function_], method)
    end

    for sampler in (
        Random.SamplerType{Bool},
        Random.SamplerType{UInt32},
        Random.SamplerType{UInt64},
        Random.SamplerType{Int32},
        Random.SamplerType{Int64},
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
    require(Random.randexp, Tuple{M})
    for T in (Float32, Float64)
        require(Random.randn, Tuple{M,Type{T}})
        require(Random.randexp, Tuple{M,Type{T}})
    end
    for T in PURE_UNIFORM_TYPES
        require(Random.rand!, Tuple{M,Vector{T}})
    end
    require(Random.rand!, Tuple{M,BitArray})
    for T in NORMAL_TYPES
        require(Random.randn!, Tuple{M,Vector{T}})
    end
    for T in EXPONENTIAL_TYPES
        require(Random.randexp!, Tuple{M,Vector{T}})
    end
    require(Random.seed!, Tuple{M,Int})
    require(copy, Tuple{M})
    require(parent, Tuple{M})

    for (function_, methods_) in required
        @test _bridge_methods(function_) == methods_
    end

    unit = UInt16(1):UInt16(2)
    stepped = UInt16(1):UInt16(2):UInt16(5)
    required_samplers = Set((
        which(Random.Sampler, Tuple{Type{M},typeof(unit),Val{1}}),
        which(Random.Sampler, Tuple{Type{M},typeof(stepped),Val{1}}),
    ))
    @test Set(
        method for method in methods(Random.Sampler) if method.module === StatefulIR
    ) == required_samplers

    ambiguities =
        filter(Test.detect_ambiguities(StatefulIR, Random; recursive = true)) do pair
            any(method -> method.module === StatefulIR, pair)
        end
    @test isempty(ambiguities)
end
