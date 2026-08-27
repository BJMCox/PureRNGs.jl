using Random

const AuditIR = PureRNGs

_audit_methods(f) = [method for method in methods(f) if method.module === AuditIR]
function _immutable_audit_methods(function_)
    return filter(_audit_methods(function_)) do method
        signature = Base.unwrap_unionall(method.sig)
        length(signature.parameters) >= 2 || return false
        return signature.parameters[2] <: AuditIR.AbstractPureRNG
    end
end

function _audit_error(call)
    try
        call()
        return nothing
    catch error
        return error
    end
end

@testset "R1, R49, and row 735 closed non-bridge method surface" begin
    foreign = IdSet{Any}()
    foreign_functions = Set{Any}()
    for module_ in (Base, Random), name in names(module_; all = true, imported = true)
        isdefined(module_, name) || continue
        function_ = getfield(module_, name)
        function_ isa Function || continue
        function_ in foreign && continue
        push!(foreign, function_)
        isempty(_audit_methods(function_)) || push!(foreign_functions, function_)
    end
    owned_functions = (
        rand,
        rand!,
        randn,
        randn!,
        randexp,
        rand_next,
        rand_next!,
        randn_next,
        randn_next!,
        randexp_next,
        randat,
        randnat,
        randexpat,
        splitrng,
        subrng,
        randsample,
        randsample_next,
    )
    @test foreign_functions ==
          Set((rand, rand!, randn, randn!, randexp, Random.seed!, copy, parent))

    required = Dict(function_ => Set{Method}() for function_ in owned_functions)
    require = function (function_, signature)
        method = which(function_, signature)
        @test method.module === AuditIR
        push!(required[function_], method)
        return nothing
    end

    rng = Philox4x32(0xa71)
    R = typeof(rng)
    require(rand, Tuple{R})
    require(randn, Tuple{R})
    require(randexp, Tuple{R})
    require(rand_next, Tuple{R})
    require(rand_next, Tuple{R,Int})
    require(randn_next, Tuple{R})
    require(randn_next, Tuple{R,Int})
    require(randexp_next, Tuple{R})
    for T in PURE_UNIFORM_TYPES
        require(rand, Tuple{R,Type{T}})
        require(rand, Tuple{R,Type{T},Int})
        require(rand!, Tuple{R,Vector{T}})
        require(rand_next, Tuple{R,Type{T}})
        require(rand_next, Tuple{R,Type{T},Int})
        require(rand_next!, Tuple{R,Vector{T}})
        require(randat, Tuple{R,Type{T},Int})
    end
    for T in NORMAL_TYPES
        require(randn, Tuple{R,Type{T}})
        require(randn, Tuple{R,Type{T},Int})
        require(randn!, Tuple{R,Vector{T}})
        require(randn_next, Tuple{R,Type{T}})
        require(randn_next, Tuple{R,Type{T},Int})
        require(randn_next!, Tuple{R,Vector{T}})
        require(randnat, Tuple{R,Type{T},Int})
    end
    for T in EXPONENTIAL_TYPES
        require(randexp, Tuple{R,Type{T}})
        require(randexp_next, Tuple{R,Type{T}})
        require(randexpat, Tuple{R,Type{T},Int})
    end
    for T in RANGE_INTS
        Range = typeof(T(1):T(2))
        require(rand, Tuple{R,Range})
        require(rand, Tuple{R,Range,Int})
        require(rand_next, Tuple{R,Range})
        require(rand_next, Tuple{R,Range,Int})
    end
    require(splitrng, Tuple{R})
    require(splitrng, Tuple{R,Int})
    require(splitrng, Tuple{R,Val{2}})
    require(subrng, Tuple{R,Int})
    population = Int32[1, 2, 3]
    weights = Float64[1, 2, 3]
    for function_ in (randsample, randsample_next)
        require(function_, Tuple{R,typeof(population)})
        require(function_, Tuple{R,typeof(population),Int})
        require(function_, Tuple{R,typeof(population),typeof(weights)})
        require(function_, Tuple{R,typeof(population),typeof(weights),Int})
    end

    for function_ in owned_functions
        methods_ =
            function_ in (rand, rand!, randn, randn!, randexp) ?
            _immutable_audit_methods(function_) : _audit_methods(function_)
        @test Set(methods_) == required[function_]
    end
    @test all(
        Base.unwrap_unionall(method.sig).parameters[2] <: AuditIR.AbstractPureRNG for
        function_ in (rand, rand!, randn, randn!, randexp) for
        method in _immutable_audit_methods(function_)
    )
    @test all(
        Base.kwarg_decl(method) == [:threaded] for
        function_ in (rand!, rand_next!, randn!, randn_next!) for method in (
            function_ in (rand!, randn!) ? _immutable_audit_methods(function_) :
            _audit_methods(function_)
        )
    )
    @test all(
        isempty(Base.kwarg_decl(method)) for function_ in (
            rand,
            randn,
            randexp,
            rand_next,
            randn_next,
            randexp_next,
            randat,
            randnat,
            randexpat,
            splitrng,
            subrng,
            randsample,
            randsample_next,
        ) for method in (
            function_ in (rand, randn, randexp) ? _immutable_audit_methods(function_) :
            _audit_methods(function_)
        )
    )

    cpu = AuditIR.MLDataDevices.CPUDevice()
    device_method = which(cpu, Tuple{R})
    @test device_method.module === AuditIR
    @test Set(method for method in methods(cpu) if method.module === AuditIR) ==
          Set((device_method,))
    @test isempty(Base.kwarg_decl(device_method))
end

@testset "R9 and R10 assigned constants" begin
    @test AuditIR.FAMILY_BITS === UInt32(0)
    @test AuditIR.FAMILY_NORMAL === UInt32(1)
    @test AuditIR.FAMILY_EXP === UInt32(2)
    @test AuditIR.FAMILY_RANGE === UInt32(3)
    @test AuditIR._DERIVE_TAG === UInt32(0xc0ffee00)
    @test AuditIR._SPLIT_SUBTAG === UInt32(0)
    @test AuditIR._FOLD_SUBTAG === UInt32(1)
    @test AuditIR._THREEFRY_FOLD_INDEX === UInt32(0xffffffff)
    @test AuditIR._NARROW_SPLIT_COUNT === UInt64(0xffffffff)
end

@testset "R47 and row 744 implemented deterministic error closure" begin
    rng = Philox4x32(0xa72)
    exhausted =
        AuditIR._rebuild(rng, AuditIR._terminal64(AuditIR._max_block(rng)), rng.device)
    wrong_uniform_destination = WrongDeviceArray(Vector{UInt32}(undef, 1))
    wrong_normal_destination = WrongDeviceArray(Vector{Float32}(undef, 1))
    argument_errors = (
        (:negative_seed, () -> Philox2x32(-1)),
        (:oversized_seed, () -> Philox2x32(big(1) << 32)),
        (:negative_split, () -> splitrng(rng, -1)),
        (:invalid_static_split, () -> splitrng(rng, Val(UInt32(1)))),
        (:narrow_split_namespace, () -> splitrng(Philox2x32(1), UInt64(0x1_0000_0000))),
        (:randat_index, () -> randat(rng, UInt32, 0)),
        (:randat_capacity, () -> randat(exhausted, UInt32, 1)),
        (:randnat_index, () -> randnat(rng, Float32, 0)),
        (:randnat_capacity, () -> randnat(exhausted, Float32, 1)),
        (:pure_capacity, () -> rand(exhausted, UInt32)),
        (:continuation_capacity, () -> rand_next(exhausted, UInt32)),
        (:fill_capacity, () -> rand!(exhausted, Vector{UInt32}(undef, 1))),
        (
            :continuation_fill_capacity,
            () -> rand_next!(exhausted, Vector{UInt32}(undef, 1)),
        ),
        (:allocating_capacity, () -> rand(exhausted, UInt32, 1)),
        (:continuation_allocating_capacity, () -> rand_next(exhausted, UInt32, 1)),
        (:normal_pure_capacity, () -> randn(exhausted, Float32)),
        (:normal_continuation_capacity, () -> randn_next(exhausted, Float32)),
        (:normal_fill_capacity, () -> randn!(exhausted, Vector{Float32}(undef, 1))),
        (
            :normal_continuation_fill_capacity,
            () -> randn_next!(exhausted, Vector{Float32}(undef, 1)),
        ),
        (:normal_allocating_capacity, () -> randn(exhausted, Float32, 1)),
        (:normal_continuation_allocating_capacity, () -> randn_next(exhausted, Float32, 1)),
        (:range_pure_capacity, () -> rand(exhausted, UInt8(1):UInt8(2))),
        (:range_continuation_capacity, () -> rand_next(exhausted, UInt8(1):UInt8(2))),
        (:range_allocating_capacity, () -> rand(exhausted, UInt8(1):UInt8(2), 1)),
        (
            :range_continuation_allocating_capacity,
            () -> rand_next(exhausted, UInt8(1):UInt8(2), 1),
        ),
        (:empty_range, () -> rand(rng, UInt8(2):UInt8(1))),
        (:empty_range_continuation, () -> rand_next(rng, UInt8(2):UInt8(1))),
        (:negative_uniform_dimension, () -> rand(rng, UInt32, -1)),
        (:negative_uniform_continuation_dimension, () -> rand_next(rng, UInt32, -1)),
        (:negative_normal_dimension, () -> randn(rng, Float32, -1)),
        (:negative_normal_continuation_dimension, () -> randn_next(rng, Float32, -1)),
        (:untyped_uniform, () -> rand(rng)),
        (:untyped_normal, () -> randn(rng)),
        (:untyped_exponential, () -> randexp(rng)),
        (:uniform_device_mismatch, () -> rand!(rng, wrong_uniform_destination)),
        (
            :uniform_continuation_device_mismatch,
            () -> rand_next!(rng, wrong_uniform_destination),
        ),
        (:normal_device_mismatch, () -> randn!(rng, wrong_normal_destination)),
        (
            :normal_continuation_device_mismatch,
            () -> randn_next!(rng, wrong_normal_destination),
        ),
    )
    type_errors = (
        (:uniform_threaded_type, () -> rand!(rng, Vector{UInt32}(undef, 1); threaded = 1)),
        (
            :uniform_continuation_threaded_type,
            () -> rand_next!(rng, Vector{UInt32}(undef, 1); threaded = 1),
        ),
        (:normal_threaded_type, () -> randn!(rng, Vector{Float32}(undef, 1); threaded = 1)),
        (
            :normal_continuation_threaded_type,
            () -> randn_next!(rng, Vector{Float32}(undef, 1); threaded = 1),
        ),
    )
    method_errors = (
        (:uniform_result_type, () -> rand(rng, Float16)),
        (:uniform_continuation_result_type, () -> rand_next(rng, Float16)),
        (:uniform_destination_type, () -> rand!(rng, Vector{Float16}(undef, 1))),
        (:normal_result_type, () -> randn(rng, Float16)),
        (:normal_continuation_result_type, () -> randn_next(rng, Float16)),
        (:normal_destination_type, () -> randn!(rng, Vector{Float16}(undef, 1))),
    )

    for (expected, cases) in (
        (ArgumentError, argument_errors),
        (TypeError, type_errors),
        (MethodError, method_errors),
    )
        for (name, call) in cases
            @testset "$name" begin
                @test typeof(_audit_error(call)) === expected
            end
        end
    end

    @test sprint(showerror, _audit_error(() -> rand(rng))) ==
          "ArgumentError: untyped immutable draws are forbidden; use rand(rng, T)"
    @test sprint(showerror, _audit_error(() -> randn(rng))) ==
          "ArgumentError: untyped immutable draws are forbidden; use randn(rng, T)"
    @test sprint(showerror, _audit_error(() -> randexp(rng))) ==
          "ArgumentError: untyped immutable draws are forbidden; use randexp(rng, T)"
end
