using Random

const AuditIR = PureRNGs

_audit_methods(f) = [method for method in methods(f) if method.module === AuditIR]
function _pure_audit_methods(function_)
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

@testset "R1 and R49 closed non-bridge method surface" begin
    foreign = Base.IdSet{Any}()
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
        randexp!,
        rand_next,
        rand_next!,
        randn_next,
        randn_next!,
        randexp_next,
        randexp_next!,
        randat,
        randnat,
        randexpat,
        splitrng,
        subrng,
        randsample,
        randsample_next,
        randsample!,
        randsample_next!,
    )
    # showerror is R70's StreamExhausted display, dispatched on the package type.
    @test foreign_functions == Set((
        rand,
        rand!,
        randn,
        randn!,
        randexp,
        randexp!,
        Random.seed!,
        copy,
        parent,
        showerror,
    ))

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
    require(rand_next, Tuple{R,Tuple{Int}})
    require(randn_next, Tuple{R})
    require(randn_next, Tuple{R,Int})
    require(randn_next, Tuple{R,Tuple{Int}})
    require(randexp_next, Tuple{R})
    require(randexp_next, Tuple{R,Int})
    require(randexp_next, Tuple{R,Tuple{Int}})
    for T in PURE_UNIFORM_TYPES
        require(rand, Tuple{R,Type{T}})
        require(rand, Tuple{R,Type{T},Int})
        require(rand, Tuple{R,Type{T},Tuple{Int}})
        require(rand!, Tuple{R,Vector{T}})
        require(rand_next, Tuple{R,Type{T}})
        require(rand_next, Tuple{R,Type{T},Int})
        require(rand_next, Tuple{R,Type{T},Tuple{Int}})
        require(rand_next!, Tuple{R,Vector{T}})
        require(randat, Tuple{R,Type{T},Int})
        require(randat, Tuple{R,Type{T},UnitRange{Int}})
    end
    for T in NORMAL_TYPES
        require(randn, Tuple{R,Type{T}})
        require(randn, Tuple{R,Type{T},Int})
        require(randn, Tuple{R,Type{T},Tuple{Int}})
        require(randn!, Tuple{R,Vector{T}})
        require(randn_next, Tuple{R,Type{T}})
        require(randn_next, Tuple{R,Type{T},Int})
        require(randn_next, Tuple{R,Type{T},Tuple{Int}})
        require(randn_next!, Tuple{R,Vector{T}})
        require(randnat, Tuple{R,Type{T},Int})
        require(randnat, Tuple{R,Type{T},UnitRange{Int}})
    end
    for T in EXPONENTIAL_TYPES
        require(randexp, Tuple{R,Type{T}})
        require(randexp, Tuple{R,Type{T},Int})
        require(randexp, Tuple{R,Type{T},Tuple{Int}})
        require(randexp!, Tuple{R,Vector{T}})
        require(randexp_next, Tuple{R,Type{T}})
        require(randexp_next, Tuple{R,Type{T},Int})
        require(randexp_next, Tuple{R,Type{T},Tuple{Int}})
        require(randexp_next!, Tuple{R,Vector{T}})
        require(randexpat, Tuple{R,Type{T},Int})
        require(randexpat, Tuple{R,Type{T},UnitRange{Int}})
    end
    for T in RANGE_INTS
        Range = typeof(T(1):T(2))
        require(rand, Tuple{R,Range})
        require(rand, Tuple{R,Range,Int})
        require(rand, Tuple{R,Range,Tuple{Int}})
        require(rand!, Tuple{R,Vector{T},Range})
        require(rand_next, Tuple{R,Range})
        require(rand_next, Tuple{R,Range,Int})
        require(rand_next, Tuple{R,Range,Tuple{Int}})
        require(rand_next!, Tuple{R,Vector{T},Range})
    end
    require(splitrng, Tuple{R})
    require(splitrng, Tuple{R,Int})
    require(splitrng, Tuple{R,Val{2}})
    require(subrng, Tuple{R,Int})
    population = Int32[1, 2, 3]
    weights = Float64[1, 2, 3]
    # A table reaches the same method as a weight vector, so the required sets stay
    # the same size and the union is pinned on every weighted position.
    for function_ in (randsample, randsample_next)
        require(function_, Tuple{R,typeof(population)})
        require(function_, Tuple{R,typeof(population),Int})
        for W in (typeof(weights), WeightTable)
            require(function_, Tuple{R,typeof(population),W})
            require(function_, Tuple{R,typeof(population),W,Int})
        end
    end
    for function_ in (randsample!, randsample_next!)
        require(function_, Tuple{R,typeof(population),Vector{Int32}})
        for W in (typeof(weights), WeightTable)
            require(function_, Tuple{R,typeof(population),W,Vector{Int32}})
        end
    end

    for function_ in owned_functions
        methods_ =
            function_ in (rand, rand!, randn, randn!, randexp, randexp!) ?
            _pure_audit_methods(function_) : _audit_methods(function_)
        @test Set(methods_) == required[function_]
    end
    @test all(
        Base.unwrap_unionall(method.sig).parameters[2] <: AuditIR.AbstractPureRNG for
        function_ in (rand, rand!, randn, randn!, randexp, randexp!) for
        method in _pure_audit_methods(function_)
    )
    @test all(
        Base.kwarg_decl(method) == [:threaded] for
        function_ in (rand!, rand_next!, randn!, randn_next!, randexp!, randexp_next!) for
        method in (
            function_ in (rand!, randn!, randexp!) ? _pure_audit_methods(function_) :
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
            function_ in (rand, randn, randexp) ? _pure_audit_methods(function_) :
            _audit_methods(function_)
        )
    )
    @test all(
        Base.kwarg_decl(method) == [:threaded] for
        function_ in (randsample!, randsample_next!) for method in _audit_methods(function_)
    )

    cpu = AuditIR.MLDataDevices.CPUDevice()
    device_method = which(cpu, Tuple{R})
    @test device_method.module === AuditIR
    # Julia 1.10 also lists the shadowed AbstractDevice fallback.
    @test Set(
        method for method in methods(cpu) if method.module === AuditIR &&
            Base.unwrap_unionall(method.sig).parameters[1] <:
            AuditIR.MLDataDevices.CPUDevice
    ) == Set((device_method,))
    @test isempty(Base.kwarg_decl(device_method))
end

@testset "R9 and R12b assigned constants" begin
    @test AuditIR._NARROW_SPLIT_COUNT === UInt64(0xffffffff)
end

@testset "R47 implemented deterministic error closure" begin
    rng = Philox4x32(0xa72)
    exhausted =
        AuditIR._rebuild(rng, AuditIR._terminal64(AuditIR._max_block(rng)), rng.device)
    wrong_uniform_destination = WrongDeviceArray(Vector{UInt32}(undef, 1))
    wrong_normal_destination = WrongDeviceArray(Vector{Float32}(undef, 1))
    wrong_exponential_destination = WrongDeviceArray(Vector{Float32}(undef, 1))
    argument_errors = (
        (:negative_seed, () -> Philox2x32(-1)),
        (:oversized_seed, () -> Philox2x32(big(1) << 32)),
        (:negative_split, () -> splitrng(rng, -1)),
        (:invalid_static_split, () -> splitrng(rng, Val(UInt32(1)))),
        (:narrow_split_namespace, () -> splitrng(Philox2x32(1), UInt64(0x1_0000_0000))),
        (:randat_index, () -> randat(rng, UInt32, 0)),
        (:randnat_index, () -> randnat(rng, Float32, 0)),
        (:randexpat_index, () -> randexpat(rng, Float32, 0)),
        (:empty_range, () -> rand(rng, UInt8(2):UInt8(1))),
        (:empty_range_continuation, () -> rand_next(rng, UInt8(2):UInt8(1))),
        (:negative_uniform_dimension, () -> rand(rng, UInt32, -1)),
        (:negative_uniform_continuation_dimension, () -> rand_next(rng, UInt32, -1)),
        (:negative_normal_dimension, () -> randn(rng, Float32, -1)),
        (:negative_normal_continuation_dimension, () -> randn_next(rng, Float32, -1)),
        (:negative_exponential_dimension, () -> randexp(rng, Float32, -1)),
        (
            :negative_exponential_continuation_dimension,
            () -> randexp_next(rng, Float32, -1),
        ),
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
        (:exponential_device_mismatch, () -> randexp!(rng, wrong_exponential_destination)),
        (
            :exponential_continuation_device_mismatch,
            () -> randexp_next!(rng, wrong_exponential_destination),
        ),
    )
    exhausted_errors = (
        (:randat_capacity, () -> randat(exhausted, UInt32, 1)),
        (:randnat_capacity, () -> randnat(exhausted, Float32, 1)),
        (:randexpat_capacity, () -> randexpat(exhausted, Float32, 1)),
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
        (:exponential_pure_capacity, () -> randexp(exhausted, Float32)),
        (:exponential_continuation_capacity, () -> randexp_next(exhausted, Float32)),
        (:exponential_fill_capacity, () -> randexp!(exhausted, Vector{Float32}(undef, 1))),
        (
            :exponential_continuation_fill_capacity,
            () -> randexp_next!(exhausted, Vector{Float32}(undef, 1)),
        ),
        (:exponential_allocating_capacity, () -> randexp(exhausted, Float32, 1)),
        (
            :exponential_continuation_allocating_capacity,
            () -> randexp_next(exhausted, Float32, 1),
        ),
        (:range_pure_capacity, () -> rand(exhausted, UInt8(1):UInt8(2))),
        (:range_continuation_capacity, () -> rand_next(exhausted, UInt8(1):UInt8(2))),
        (:range_allocating_capacity, () -> rand(exhausted, UInt8(1):UInt8(2), 1)),
        (
            :range_continuation_allocating_capacity,
            () -> rand_next(exhausted, UInt8(1):UInt8(2), 1),
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
        (
            :exponential_threaded_type,
            () -> randexp!(rng, Vector{Float32}(undef, 1); threaded = 1),
        ),
        (
            :exponential_continuation_threaded_type,
            () -> randexp_next!(rng, Vector{Float32}(undef, 1); threaded = 1),
        ),
    )
    method_errors = (
        (:uniform_result_type, () -> rand(rng, Float16)),
        (:uniform_continuation_result_type, () -> rand_next(rng, Float16)),
        (:uniform_destination_type, () -> rand!(rng, Vector{Float16}(undef, 1))),
        (:normal_result_type, () -> randn(rng, Float16)),
        (:normal_continuation_result_type, () -> randn_next(rng, Float16)),
        (:normal_destination_type, () -> randn!(rng, Vector{Float16}(undef, 1))),
        (:exponential_result_type, () -> randexp(rng, Float16)),
        (:exponential_continuation_result_type, () -> randexp_next(rng, Float16)),
        (:exponential_destination_type, () -> randexp!(rng, Vector{Float16}(undef, 1))),
    )

    for (expected, cases) in (
        (ArgumentError, argument_errors),
        (StreamExhausted, exhausted_errors),
        (TypeError, type_errors),
        (MethodError, method_errors),
    )
        for (name, call) in cases
            @testset "$name" begin
                # StreamExhausted is parametric, so compare by subtyping.
                @test typeof(_audit_error(call)) <: expected
            end
        end
    end

end

@testset "R70 StreamExhausted payload" begin
    base = Philox4x32(0x970)
    near_end = AuditIR._rebuild(
        base,
        _range_position_from_absolute(base, _range_capacity(base) - 10),
        base.device,
    )
    population = Int32[2, 3, 5, 7]
    for (name, span, call) in (
        (:scalar, UInt128(32), () -> rand(near_end, UInt32)),
        (:fill, UInt128(64), () -> rand!(near_end, Vector{UInt32}(undef, 2))),
        (:addressed, UInt128(64), () -> randat(near_end, UInt32, 2)),
        (:sampling, UInt128(128), () -> randsample(near_end, population, 2)),
    )
        @testset "$name" begin
            exhausted = _audit_error(call)
            @test exhausted isa StreamExhausted
            @test exhausted.rng === near_end
            @test exhausted.bits === span
            @test occursin("Philox4x32", sprint(showerror, exhausted))
        end
    end

    @test_throws ArgumentError Philox4x32(rngkey(near_end), rngposition(near_end) + 11)
end
