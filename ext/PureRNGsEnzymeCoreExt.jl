module PureRNGsEnzymeCoreExt

import EnzymeCore
import PureRNGs
import Random

const IR = PureRNGs
const ER = EnzymeCore.EnzymeRules
const _FloatArray = Array{<:Union{Float32,Float64}}
const _FloatAbstractArray = AbstractArray{<:Union{Float32,Float64}}
const _Duplicated = Union{EnzymeCore.Duplicated,EnzymeCore.DuplicatedNoNeed}
const _BatchDuplicated = Union{EnzymeCore.BatchDuplicated,EnzymeCore.BatchDuplicatedNoNeed}

@inline ER.inactive_type(::Type{<:IR.AbstractPureRNG}) = true

@inline _zero_shadow!(::EnzymeCore.Const) = nothing
@inline function _zero_shadow!(destination::_Duplicated)
    fill!(destination.dval, zero(eltype(destination.dval)))
    return nothing
end
@inline function _zero_shadow!(destination::_BatchDuplicated)
    for lane in destination.dval
        fill!(lane, zero(eltype(lane)))
    end
    return nothing
end

@inline _return_shadow(result::AbstractArray, destination::_Duplicated) = destination.dval
@inline _return_shadow(result::AbstractArray, destination::_BatchDuplicated) =
    destination.dval
@inline _return_shadow(result::Tuple, destination::_Duplicated) =
    (destination.dval, last(result))
@inline _return_shadow(result::Tuple, destination::_BatchDuplicated) =
    map(lane -> (lane, last(result)), destination.dval)

@inline _zero_return(result::AbstractArray) = EnzymeCore.make_zero(result)
@inline _zero_return(result::Tuple) = (EnzymeCore.make_zero(first(result)), last(result))
@inline _return_shadow(config, result, destination) = _return_shadow(result, destination)
@inline function _return_shadow(config, result, ::EnzymeCore.Const)
    ER.width(config) == 1 && return _zero_return(result)
    return ntuple(_ -> _zero_return(result), Val(ER.width(config)))
end

@inline function _forward_return(config, result, destination)
    ER.needs_shadow(config) || return ER.needs_primal(config) ? result : nothing
    shadow = _return_shadow(config, result, destination)
    ER.needs_primal(config) || return shadow
    return ER.width(config) == 1 ? EnzymeCore.Duplicated(result, shadow) :
           EnzymeCore.BatchDuplicated(result, shadow)
end

@inline _primal(annotation::EnzymeCore.Annotation) = annotation.val

# `arguments` is the whole call in order, so a fill may carry its destination in
# any position.
@inline function _forward_pure(
    config,
    function_annotation,
    arguments,
    destination,
    threaded,
)
    result = function_annotation.val(map(_primal, arguments)...; threaded = threaded)
    _zero_shadow!(destination)
    return _forward_return(config, result, destination)
end

@inline function _augmented_pure(
    config,
    function_annotation,
    arguments,
    destination,
    threaded,
)
    result = function_annotation.val(map(_primal, arguments)...; threaded = threaded)
    _zero_shadow!(destination)
    primal = ER.needs_primal(config) ? result : nothing
    shadow = ER.needs_shadow(config) ? _return_shadow(config, result, destination) : nothing
    return ER.AugmentedReturn(primal, shadow, nothing)
end

# An active input still carries no tangent, but reverse mode needs a value for it
# rather than the `nothing` a Const or Duplicated input takes.
@inline _argument_adjoint(::EnzymeCore.Annotation) = nothing
@inline _argument_adjoint(argument::EnzymeCore.Active) = EnzymeCore.make_zero(argument.val)

@inline function _forward_stateful(config, function_annotation, rng, destination)
    result = function_annotation.val(rng.val, destination.val)
    _zero_shadow!(destination)
    return _forward_return(config, result, destination)
end

@inline function _augmented_stateful(config, function_annotation, rng, destination)
    result = function_annotation.val(rng.val, destination.val)
    _zero_shadow!(destination)
    primal = ER.needs_primal(config) ? result : nothing
    shadow = ER.needs_shadow(config) ? _return_shadow(config, result, destination) : nothing
    return ER.AugmentedReturn(primal, shadow, nothing)
end

for fill_function in (
    Random.rand!,
    Random.randn!,
    Random.randexp!,
    IR.rand_next!,
    IR.randn_next!,
    IR.randexp_next!,
)
    @eval begin
        @inline function ER.forward(
            config::ER.FwdConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Const{<:IR.AbstractPureRNG},
            destination::EnzymeCore.Annotation{<:_FloatAbstractArray};
            threaded::Bool = false,
        )
            return _forward_pure(
                config,
                function_annotation,
                (rng, destination),
                destination,
                threaded,
            )
        end

        @inline function ER.augmented_primal(
            config::ER.RevConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Const{<:IR.AbstractPureRNG},
            destination::EnzymeCore.Annotation{<:_FloatAbstractArray};
            threaded::Bool = false,
        )
            return _augmented_pure(
                config,
                function_annotation,
                (rng, destination),
                destination,
                threaded,
            )
        end

        @inline function ER.reverse(
            ::ER.RevConfig,
            ::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            ::Nothing,
            ::EnzymeCore.Const{<:IR.AbstractPureRNG},
            destination::EnzymeCore.Annotation{<:_FloatAbstractArray};
            threaded::Bool = false,
        )
            _zero_shadow!(destination)
            return nothing, nothing
        end
    end
end

# The range fill puts its destination second, so it needs its own rule rather
# than the untyped three-argument one above.
for fill_function in (Random.rand!, IR.rand_next!)
    @eval begin
        @inline function ER.forward(
            config::ER.FwdConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Const{<:IR.AbstractPureRNG},
            destination::EnzymeCore.Annotation{<:AbstractArray},
            range::EnzymeCore.Annotation{<:AbstractRange};
            threaded::Bool = false,
        )
            return _forward_pure(
                config,
                function_annotation,
                (rng, destination, range),
                destination,
                threaded,
            )
        end

        @inline function ER.augmented_primal(
            config::ER.RevConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Const{<:IR.AbstractPureRNG},
            destination::EnzymeCore.Annotation{<:AbstractArray},
            range::EnzymeCore.Annotation{<:AbstractRange};
            threaded::Bool = false,
        )
            return _augmented_pure(
                config,
                function_annotation,
                (rng, destination, range),
                destination,
                threaded,
            )
        end

        @inline function ER.reverse(
            ::ER.RevConfig,
            ::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            ::Nothing,
            ::EnzymeCore.Const{<:IR.AbstractPureRNG},
            destination::EnzymeCore.Annotation{<:AbstractArray},
            range::EnzymeCore.Annotation{<:AbstractRange};
            threaded::Bool = false,
        )
            _zero_shadow!(destination)
            return nothing, nothing, _argument_adjoint(range)
        end
    end
end

for fill_function in (Random.rand!, Random.randn!, Random.randexp!)
    @eval begin
        @inline function ER.forward(
            config::ER.FwdConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Annotation{<:IR.StatefulRNG},
            destination::EnzymeCore.Annotation{<:_FloatArray},
        )
            return _forward_stateful(config, function_annotation, rng, destination)
        end

        @inline function ER.augmented_primal(
            config::ER.RevConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Annotation{<:IR.StatefulRNG},
            destination::EnzymeCore.Annotation{<:_FloatArray},
        )
            return _augmented_stateful(config, function_annotation, rng, destination)
        end

        @inline function ER.reverse(
            ::ER.RevConfig,
            ::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            ::Nothing,
            ::EnzymeCore.Annotation{<:IR.StatefulRNG},
            destination::EnzymeCore.Annotation{<:_FloatArray},
        )
            _zero_shadow!(destination)
            return nothing, nothing
        end
    end
end

# Distribution and population fills have no rule: Enzyme differentiates them
# directly and returns the pathwise gradient, holding the random bits fixed.
# Enzyme cannot yet differentiate the task scheduler of a threaded CPU fill (the
# process exits), so an active threaded fill stops here with an error instead.
@noinline function _threaded_fill_not_differentiable()
    throw(
        ArgumentError(
            "threaded distribution and population fills are not differentiable; use threaded = false",
        ),
    )
end

@inline ER.forward(
    ::ER.FwdConfig,
    ::EnzymeCore.Const{typeof(IR._run_chunks)},
    ::Type,
    ::EnzymeCore.Annotation,
    ::EnzymeCore.Annotation{Int},
    ::EnzymeCore.Annotation{Int},
) = _threaded_fill_not_differentiable()

@inline ER.augmented_primal(
    ::ER.RevConfig,
    ::EnzymeCore.Const{typeof(IR._run_chunks)},
    ::Type,
    ::EnzymeCore.Annotation,
    ::EnzymeCore.Annotation{Int},
    ::EnzymeCore.Annotation{Int},
) = _threaded_fill_not_differentiable()

@inline ER.reverse(
    ::ER.RevConfig,
    ::EnzymeCore.Const{typeof(IR._run_chunks)},
    ::Type,
    tape,
    ::EnzymeCore.Annotation,
    ::EnzymeCore.Annotation{Int},
    ::EnzymeCore.Annotation{Int},
) = _threaded_fill_not_differentiable()

end
