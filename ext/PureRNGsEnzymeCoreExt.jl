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
            threaded::Bool = true,
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
            threaded::Bool = true,
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
            threaded::Bool = true,
        )
            _zero_shadow!(destination)
            return nothing, nothing
        end
    end
end

# Distribution fills and unweighted population fills share the shape
# `(rng, argument, destination)`. The distribution types live behind the
# Distributions extension, so the argument position stays untyped; the range
# rule below is more specific and keeps `rand!(rng, destination, range)` out.
for fill_function in (Random.rand!, IR.rand_next!, IR.randsample!, IR.randsample_next!)
    @eval begin
        @inline function ER.forward(
            config::ER.FwdConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Const{<:IR.AbstractPureRNG},
            argument::EnzymeCore.Annotation,
            destination::EnzymeCore.Annotation{<:AbstractArray};
            threaded::Bool = true,
        )
            return _forward_pure(
                config,
                function_annotation,
                (rng, argument, destination),
                destination,
                threaded,
            )
        end

        @inline function ER.augmented_primal(
            config::ER.RevConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Const{<:IR.AbstractPureRNG},
            argument::EnzymeCore.Annotation,
            destination::EnzymeCore.Annotation{<:AbstractArray};
            threaded::Bool = true,
        )
            return _augmented_pure(
                config,
                function_annotation,
                (rng, argument, destination),
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
            argument::EnzymeCore.Annotation,
            destination::EnzymeCore.Annotation{<:AbstractArray};
            threaded::Bool = true,
        )
            _zero_shadow!(destination)
            return nothing, _argument_adjoint(argument), nothing
        end
    end
end

for fill_function in (IR.randsample!, IR.randsample_next!)
    @eval begin
        @inline function ER.forward(
            config::ER.FwdConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Const{<:IR.AbstractPureRNG},
            population::EnzymeCore.Annotation,
            weights::EnzymeCore.Annotation,
            destination::EnzymeCore.Annotation{<:AbstractArray};
            threaded::Bool = true,
        )
            return _forward_pure(
                config,
                function_annotation,
                (rng, population, weights, destination),
                destination,
                threaded,
            )
        end

        @inline function ER.augmented_primal(
            config::ER.RevConfig,
            function_annotation::EnzymeCore.Const{typeof($fill_function)},
            ::Type,
            rng::EnzymeCore.Const{<:IR.AbstractPureRNG},
            population::EnzymeCore.Annotation,
            weights::EnzymeCore.Annotation,
            destination::EnzymeCore.Annotation{<:AbstractArray};
            threaded::Bool = true,
        )
            return _augmented_pure(
                config,
                function_annotation,
                (rng, population, weights, destination),
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
            population::EnzymeCore.Annotation,
            weights::EnzymeCore.Annotation,
            destination::EnzymeCore.Annotation{<:AbstractArray};
            threaded::Bool = true,
        )
            _zero_shadow!(destination)
            return nothing,
            _argument_adjoint(population),
            _argument_adjoint(weights),
            nothing
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
            threaded::Bool = true,
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
            threaded::Bool = true,
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
            threaded::Bool = true,
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

end
