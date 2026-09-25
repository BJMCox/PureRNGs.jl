module PureRNGsEnzymeCoreExt

import EnzymeCore
import PureRNGs
import Random

const IR = PureRNGs
const ER = EnzymeCore.EnzymeRules
const _FloatElement = Union{IR._TransformFloat,IR._ComplexResult}
const _FloatArray = Array{<:_FloatElement}
const _FloatAbstractArray = AbstractArray{<:_FloatElement}
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

# The range fill puts its destination second, so it needs its own rule.
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
# process exits), so a threaded fill without a rule of its own stops here with an
# error instead: allocating draws, range and population sampling, and distribution fills.
@noinline function _threaded_fill_not_differentiable()
    throw(ArgumentError("threaded fills are not differentiable; use threaded = false"))
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

# The rejection test picks which candidate a Gamma draw takes, so
# differentiating through it would bias the shape gradient. These rules give
# the draw the implicit shape derivative of the Gamma CDF instead; only the
# shape carries a derivative.
@inline _scaled(config, direction, derivative) =
    ER.width(config) == 1 ? direction * derivative : map(d -> d * derivative, direction)
@inline _zero_tangent(config, value) =
    ER.width(config) == 1 ? EnzymeCore.make_zero(value) :
    ntuple(_ -> EnzymeCore.make_zero(value), Val(ER.width(config)))

for (draw, slope) in (
    (IR._gamma_value, IR._gamma_shape_derivative),
    (IR._gamma_log_value, IR._gamma_log_shape_derivative),
)
    @eval begin
        function ER.forward(
            config::ER.FwdConfig,
            ::EnzymeCore.Const{typeof($draw)},
            ::Type,
            shape::EnzymeCore.Annotation{F},
            codec::EnzymeCore.Annotation,
            rng::EnzymeCore.Annotation,
            position::EnzymeCore.Annotation,
            cursor::EnzymeCore.Annotation,
        ) where {F<:AbstractFloat}
            s = shape.val
            value = $draw(s, codec.val, rng.val, position.val, cursor.val)
            ER.needs_shadow(config) || return ER.needs_primal(config) ? value : nothing
            shadow =
                shape isa EnzymeCore.Const ? _zero_tangent(config, value) :
                _scaled(config, shape.dval, $slope(s, value))
            ER.needs_primal(config) || return shadow
            return ER.width(config) == 1 ? EnzymeCore.Duplicated(value, shadow) :
                   EnzymeCore.BatchDuplicated(value, shadow)
        end

        function ER.augmented_primal(
            config::ER.RevConfig,
            ::EnzymeCore.Const{typeof($draw)},
            ::Type,
            shape::EnzymeCore.Annotation{F},
            codec::EnzymeCore.Annotation,
            rng::EnzymeCore.Annotation,
            position::EnzymeCore.Annotation,
            cursor::EnzymeCore.Annotation,
        ) where {F<:AbstractFloat}
            s = shape.val
            value = $draw(s, codec.val, rng.val, position.val, cursor.val)
            primal = ER.needs_primal(config) ? value : nothing
            return ER.AugmentedReturn(primal, nothing, $slope(s, value))
        end

        function ER.reverse(
            config::ER.RevConfig,
            ::EnzymeCore.Const{typeof($draw)},
            dvalue::EnzymeCore.Active,
            derivative,
            shape::EnzymeCore.Annotation{F},
            codec::EnzymeCore.Annotation,
            rng::EnzymeCore.Annotation,
            position::EnzymeCore.Annotation,
            cursor::EnzymeCore.Annotation,
        ) where {F<:AbstractFloat}
            dshape =
                shape isa EnzymeCore.Active ? _scaled(config, dvalue.val, derivative) :
                nothing
            dcodec =
                codec isa EnzymeCore.Active ? _zero_tangent(config, codec.val) : nothing
            return (dshape, dcodec, nothing, nothing, nothing)
        end

        # A constant result, as for Beta's second draw at a fixed shape,
        # passes no derivative back.
        function ER.reverse(
            config::ER.RevConfig,
            ::EnzymeCore.Const{typeof($draw)},
            ::Type,
            derivative,
            shape::EnzymeCore.Annotation{F},
            codec::EnzymeCore.Annotation,
            rng::EnzymeCore.Annotation,
            position::EnzymeCore.Annotation,
            cursor::EnzymeCore.Annotation,
        ) where {F<:AbstractFloat}
            dshape =
                shape isa EnzymeCore.Active ? _zero_tangent(config, shape.val) : nothing
            dcodec =
                codec isa EnzymeCore.Active ? _zero_tangent(config, codec.val) : nothing
            return (dshape, dcodec, nothing, nothing, nothing)
        end
    end
end


# A device fill of a codec with parameters, such as a distribution's. The
# KernelAbstractions rules take no active kernel argument on a GPU, so these
# rules differentiate each element in a kernel of their own, in forward mode:
# the element is the generic kernel's draw, which every fill plan equals.
# Reverse mode runs one tangent fill per float field of the codec and dots it
# with the adjoints. Custom rules do not reach a device kernel, so the Gamma
# family takes the core's tangent, which carries the implicit shape derivative.
@inline function _element_position(codec, rng, ordinal, ::Val{T}) where {T}
    bits_lo, bits_hi = IR._bit_span(UInt64(ordinal - 1), IR._fill_width(codec, T))
    return IR._advance_position_unchecked(rng, bits_lo, bits_hi)
end

@inline _element(codec, rng, ordinal, element_type::Val{T}) where {T} =
    IR._transformed_draw_unchecked(
        codec,
        rng,
        _element_position(codec, rng, ordinal, element_type),
        T,
    )

@inline _element_tangent(codec, dcodec, rng, ordinal, element_type) = only(
    EnzymeCore.autodiff_deferred(
        EnzymeCore.Forward,
        EnzymeCore.Const(_element),
        EnzymeCore.Duplicated,
        EnzymeCore.Duplicated(codec, dcodec),
        EnzymeCore.Const(rng),
        EnzymeCore.Const(ordinal),
        EnzymeCore.Const(element_type),
    ),
)
@inline _element_tangent(codec::IR._GammaFamilyCodec, dcodec, rng, ordinal, element_type) =
    IR._transformed_tangent_unchecked(
        codec,
        dcodec,
        rng,
        _element_position(codec, rng, ordinal, element_type),
    )

@inline function _element_tangent!(tangents, ordinal, rng, codec, dcodec, element_type)
    tangents[1, ordinal] = _element_tangent(codec, dcodec, rng, ordinal, element_type)
    return nothing
end

@inline _as_columns(array) = reshape(array, 1, :)

@inline function _fill_tangent!(backend, tangents, rng, codec, dcodec, ::Type{T}) where {T}
    IR._foreach_column!(
        backend,
        _element_tangent!,
        _as_columns(tangents),
        false,
        rng,
        codec,
        dcodec,
        Val(T),
    )
    return tangents
end

# The float fields of a codec, depth first, and the codec with them replaced.
function _float_paths(::Type{S}, path = ()) where {S}
    S <: AbstractFloat && return [path]
    (isstructtype(S) && fieldcount(S) > 0) || return Tuple[]
    return reduce(
        vcat,
        (_float_paths(fieldtype(S, i), (path..., i)) for i = 1:fieldcount(S)),
    )
end

@generated function _floats(value::T) where {T}
    fields = map(_float_paths(T)) do path
        foldl((expression, i) -> :(getfield($expression, $i)), path; init = :value)
    end
    return Expr(:tuple, fields...)
end

@generated function _with_floats(value::T, floats) where {T}
    count = Ref(0)
    function build(S, expression)
        if S <: AbstractFloat
            count[] += 1
            return :(floats[$(count[])])
        end
        (isstructtype(S) && fieldcount(S) > 0) || return expression
        fields =
            (build(fieldtype(S, i), :(getfield($expression, $i))) for i = 1:fieldcount(S))
        return Expr(:new, S, fields...)
    end
    return build(T, :value)
end

@inline _direction(floats, k) =
    ntuple(j -> j == k ? one(floats[j]) : zero(floats[j]), Val(length(floats)))

const _DeviceFloatFill = EnzymeCore.Const{<:Type{<:AbstractFloat}}

function ER.forward(
    config::ER.FwdConfig,
    ::EnzymeCore.Const{typeof(IR._launch_device_fill!)},
    ::Type,
    backend::EnzymeCore.Const,
    rng::EnzymeCore.Const,
    destination::EnzymeCore.Duplicated,
    element_type::_DeviceFloatFill,
    codec::EnzymeCore.Duplicated,
    plan::EnzymeCore.Const,
)
    T = element_type.val
    IR._launch_device_fill!(backend.val, rng.val, destination.val, T, codec.val, plan.val)
    _fill_tangent!(backend.val, destination.dval, rng.val, codec.val, codec.dval, T)
    return _forward_return(config, destination.val, destination)
end

function ER.augmented_primal(
    config::ER.RevConfig,
    ::EnzymeCore.Const{typeof(IR._launch_device_fill!)},
    ::Type,
    backend::EnzymeCore.Const,
    rng::EnzymeCore.Const,
    destination::EnzymeCore.Duplicated,
    element_type::_DeviceFloatFill,
    codec::EnzymeCore.Active,
    plan::EnzymeCore.Const,
)
    T = element_type.val
    IR._launch_device_fill!(backend.val, rng.val, destination.val, T, codec.val, plan.val)
    primal = ER.needs_primal(config) ? destination.val : nothing
    shadow = ER.needs_shadow(config) ? destination.dval : nothing
    return ER.AugmentedReturn(primal, shadow, nothing)
end

function ER.reverse(
    config::ER.RevConfig,
    ::EnzymeCore.Const{typeof(IR._launch_device_fill!)},
    ::Type,
    tape,
    backend::EnzymeCore.Const,
    rng::EnzymeCore.Const,
    destination::EnzymeCore.Duplicated,
    element_type::_DeviceFloatFill,
    codec::EnzymeCore.Active,
    plan::EnzymeCore.Const,
)
    T = element_type.val
    adjoints = destination.dval
    tangents = similar(adjoints)
    floats = _floats(codec.val)
    gradient = ntuple(Val(length(floats))) do k
        direction = _with_floats(codec.val, _direction(floats, k))
        _fill_tangent!(backend.val, tangents, rng.val, codec.val, direction, T)
        return oftype(floats[k], sum(tangents .* adjoints))
    end
    # The fill overwrote the destination, so its adjoint stops here.
    fill!(adjoints, zero(eltype(adjoints)))
    return (nothing, nothing, nothing, nothing, _with_floats(codec.val, gradient), nothing)
end

end
