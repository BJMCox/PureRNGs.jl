# A Dirichlet draw is a column of the core's Dirichlet codec, so an engine's
# column hook serves it. The bodies take float and dual shapes alike; the
# ForwardDiff extension forwards the dual ones.
const _FloatDirichlet = Distributions.Dirichlet{<:_FloatType}

@inline function _dirichlet_shape_flags(a)
    value = IR._primal_value(a)
    tiny = value < IR._gamma_log_overflow_bound(typeof(value))
    return UInt8(_positive_finite(a)) | (UInt8(tiny) << 1)
end

# One validation reduction also checks whether all log-Gamma draws can overflow.
# Ordinary device kernels then omit the cold repair and its live cursor state.
function _validate_dirichlet(d)
    flags = mapreduce(_dirichlet_shape_flags, &, d.alpha; init = UInt8(3))
    iszero(flags & UInt8(1)) && throw(ArgumentError("invalid Dirichlet parameters"))
    return iszero(flags & UInt8(2)) ? Val(false) : Val(true)
end

_dirichlet_storage(rng, alpha::Array) = IR._transfer_array(IR._engine_backend(rng), alpha)
function _dirichlet_storage(rng, alpha)
    IR._check_fill_device(rng, alpha)
    return alpha
end

_dirichlet_codec(rng, d, recover = _validate_dirichlet(d)) =
    IR._DirichletCodec(_dirichlet_storage(rng, d.alpha), recover)

function IR._engine_rand_next!(
    rng,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:Real}
    recover = _validate_dirichlet(d)
    IR._check_serviceability(rng, IR._primal_float(T))
    IR._check_fill_device(rng, destination)
    IR._check_parameter_overlap(destination, d.alpha, "the concentrations")
    size(destination, 1) == length(d) || throw(
        DimensionMismatch(
            "destination has $(size(destination, 1)) rows for a $(length(d))-component Dirichlet",
        ),
    )
    columns = reshape(destination, length(d), :)
    _, next_rng =
        IR._engine_fill_columns!(rng, columns, threaded, _dirichlet_codec(rng, d, recover))
    return destination, next_rng
end

# The type check comes first, since a device may not hold the result type at all.
function _dirichlet_array(rng, d, dims)
    T = Distributions.partype(d)
    IR._check_serviceability(rng, IR._primal_float(T))
    return IR._allocate_draw_array(IR._engine_backend(rng), T, dims)
end

IR._engine_rand_next(rng, d::Distributions.Dirichlet) =
    IR._engine_rand_next!(rng, d, _dirichlet_array(rng, d, (length(d),)))
IR._engine_rand_next(rng, d::Distributions.Dirichlet, n::Integer; threaded::Bool = false) =
    IR._engine_rand_next!(rng, d, _dirichlet_array(rng, d, (length(d), n)); threaded)

# Draw `index` is one column-codec draw, so the engine sees the index whole.
function IR._engine_rand_at(rng, d::Distributions.Dirichlet, index::Integer)
    recover = _validate_dirichlet(d)
    index < 1 && IR._invalid_address_index()
    codec = _dirichlet_codec(rng, d, recover)
    count, width = IR._codec_takes(codec, Distributions.partype(d))
    addressed = IR._addressed_state(rng, count, width, index)
    destination = _dirichlet_array(rng, d, (length(d),))
    IR._engine_fill_columns!(addressed, reshape(destination, :, 1), false, codec)
    return destination
end

IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType} = IR._engine_rand_next!(rng, d, destination; threaded)
Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType} = first(IR._engine_rand_next!(rng, d, destination; threaded))
IR.rand_next(rng::IR._ScalarUniformGenerators, d::_FloatDirichlet) =
    IR._engine_rand_next(rng, d)
Random.rand(rng::IR._ScalarUniformGenerators, d::_FloatDirichlet) =
    first(IR._engine_rand_next(rng, d))
IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::_FloatDirichlet,
    n::Integer;
    threaded::Bool = false,
) = IR._engine_rand_next(rng, d, n; threaded)
Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::_FloatDirichlet,
    n::Integer;
    threaded::Bool = false,
) = first(IR._engine_rand_next(rng, d, n; threaded))
IR.rand_at(rng::IR._ScalarUniformGenerators, d::_FloatDirichlet, index::Integer) =
    IR._engine_rand_at(rng, d, index)
