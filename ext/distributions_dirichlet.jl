# A Dirichlet draw is a column of the core's Dirichlet codec, so an engine's
# column hook serves it. The bodies take float and dual shapes alike; the
# ForwardDiff extension forwards the dual ones.
const _FloatDirichlet = Distributions.Dirichlet{<:_FloatType}

function _validate_dirichlet(d)
    all(_positive_finite, d.alpha) || throw(ArgumentError("invalid Dirichlet parameters"))
    return nothing
end

_dirichlet_codec(rng, d) =
    IR._DirichletCodec(IR._transfer_array(IR._engine_backend(rng), d.alpha))

function IR._engine_rand_next!(
    rng,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:Real}
    _validate_dirichlet(d)
    IR._check_serviceability(rng, IR._primal_float(T))
    IR._check_fill_device(rng, destination)
    size(destination, 1) == length(d) || throw(
        DimensionMismatch(
            "destination has $(size(destination, 1)) rows for a $(length(d))-component Dirichlet",
        ),
    )
    columns = reshape(destination, length(d), :)
    _, next_rng = IR._engine_fill_columns!(rng, columns, threaded, _dirichlet_codec(rng, d))
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
    _validate_dirichlet(d)
    index < 1 && IR._invalid_address_index()
    codec = _dirichlet_codec(rng, d)
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
