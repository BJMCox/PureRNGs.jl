# A Dirichlet draw normalizes length(alpha) log-gamma draws in consecutive spans
# by log-sum-exp, so small shapes still sum to one. A draw fills one column, so
# a GPU runs a workitem per draw, with the shapes copied to the device.
const _FloatDirichlet = Distributions.Dirichlet{<:_FloatType}

function _validate_dirichlet(d)
    all(_positive_finite, d.alpha) || throw(ArgumentError("invalid Dirichlet parameters"))
    return nothing
end

@inline _dirichlet_takes(::Type{T}) where {T} =
    (2IR._GAMMA_CANDIDATES + 1, Val(Int(IR._normal_bits(T))))

# Draw `column` of `destination` from the spans past `rng`'s position. The loops
# add in component order, so every backend rounds the same way.
@inline function _dirichlet_column!(destination, column, rng, alpha)
    T = eltype(destination)
    count, width = _dirichlet_takes(T)
    first_span = UInt64(column - 1) * UInt64(length(alpha))
    cursor = IR._fill_cursor(rng, count, width, first_span)
    largest = typemin(T)
    for component in eachindex(alpha)
        component == firstindex(alpha) ||
            (cursor = IR._skip_takes(rng, cursor, count, width))
        shape = alpha[component]
        codec = IR._GammaCodec(shape, one(T), IR._engine_backend(rng), IR._GAMMA_CANDIDATES)
        ordinal = IR._cursor_ordinal(rng, cursor)
        value = IR._gamma_log_value(shape, codec, rng, ordinal, cursor)
        destination[component, column] = value
        largest = max(largest, value)
    end
    total = zero(T)
    for component in eachindex(alpha)
        value = exp(destination[component, column] - largest)
        destination[component, column] = value
        total += value
    end
    for component in eachindex(alpha)
        destination[component, column] /= total
    end
    return destination
end

function _reserve_dirichlet(rng, d, draws, ::Type{T}) where {T}
    count, width = _dirichlet_takes(T)
    takes = UInt64(draws) * UInt64(length(d.alpha)) * UInt64(count)
    return last(IR._draw_cursor(rng, takes, width))
end

function _fill_dirichlet!(rng, d, destination, threaded::Bool)
    columns = reshape(destination, length(d), :)
    backend = IR._engine_backend(rng)
    alpha = IR._transfer_array(backend, d.alpha)
    return IR._foreach_column!(
        IR._fill_backend(backend, destination),
        _dirichlet_column!,
        columns,
        threaded,
        rng,
        alpha,
    )
end

function IR._engine_rand_next!(
    rng,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType}
    _validate_dirichlet(d)
    IR._check_serviceability(rng, T)
    IR._check_fill_device(rng, destination)
    size(destination, 1) == length(d) || throw(
        DimensionMismatch(
            "destination has $(size(destination, 1)) rows for a $(length(d))-component Dirichlet",
        ),
    )
    next_rng = _reserve_dirichlet(rng, d, size(destination, 2), T)
    _fill_dirichlet!(rng, d, destination, threaded)
    return destination, next_rng
end

# The type check comes first, since a device may not hold the result type at all.
function _dirichlet_array(rng, d, dims)
    T = Distributions.partype(d)
    IR._check_serviceability(rng, T)
    return IR._allocate_draw_array(IR._engine_backend(rng), T, dims)
end

IR._engine_rand_next(rng, d::_FloatDirichlet) =
    IR._engine_rand_next!(rng, d, _dirichlet_array(rng, d, (length(d),)))
IR._engine_rand_next(rng, d::_FloatDirichlet, n::Integer; threaded::Bool = false) =
    IR._engine_rand_next!(rng, d, _dirichlet_array(rng, d, (length(d), n)); threaded)

function IR._engine_rand_at(rng, d::_FloatDirichlet, index::Integer)
    _validate_dirichlet(d)
    index < 1 && IR._invalid_address_index()
    T = Distributions.partype(d)
    count, width = _dirichlet_takes(T)
    addressed = IR._addressed_state(rng, count, width, (index - 1) * length(d) + 1)
    _reserve_dirichlet(addressed, d, 1, T)
    destination = _dirichlet_array(rng, d, (length(d),))
    _fill_dirichlet!(addressed, d, destination, false)
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
