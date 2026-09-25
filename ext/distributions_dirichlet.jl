# A Dirichlet draw normalizes length(alpha) log-gamma draws in consecutive spans
# by log-sum-exp, so small shapes still sum to one. A draw fills one column, so
# a GPU runs a workitem per draw, with the shapes copied to the device.
const _FloatDirichlet = Distributions.Dirichlet{<:_FloatType}

@inline _dirichlet_spans(d, draws) = UInt128(draws) * UInt128(length(d.alpha))

function _validate_dirichlet(d)
    all(_positive_finite, d.alpha) || throw(ArgumentError("invalid Dirichlet parameters"))
    return nothing
end

# Draw `column` of `destination` from the spans past `rng`'s position. The loops
# add in component order, so every backend rounds the same way.
@inline function _dirichlet_column!(destination, column, rng, alpha)
    T = eltype(destination)
    span = IR._gamma_span(T, IR._GAMMA_CANDIDATES)
    first_span = UInt64(column - 1) * UInt64(length(alpha))
    largest = typemin(T)
    for component in eachindex(alpha)
        shape = alpha[component]
        codec = IR._GammaCodec(shape, one(T), rng.device, IR._GAMMA_CANDIDATES)
        bits_lo, bits_hi = IR._bit_span(first_span + UInt64(component - 1), span)
        position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
        cursor = IR._gamma_cursor(rng, position)
        value = IR._gamma_log_value(shape, codec, rng, position, cursor)
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
    bits = _dirichlet_spans(d, draws) * UInt128(IR._gamma_span(T, IR._GAMMA_CANDIDATES))
    return IR._reserve(rng, bits % UInt64, (bits >> 64) % UInt64)
end

function _fill_dirichlet!(rng, d, destination, threaded::Bool)
    columns = reshape(destination, length(d), :)
    alpha = IR._transfer_array(rng.device, d.alpha)
    backend = IR._fill_backend(rng.device, destination)
    return IR._foreach_column!(backend, _dirichlet_column!, columns, threaded, rng, alpha)
end

function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
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
Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType} = first(IR.rand_next!(rng, d, destination; threaded))

# The type check comes first, since a device may not hold the result type at all.
function _dirichlet_array(rng, d, dims)
    T = Distributions.partype(d)
    IR._check_serviceability(rng, T)
    return IR._allocate_draw_array(rng.device, T, dims)
end

IR.rand_next(rng::IR._ScalarUniformGenerators, d::_FloatDirichlet) =
    IR.rand_next!(rng, d, _dirichlet_array(rng, d, (length(d),)))
Random.rand(rng::IR._ScalarUniformGenerators, d::_FloatDirichlet) =
    first(IR.rand_next(rng, d))
function IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::_FloatDirichlet,
    n::Integer;
    threaded::Bool = false,
)
    return IR.rand_next!(rng, d, _dirichlet_array(rng, d, (length(d), n)); threaded)
end
Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::_FloatDirichlet,
    n::Integer;
    threaded::Bool = false,
) = first(IR.rand_next(rng, d, n; threaded))

function IR.rand_at(rng::IR._ScalarUniformGenerators, d::_FloatDirichlet, index::Integer)
    _validate_dirichlet(d)
    index < 1 && IR._invalid_address_index()
    T = Distributions.partype(d)
    span = IR._gamma_span(T, IR._GAMMA_CANDIDATES)
    addressed = IR._addressed_rng(rng, span, (index - 1) * length(d) + 1)
    _reserve_dirichlet(addressed, d, 1, T)
    destination = _dirichlet_array(rng, d, (length(d),))
    _fill_dirichlet!(addressed, d, destination, false)
    return destination
end
