# A Dirichlet draw normalizes length(alpha) log-gamma draws in consecutive spans
# by log-sum-exp, so small shapes still sum to one. The vector result runs on the
# CPU, as MvNormal draws do.
const _FloatDirichlet = Distributions.Dirichlet{<:_FloatType}

@inline _dirichlet_spans(d, draws) = UInt128(draws) * UInt128(length(d.alpha))

function _validate_dirichlet(d)
    all(_positive_finite, d.alpha) || throw(ArgumentError("invalid Dirichlet parameters"))
    return nothing
end

# Draw `ordinal` (one-based) past `rng`'s position into `destination`.
function _dirichlet_draw!(destination, rng, d, ordinal::Integer)
    T = eltype(destination)
    span = IR._gamma_span(T, IR._GAMMA_CANDIDATES)
    first_span = UInt64(ordinal - 1) * UInt64(length(d.alpha))
    for (component, shape) in enumerate(d.alpha)
        codec = IR._GammaCodec(shape, one(T), rng.device, IR._GAMMA_CANDIDATES)
        bits_lo, bits_hi = IR._bit_span(first_span + UInt64(component - 1), span)
        position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
        destination[component] = IR._gamma_log_value(
            shape,
            codec,
            rng,
            position,
            IR._gamma_cursor(rng, position),
        )
    end
    largest = maximum(destination)
    destination .= exp.(destination .- largest)
    destination ./= sum(destination)
    return destination
end

function _reserve_dirichlet(rng, d, draws, ::Type{T}) where {T}
    bits = _dirichlet_spans(d, draws) * UInt128(IR._gamma_span(T, IR._GAMMA_CANDIDATES))
    return IR._reserve(rng, bits % UInt64, (bits >> 64) % UInt64)
end

function IR.rand_next!(
    rng::IR._CPUGenerators,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType}
    _validate_dirichlet(d)
    size(destination, 1) == length(d) || throw(
        DimensionMismatch(
            "destination has $(size(destination, 1)) rows for a $(length(d))-component Dirichlet",
        ),
    )
    draws = size(destination, 2)
    next_rng = _reserve_dirichlet(rng, d, draws, T)
    columns = eachcol(reshape(destination, length(d), draws))
    if threaded
        Threads.@threads for ordinal = 1:draws
            _dirichlet_draw!(columns[ordinal], rng, d, ordinal)
        end
    else
        for ordinal = 1:draws
            _dirichlet_draw!(columns[ordinal], rng, d, ordinal)
        end
    end
    return destination, next_rng
end
Random.rand!(
    rng::IR._CPUGenerators,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType} = first(IR.rand_next!(rng, d, destination; threaded))

IR.rand_next(rng::IR._CPUGenerators, d::_FloatDirichlet) =
    IR.rand_next!(rng, d, Vector{Distributions.partype(d)}(undef, length(d)))
Random.rand(rng::IR._CPUGenerators, d::_FloatDirichlet) = first(IR.rand_next(rng, d))
IR.rand_next(
    rng::IR._CPUGenerators,
    d::_FloatDirichlet,
    n::Integer;
    threaded::Bool = false,
) = IR.rand_next!(rng, d, Matrix{Distributions.partype(d)}(undef, length(d), n); threaded)
Random.rand(
    rng::IR._CPUGenerators,
    d::_FloatDirichlet,
    n::Integer;
    threaded::Bool = false,
) = first(IR.rand_next(rng, d, n; threaded))

function IR.rand_at(rng::IR._CPUGenerators, d::_FloatDirichlet, index::Integer)
    _validate_dirichlet(d)
    index < 1 && IR._invalid_address_index()
    T = Distributions.partype(d)
    span = IR._gamma_span(T, IR._GAMMA_CANDIDATES)
    addressed = IR._addressed_rng(rng, span, (index - 1) * length(d) + 1)
    _reserve_dirichlet(addressed, d, 1, T)
    return _dirichlet_draw!(Vector{T}(undef, length(d)), addressed, d, 1)
end
