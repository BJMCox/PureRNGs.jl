# A Dirichlet draw fills one column: it normalizes one log-gamma draw per shape,
# in consecutive Gamma spans, by log-sum-exp, so small shapes still sum to one.
# `alpha` sits on the engine's backend, so a device fill reads it in the kernel.
struct _DirichletCodec{A<:AbstractVector}
    alpha::A
end

@inline _codec_takes(codec::_DirichletCodec, ::Type{T}) where {T} = (
    length(codec.alpha) * (2_GAMMA_CANDIDATES + 1),
    Val(Int(_normal_bits(_primal_float(T)))),
)

# Writes column `column` of `destination` from the takes at `cursor` and returns
# the cursor past them. A dual shape reaches the AD rules of `_gamma_log_value`.
# The loops add in component order, so every backend rounds the same way.
@inline function _column_take!(codec::_DirichletCodec, rng, cursor, destination, column)
    T = eltype(destination)
    alpha = codec.alpha
    count, width = 2_GAMMA_CANDIDATES + 1, Val(Int(_normal_bits(_primal_float(T))))
    largest = typemin(T)
    for component in eachindex(alpha)
        component == firstindex(alpha) || (cursor = _skip_takes(rng, cursor, count, width))
        shape = alpha[component]
        gamma = _GammaCodec(shape, one(T), _engine_backend(rng), _GAMMA_CANDIDATES)
        value = _gamma_log_value(shape, gamma, rng, _cursor_ordinal(rng, cursor), cursor)
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
    return _skip_takes(rng, cursor, count, width)
end
