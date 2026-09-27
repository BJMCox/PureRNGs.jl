# A Dirichlet draw fills one column: it normalizes one log-gamma draw per shape,
# in consecutive Gamma spans, by log-sum-exp, so small shapes still sum to one.
# `alpha` sits on the engine's backend, so a device fill reads it in the kernel.
struct _DirichletCodec{A<:AbstractVector}
    alpha::A
end

# One component's takes. A column is `length(alpha)` components in order, so
# component `c` of column `j` is draw `(j - 1) * length(alpha) + c` of a fill of
# these spans, and an engine can fill the log-gamma matrix one element per draw.
@inline _component_takes(::_DirichletCodec, ::Type{T}) where {T} =
    (2_GAMMA_CANDIDATES + 1, Val(Int(_normal_bits(_primal_float(T)))))

@inline function _codec_takes(codec::_DirichletCodec, ::Type{T}) where {T}
    count, width = _component_takes(codec, T)
    return length(codec.alpha) * count, width
end

# The log-gamma draw of component `component` from the takes at `cursor`. A dual
# shape reaches the AD rules of `_gamma_log_value`.
@inline function _component_log(
    codec::_DirichletCodec,
    rng,
    cursor,
    component::Integer,
    ::Type{T},
) where {T}
    shape = codec.alpha[component]
    gamma = _GammaCodec(shape, one(T), _engine_backend(rng), _GAMMA_CANDIDATES)
    return _gamma_log_value(shape, gamma, rng, _cursor_ordinal(rng, cursor), cursor)
end

# Turns the log-gamma values in column `column` into the Dirichlet draw. The
# loops add in component order, so every backend rounds the same way.
@inline function _normalize_column!(destination, column)
    T = eltype(destination)
    components = axes(destination, 1)
    largest = typemin(T)
    for component in components
        largest = max(largest, destination[component, column])
    end
    total = zero(T)
    for component in components
        value = exp(destination[component, column] - largest)
        destination[component, column] = value
        total += value
    end
    for component in components
        destination[component, column] /= total
    end
    return destination
end

# Writes column `column` of `destination` from the takes at `cursor` and returns
# the cursor past them.
@inline function _column_take!(codec::_DirichletCodec, rng, cursor, destination, column)
    T = eltype(destination)
    count, width = _component_takes(codec, T)
    for component in eachindex(codec.alpha)
        component == firstindex(codec.alpha) ||
            (cursor = _skip_takes(rng, cursor, count, width))
        destination[component, column] = _component_log(codec, rng, cursor, component, T)
    end
    _normalize_column!(destination, column)
    return _skip_takes(rng, cursor, count, width)
end
