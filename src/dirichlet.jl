# A Dirichlet draw fills one column: it normalizes one log-gamma draw per shape,
# in consecutive Gamma spans, by log-sum-exp, so small shapes still sum to one.
# `alpha` sits on the engine's backend, so a device fill reads it in the kernel.
struct _DirichletCodec{A<:AbstractVector,Recover}
    alpha::A
end

_DirichletCodec(alpha::A, ::Val{Recover} = Val(true)) where {A,Recover} =
    _DirichletCodec{A,Recover}(alpha)
@inline _recover_log_overflow(::_DirichletCodec{A,Recover}) where {A,Recover} = Recover

@inline function _normalize_fill!(destination, column, rng, codec::_DirichletCodec)
    largest, best = _column_maximum(destination, column)
    if _recover_log_overflow(codec) && isinf(largest) && largest < zero(largest)
        count, width = _codec_takes(codec, eltype(destination))
        cursor = _fill_cursor(rng, count, width, UInt64(column - 1))
        _normalize_tiny_column!(destination, column, codec, rng, cursor)
    else
        _normalize_column!(destination, column, largest, best)
    end
    return nothing
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
@inline function _column_maximum(destination, column)
    T = eltype(destination)
    components = axes(destination, 1)
    largest = typemin(T)
    best = first(components)
    for component in components
        value = destination[component, column]
        if value > largest
            largest, best = value, component
        end
    end
    return largest, best
end

@inline _normalize_column!(destination, column) =
    _normalize_column!(destination, column, _column_maximum(destination, column)...)

@inline function _normalize_column!(destination, column, largest, best)
    T = eltype(destination)
    components = axes(destination, 1)
    total = zero(T)
    for component in components
        value = component == best ? one(T) : exp(destination[component, column] - largest)
        # A rounded zero has zero tangent even when the log derivative overflows.
        iszero(_primal_value(value)) && (value = zero(T))
        destination[component, column] = value
        total += value
    end
    for component in components
        destination[component, column] /= total
    end
    return destination
end

@inline function _normalize_column!(destination, column, codec, rng, cursor)
    largest, best = _column_maximum(destination, column)
    _recover_log_overflow(codec) &&
        isinf(largest) &&
        largest < zero(largest) &&
        return _normalize_tiny_column!(destination, column, codec, rng, cursor)
    return _normalize_column!(destination, column, largest, best)
end

# Only the all-overflow column needs another pass over its held bits. Compare
# the scaled boost terms before subtracting log-Gamma values, then normalize
# relative to the winning component. No extra array or parent bits are needed.
@noinline function _normalize_tiny_column!(destination, column, codec, rng, first_cursor)
    T = eltype(destination)
    count, width = _component_takes(codec, T)
    gamma(component) =
        _GammaCodec(codec.alpha[component], one(T), _engine_backend(rng), _GAMMA_CANDIDATES)
    best = firstindex(codec.alpha)
    best_gamma, best_cursor = gamma(best), first_cursor
    cursor = first_cursor
    for component in eachindex(codec.alpha)
        current = gamma(component)
        delta = _tiny_gamma_log_difference(current, best_gamma, rng, cursor, best_cursor)
        if delta > zero(delta)
            best, best_gamma, best_cursor = component, current, cursor
        end
        cursor = _skip_takes(rng, cursor, count, width)
    end
    cursor = first_cursor
    total = zero(T)
    for component in eachindex(codec.alpha)
        delta = _tiny_gamma_log_difference(
            gamma(component),
            best_gamma,
            rng,
            cursor,
            best_cursor,
        )
        value = component == best ? one(T) : isinf(delta) ? zero(T) : exp(delta)
        destination[component, column] = value
        total += value
        cursor = _skip_takes(rng, cursor, count, width)
    end
    for component in eachindex(codec.alpha)
        destination[component, column] /= total
    end
    return destination
end

# Writes column `column` of `destination` from the takes at `cursor` and returns
# the cursor past them.
@inline function _column_take!(codec::_DirichletCodec, rng, cursor, destination, column)
    T = eltype(destination)
    count, width = _component_takes(codec, T)
    first_cursor = cursor
    for component in eachindex(codec.alpha)
        component == firstindex(codec.alpha) ||
            (cursor = _skip_takes(rng, cursor, count, width))
        destination[component, column] = _component_log(codec, rng, cursor, component, T)
    end
    _normalize_column!(destination, column, codec, rng, first_cursor)
    return _skip_takes(rng, cursor, count, width)
end
