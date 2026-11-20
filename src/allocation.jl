const _CPUGenerators = _BackendGenerators{_CPUBackend}

@inline _allocate_array(::_CPUBackend, ::Type{T}, dims::Tuple) where {T} =
    Array{T}(undef, dims)

# Host data a draw reads on the generator's device, such as weights or shapes.
_transfer_array(::_CPUBackend, values::Array) = values
_transfer_array(device, values::Array{T}) where {T} =
    copyto!(_allocate_array(device, T, size(values)), values)

@noinline _invalid_dimensions() = throw(ArgumentError("dimensions must be non-negative"))

@inline function _check_draw_dimensions(dims::Tuple)
    for dim in dims
        dim < 0 && _invalid_dimensions()
    end
    return nothing
end

@inline function _allocate_draw_array(device, ::Type{T}, dims::Tuple) where {T}
    _check_draw_dimensions(dims)
    return _allocate_array(device, T, dims)
end

# The `ordinal`-th position of `indices`, counting from one, for destinations
# whose axes do not start at one.
Base.@propagate_inbounds _destination_index(indices, ordinal::Integer) =
    indices[firstindex(indices)+ordinal-1]

# Plain ranges make view axes one-based without copying the storage.
@inline function _one_based_view(array::AbstractArray)
    Base.has_offset_axes(array) || return array
    return view(array, map(axis -> first(axis):last(axis), axes(array))...)
end
