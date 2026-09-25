const _CPUGenerators = _BackendGenerators{_CPUBackend}

@inline _allocate_array(::_CPUBackend, ::Type{T}, dims::Tuple) where {T} =
    Array{T}(undef, dims)

# Host data a draw reads on the generator's device, such as weights or shapes.
_transfer_array(::_CPUBackend, values::Array) = values
_transfer_array(device, values::Array{T}) where {T} =
    copyto!(_allocate_array(device, T, size(values)), values)

@noinline _invalid_dimensions() = throw(ArgumentError("dimensions must be non-negative"))

@inline function _allocate_draw_array(device, ::Type{T}, dims::Tuple) where {T}
    for dim in dims
        dim < 0 && _invalid_dimensions()
    end
    return _allocate_array(device, T, dims)
end

# The `ordinal`-th position of `indices`, counting from one, for destinations
# whose axes do not start at one.
Base.@propagate_inbounds _destination_index(indices, ordinal::Integer) =
    indices[firstindex(indices)+ordinal-1]
