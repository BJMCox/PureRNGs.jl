KernelAbstractions.@kernel function _uniform_fill_kernel!(
    rng,
    destination,
    ::Val{T},
) where {T}
    ordinal = @index(Global, Linear)
    indices = eachindex(destination)
    index = @inbounds indices[firstindex(indices)+ordinal-1]
    bits_lo, bits_hi = _bit_span(UInt64(ordinal - 1), _draw_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = _draw_unchecked(rng, position, T)
end

KernelAbstractions.@kernel function _uniform_fill_grouped_kernel!(
    rng,
    destination,
    ::Val{T},
    group::Val{N},
) where {T,N}
    workitem = @index(Global, Linear)
    first = (workitem - 1) * N + 1
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _draw_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_uniform_grouped_unchecked!(rng, position, destination, T, first, group)
end

@inline _cooperative_value(::Val{:uniform}, ::Type{T}, raw) where {T} = _from_bits(T, raw)
@inline _fill_width(::Val{:uniform}, ::Type{T}) where {T} = _draw_bits(T)
@inline _fill_kernel(::Val{:uniform}) = _uniform_fill_kernel!
@inline _fill_grouped_kernel(::Val{:uniform}) = _uniform_fill_grouped_kernel!

@inline _fill_backend(destination) = KernelAbstractions.get_backend(destination)
@inline _fill_backend(destination::BitArray) =
    KernelAbstractions.get_backend(destination.chunks)

@inline function _launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    ::Nothing,
) where {T}
    _fill_kernel(codec)(backend)(rng, destination, Val(T); ndrange = length(destination))
    return destination
end

@inline function _launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    plan::Tuple{Val{:grouped},Val{N}},
) where {T,N}
    group = plan[2]
    workitems = cld(length(destination), _fill_group_size(group))
    _fill_grouped_kernel(codec)(backend)(
        rng,
        destination,
        Val(T),
        group;
        ndrange = workitems,
    )
    return destination
end
