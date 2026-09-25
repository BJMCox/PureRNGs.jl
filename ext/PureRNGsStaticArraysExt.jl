module PureRNGsStaticArraysExt

# A static array of `N` elements of type `T` is the next `N` scalar draws of `T`
# in linear order, so it equals a length-`N` fill, and an array of static arrays
# is the fill of its `reinterpret` as `T`.

import PureRNGs
import Random
using StaticArrays: StaticArray

const IR = PureRNGs

@inline _element(::Type{SA}) where {SA<:StaticArray} = _element(eltype(SA), SA)
@inline _element(::Type{T}, ::Type) where {T} = T
@noinline _element(::Type{Any}, ::Type{SA}) where {SA} = throw(
    ArgumentError(
        "untyped immutable draws are forbidden: $SA has no element type; use a type such as $SA{Float64}",
    ),
)

# The draws are unrolled over `N`, so they allocate nothing and run in a kernel.
@inline _chain(draw, rng, ::Val{0}) = ((), rng)
@inline function _chain(draw, rng, ::Val{N}) where {N}
    value, rng = draw(rng)
    rest, rng = _chain(draw, rng, Val(N - 1))
    return (value, rest...), rng
end

@inline function _static_next(draw, rng, ::Type{SA}) where {SA<:StaticArray}
    values, next_rng = _chain(draw, rng, Val(length(SA)))
    return SA(values), next_rng
end

# Static array `i` starts at scalar draw `(i - 1) * N + 1`.
@inline function _static_address(rng, width::UInt16, n::Int, i::Integer)
    i < 1 && IR._invalid_address_index()
    offset = Base.Checked.checked_mul(UInt64(i) - one(UInt64), UInt64(n))
    return IR._addressed_rng(rng, width, Base.Checked.checked_add(offset, one(UInt64)))
end

@inline function _static_array_next(
    fill_next!,
    rng,
    ::Type{SA},
    dims::Dims,
    threaded,
) where {SA}
    destination = IR._allocate_array(rng.device, SA, dims)
    return destination, last(fill_next!(rng, destination; threaded))
end

for (held, fill!, next, next!, at, width) in (
    (:rand, :rand!, :rand_next, :rand_next!, :rand_at, :_draw_bits),
    (:randn, :randn!, :randn_next, :randn_next!, :randn_at, :_normal_bits),
    (:randexp, :randexp!, :randexp_next, :randexp_next!, :randexp_at, :_exponential_bits),
)
    @eval begin
        @inline IR.$next(
            rng::IR._ScalarUniformGenerators,
            ::Type{SA},
        ) where {SA<:StaticArray} =
            _static_next(current -> IR.$next(current, _element(SA)), rng, SA)
        @inline Random.$held(
            rng::IR._ScalarUniformGenerators,
            ::Type{SA},
        ) where {SA<:StaticArray} = first(IR.$next(rng, SA))
        @inline function IR.$at(
            rng::IR._ScalarUniformGenerators,
            ::Type{SA},
            i::Integer,
        ) where {SA<:StaticArray}
            width = IR.$width(_element(SA))
            return first(IR.$next(_static_address(rng, width, length(SA), i), SA))
        end

        @inline IR.$next(
            rng::IR._ScalarUniformGenerators,
            ::Type{SA},
            dims::Dims;
            threaded::Bool = false,
        ) where {SA<:StaticArray} = _static_array_next(IR.$next!, rng, SA, dims, threaded)
        @inline IR.$next(
            rng::IR._ScalarUniformGenerators,
            ::Type{SA},
            dim1::Integer,
            dims::Integer...;
            threaded::Bool = false,
        ) where {SA<:StaticArray} =
            _static_array_next(IR.$next!, rng, SA, (dim1, dims...), threaded)
        @inline Random.$held(
            rng::IR._ScalarUniformGenerators,
            ::Type{SA},
            dims::Dims;
            threaded::Bool = false,
        ) where {SA<:StaticArray} = first(IR.$next(rng, SA, dims; threaded))
        @inline Random.$held(
            rng::IR._ScalarUniformGenerators,
            ::Type{SA},
            dim1::Integer,
            dims::Integer...;
            threaded::Bool = false,
        ) where {SA<:StaticArray} = first(IR.$next(rng, SA, (dim1, dims...); threaded))
        @inline function IR.$at(
            rng::IR._ScalarUniformGenerators,
            ::Type{SA},
            indices::AbstractUnitRange{<:Integer};
            threaded::Bool = false,
        ) where {SA<:StaticArray}
            isempty(indices) && return IR._allocate_array(rng.device, SA, (0,))
            width = IR.$width(_element(SA))
            addressed = _static_address(rng, width, length(SA), first(indices))
            return first(IR.$next(addressed, SA, length(indices); threaded))
        end

        @inline function IR.$next!(
            rng::IR._ScalarUniformGenerators,
            destination::AbstractArray{SA};
            threaded::Bool = false,
        ) where {SA<:StaticArray}
            elements = reinterpret(_element(SA), destination)
            return destination, last(IR.$next!(rng, elements; threaded))
        end
        @inline Random.$fill!(
            rng::IR._ScalarUniformGenerators,
            destination::AbstractArray{SA};
            threaded::Bool = false,
        ) where {SA<:StaticArray} = first(IR.$next!(rng, destination; threaded))
    end
end

# StaticArrays draws from a collection or distribution the same way.
@inline IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    X,
    ::Type{SA},
) where {SA<:StaticArray} = _static_next(current -> IR.rand_next(current, X), rng, SA)
@inline Random.rand(
    rng::IR._ScalarUniformGenerators,
    X,
    ::Type{SA},
) where {SA<:StaticArray} = first(IR.rand_next(rng, X, SA))

end
