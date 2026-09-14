const SIX_TRANSFORM_TYPES = Union{Logistic,Gumbel,Pareto,Frechet,Cauchy,TriangularDist}

# Distributions defaults eltype to Float64 for several Float32 models.
six_transform_result_type(
    ::Union{Logistic{T},Gumbel{T},Pareto{T},Frechet{T},Cauchy{T},TriangularDist{T}},
) where {T} = T

function six_transform_distributions(::Type{T}) where {T}
    return (
        Logistic(T(0.25), T(0.75)),
        Gumbel(T(0.25), T(0.75)),
        Pareto(T(2), T(3)),
        Frechet(T(2), T(3)),
        Cauchy(T(0.25), T(0.75)),
        TriangularDist(T(-1.5), T(2.25), T(0.25)),
    )
end

# Recover the odd-numerator grid independently from the wider public uniform.
function six_transform_midpoint(rng, ::Type{T}) where {T}
    scale = T === Float32 ? T(0x1p23) : T(0x1p52)
    return (floor(rand(rng, T) * scale) + T(0.5)) / scale
end

function six_transform_input_next(
    rng,
    ::Union{Logistic{T},Gumbel{T},Frechet{T},Cauchy{T}},
) where {T}
    _, next_rng = randn_next(rng, T)
    return six_transform_midpoint(rng, T), next_rng
end
six_transform_input_next(rng, ::Pareto{T}) where {T} = randexp_next(rng, T)
six_transform_input_next(rng, ::TriangularDist{T}) where {T} = rand_next(rng, T)

six_transform_formula(d::Logistic, u, affine = fma) = affine(d.θ, log(u) - log1p(-u), d.μ)
six_transform_formula(d::Cauchy{T}, u, affine = fma) where {T} =
    affine(d.σ, tanpi(u - T(0.5)), d.μ)
six_transform_formula(d::Gumbel{T}, u, affine = fma) where {T} =
    affine(-d.θ, log(-log(one(T) - u)), d.μ)
six_transform_formula(d::Frechet{T}, u, affine = fma) where {T} =
    d.θ * (-log(one(T) - u))^(-inv(d.α))
six_transform_formula(d::Pareto, e, affine = fma) = d.θ * exp(e / d.α)
function six_transform_formula(d::TriangularDist{T}, u, affine = fma) where {T}
    d.a == d.b && return d.a
    p = (d.c - d.a) / (d.b - d.a)
    lower = affine(d.c - d.a, sqrt(u / p), d.a)
    upper = affine(d.c - d.b, sqrt((one(T) - u) / (one(T) - p)), d.b)
    return ifelse(iszero(u), d.a, ifelse(u <= p, lower, upper))
end
