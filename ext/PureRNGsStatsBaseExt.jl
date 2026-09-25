module PureRNGsStatsBaseExt

# Pure generators are not `AbstractRNG`s, so StatsBase's own methods never see
# them. Each method here is the `randsample` draw at the held position; like
# every non-continuation form, it does not advance the generator.

using PureRNGs: PureRNGs, AbstractPureRNG, rand_next, randsample
using StatsBase: StatsBase, AbstractWeights, UnitWeights, sample, weights

_randsample(rng, a, ::Nothing, n, replace) = randsample(rng, a, n; replace)
_randsample(rng, a, wv, n, replace) = randsample(rng, a, wv, n; replace)

# StatsBase treats unit weights as no weights, so they take the unweighted law.
function _randsample(rng, a, wv::UnitWeights, n, replace)
    length(wv) == length(a) || PureRNGs._weight_length_error(length(wv), length(a))
    return randsample(rng, a, n; replace)
end

# An ordered sample lists its elements in population order. Sampling positions
# consumes what sampling the elements does, so it keeps the unordered law.
_sample(rng, a, wv, n, replace::Bool, ordered::Bool) =
    ordered ? a[sort!(_randsample(rng, LinearIndices(a), wv, n, replace))] :
    _randsample(rng, a, wv, n, replace)

StatsBase.sample(rng::AbstractPureRNG, a::AbstractArray) = rand(rng, a)
StatsBase.sample(rng::AbstractPureRNG, wv::AbstractWeights) =
    only(_randsample(rng, Base.OneTo(length(wv)), wv, 1, true))
StatsBase.sample(rng::AbstractPureRNG, a::AbstractArray, wv::AbstractWeights) =
    only(_randsample(rng, a, wv, 1, true))

StatsBase.sample(
    rng::AbstractPureRNG,
    a::AbstractArray,
    n::Integer;
    replace::Bool = true,
    ordered::Bool = false,
) = _sample(rng, a, nothing, n, replace, ordered)
StatsBase.sample(
    rng::AbstractPureRNG,
    a::AbstractArray,
    wv::AbstractWeights,
    n::Integer;
    replace::Bool = true,
    ordered::Bool = false,
) = _sample(rng, a, wv, n, replace, ordered)
StatsBase.sample(
    rng::AbstractPureRNG,
    a::AbstractArray,
    dims::Dims;
    replace::Bool = true,
    ordered::Bool = false,
) = reshape(_sample(rng, a, nothing, prod(dims), replace, ordered), dims)
StatsBase.sample(
    rng::AbstractPureRNG,
    a::AbstractArray,
    wv::AbstractWeights,
    dims::Dims;
    replace::Bool = true,
    ordered::Bool = false,
) = reshape(_sample(rng, a, wv, prod(dims), replace, ordered), dims)

# StatsBase converts into any destination element type, so the sample is drawn
# and then copied.
StatsBase.sample!(
    rng::AbstractPureRNG,
    a::AbstractArray,
    x::AbstractArray;
    replace::Bool = true,
    ordered::Bool = false,
) = copyto!(x, _sample(rng, a, nothing, length(x), replace, ordered))
StatsBase.sample!(
    rng::AbstractPureRNG,
    a::AbstractArray,
    wv::AbstractWeights,
    x::AbstractArray;
    replace::Bool = true,
    ordered::Bool = false,
) = copyto!(x, _sample(rng, a, wv, length(x), replace, ordered))

StatsBase.wsample(rng::AbstractPureRNG, w::AbstractVector{<:Real}) = sample(rng, weights(w))
StatsBase.wsample(rng::AbstractPureRNG, a::AbstractArray, w::AbstractVector{<:Real}) =
    sample(rng, a, weights(w))
StatsBase.wsample(
    rng::AbstractPureRNG,
    a::AbstractArray,
    w::AbstractVector{<:Real},
    n::Union{Integer,Dims};
    replace::Bool = true,
    ordered::Bool = false,
) = sample(rng, a, weights(w), n; replace, ordered)
StatsBase.wsample!(
    rng::AbstractPureRNG,
    a::AbstractArray,
    w::AbstractVector{<:Real},
    x::AbstractArray;
    replace::Bool = true,
    ordered::Bool = false,
) = StatsBase.sample!(rng, a, weights(w), x; replace, ordered)

# Two range draws, as StatsBase makes them: `j` from `1:n-1` stands for `n` when
# it equals `i`, which makes every ordered pair of distinct values equally likely.
function StatsBase.samplepair(rng::AbstractPureRNG, n::Integer)
    i, rng = rand_next(rng, one(n):n)
    j = rand(rng, one(n):(n-one(n)))
    return i, ifelse(j == i, n, j)
end
function StatsBase.samplepair(rng::AbstractPureRNG, a::AbstractArray)
    i, j = StatsBase.samplepair(rng, length(a))
    return a[firstindex(a)+i-1], a[firstindex(a)+j-1]
end

end
