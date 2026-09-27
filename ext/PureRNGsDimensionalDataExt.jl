module PureRNGsDimensionalDataExt

import DimensionalData
import PureRNGs
import Random

# DimensionalData's `rand(x, dims::DimTuple)` draws from `x` into an array with
# those dimensions, and PureRNGs' `rand(rng, population)` picks one element of a
# tuple. A PureRNGs generator with a tuple of dimensions matches both, so this
# method settles the call with the PureRNGs pick.
Random.rand(
    rng::PureRNGs._ScalarUniformGenerators,
    population::DimensionalData.Dimensions.DimTuple,
) = first(PureRNGs._rand_next_pick(rng, population))

end
