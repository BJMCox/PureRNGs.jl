# Sampling

## Integer ranges

```@example sampling
using PureRNGs, Random

rng = Philox4x32(42)
rng, die = rand_next(rng, 1:6)
rng, evens = rand_next(rng, 2:2:20, 5)
rng, descending = rand_next(rng, 10:-1:1, 5)
evens
```

Ranges support signed and unsigned integer element types from 8 through 64 bits.

Range sampling uses a fixed-width multiply-high mapping, not rejection sampling.
Arbitrary range lengths can have a small finite mapping bias. Preimage counts differ by at most one.

The candidate uses 64 bits for lengths up to 2^32 and 128 bits for larger lengths.

## Sample a population

```@example sampling
population = [:red, :green, :blue]
rng, draws = randsample_next(rng, population, 6)
same_length = randsample(rng, population)
@assert length(same_length) == length(population)
draws
```

Sampling is always **with replacement**. Omitting the count draws as many values as the population contains.
It does not shuffle the population.

Every call returns a vector. Integer ranges have a direct path without materializing the population.

```@example sampling
rng, indices = randsample_next(rng, 1:1_000_000, 8)
indices
```

Arrays use the order of `CartesianIndices(axes(population))`.
This matches ordinary column-major order and also defines positions for custom axes.

Finite non-array iterators are materialized. Do not pass infinite iterators.

## Supply weights

```@example sampling
weights = [1.0, 3.0, 0.0]
rng, draws = randsample_next(rng, population, weights, 12)
@assert all(!=(:blue), draws)
draws
```

Pass a plain vector of real weights. No weight wrapper is needed.

Weights follow population positions. They must convert to finite, nonnegative `Float64` values with a finite, positive left-fold total.
Zero weights exclude elements. Zero requested samples still require valid weights.

A weighted batch sorts its thresholds once, sweeps the weights, then restores draw order.
The preparation is shared within that call, not cached across calls.

## Keep data on the right device

An array population must match the generator's backend. The same rule applies to array weights.
A host array with a CUDA-bound generator throws instead of silently copying.

Device-agnostic inputs, such as integer ranges, can serve a device-bound generator.
GPU populations must also support device indexing and storage.

Results stay on the generator's backend. Weighted validation can require a small host result.
Weighted sampling does not have the same performance profile as primitive fills.
