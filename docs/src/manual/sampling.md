# Sampling

## Integer ranges

```@example sampling
using PureRNGs, Random

rng = Philox4x32(42)
die, rng = rand_next(rng, 1:6)
evens, rng = rand_next(rng, 2:2:20, 5)
descending, rng = rand_next(rng, 10:-1:1, 5)
evens
```

Ranges support signed and unsigned integer element types from 8 through 64 bits.

Range sampling uses a fixed-width multiply-high mapping, not rejection sampling.
Arbitrary range lengths can have a small finite mapping bias. Preimage counts differ by at most one.

The candidate uses 64 bits for spans up to 2^32 and 128 bits for larger spans.
For a span `s` and a candidate width `K`, the relative probability imbalance
between two values is exactly `1/floor(2^K / s)`. It is zero when `s` divides
`2^K`, so powers of two are unbiased. It is largest just below each width
change: about 2.33e-10 at `s = 2^32 - 1`, and about 5.42e-20 at
`s = 2^64 - 1`.

```@example sampling
dice = Vector{Int}(undef, 16)
_, rng = rand_next!(rng, dice, 1:6; threaded=false)
dice
```

The range follows the destination.

`rand_at(rng, range, i)` returns the `i`th range draw without advancing `rng`.

## Sample a population

```@example sampling
population = [:red, :green, :blue]
draws, rng = randsample_next(rng, population, 6)
same_length = randsample(rng, population)
@assert length(same_length) == length(population)
draws
```

Sampling is always **with replacement**. Omitting the count draws as many values as the population contains.
It does not shuffle the population. The allocating forms return a vector.

Integer ranges have a direct allocating path without materializing the population.

```@example sampling
indices, rng = randsample_next(rng, 1:1_000_000, 8)
indices
```

Arrays use the order of `CartesianIndices(axes(population))`.
This matches ordinary column-major order and also defines positions for custom axes.

Finite non-array iterators are materialized. Do not pass infinite iterators.

## Supply weights

```@example sampling
weights = [1.0, 3.0, 0.0]
draws, rng = randsample_next(rng, population, weights, 12)
@assert all(!=(:blue), draws)
draws
```

Pass a plain vector of real weights.

Weights follow population positions. They must convert to finite, nonnegative `Float64` values with a finite, positive left-fold total.
Zero weights exclude elements. Zero requested samples still require valid weights.

On CPU, a weighted batch builds its exact cumulative `Float64` weights once per call and
looks up samples in draw order. Other backends may sort thresholds for a batch.
The preparation is shared within that call, not cached across calls. It allocates
weighted scratch space, so an in-place weighted fill does not promise zero
allocations. CPU weighted fills follow the `threaded` keyword like the other
fills.

```julia
table = WeightTable([1.0, 3.0, 0.0])
draws, rng = randsample_next(rng, population, table, 12)
```

Build a `WeightTable` once to reuse the cumulative table across calls. It holds
CPU data and is accepted wherever a weight vector is.

Each category's realized share is a whole number of `2^-53` cells of the
cumulative total. Shares below about `1e-16` of the total are not represented
faithfully and depend on weight order.

## Fill an existing destination

```@example sampling
expected, next_rng = randsample_next(rng, population, weights, 12)
destination = similar(expected)
_, rng = randsample_next!(rng, population, weights, destination; threaded=false)
@assert destination == expected
@assert rng == next_rng
destination
```

`randsample!` returns the identical destination, while `randsample_next!`
returns `(destination, next_rng)`. The destination length determines the draw
count; its axes are preserved and values are written in its native `eachindex`
order. Its element type must exactly equal the prepared population element
type.

All inputs, including weights and the complete random span, are validated
before the destination is changed. Empty destinations still validate the
population and weights, then consume no bits. A destination that might alias
the population is rejected. It may alias weights only after those weights have
been validated and privately prepared. On CPU, `threaded=false` uses the
calling task directly; it does not look up or launch a backend.

## Keep data on the right device

An array population must match the generator's backend. The same rule applies to array weights.
A host array with a CUDA-bound generator throws instead of silently copying.

Device-agnostic inputs, such as integer ranges, can serve a device-bound generator.
GPU populations must also support device indexing and storage.

Results stay on the generator's backend. Weighted validation can require a small host result.
Weighted sampling does not have the same performance profile as primitive fills.

## Design limits

Two alternatives were considered and not taken.

There is no rejection sampler for ranges. Rejection removes the mapping bias,
but it costs a variable number of draws per value. That breaks the fixed work
per value, addressed draws, and tuned GPU fills. Detecting the bias it would
remove takes about 2^64 draws.

There is no wider grid for normal draws. A 64-bit grid would raise the
`Float64` normal cap from 8.2095 to 9.1553 for 23% more bits per draw, and it
would give up the exact lattice of the current grid.
