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

Ranges support signed and unsigned integer element types from 8 through 128 bits.
A range with more than 2^64 values reduces a 192-bit candidate; the 128-bit element types run on the CPU only.

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
_, rng = rand_next!(rng, dice, 1:6)
dice
```

The range follows the destination.

`rand_at(rng, range, i)` returns the `i`th range draw without advancing `rng`, and `rand_at(rng, collection, i)` the `i`th pick.

## Pick from a collection

```@example sampling
color, rng = rand_next(rng, [:red, :green, :blue])
letters, rng = rand_next(rng, "abc", 4)
half_steps, rng = rand_next(rng, 0.0:0.5:2.0, 3)
letters
```

A pick accepts any array or range, a tuple, a string, a dict, or a set, as `rand` does in `Random`.
It consumes and returns exactly what `randsample_next(rng, collection, 1)` does, and the array forms equal `randsample_next`.
Strings, dicts, and sets reach the drawn position by iteration, so one pick costs time linear in their length.
A tuple of `Int` passed to `rand_next` is a shape, not a collection.

`rand_next(rng, Char)` is uniform over the 1,112,064 Unicode scalar values, as in `Random`.
It draws the offset `k` from `0:0x10f7ff` and skips the surrogates: `k < 0xd800 ? Char(k) : Char(k + 0x800)`.

## Sample a population

```@example sampling
population = [:red, :green, :blue]
draws, rng = randsample_next(rng, population, 6)
same_length = randsample(rng, population)
@assert length(same_length) == length(population)
draws
```

Sampling is **with replacement** unless you pass `replace = false`. Omitting the count draws as many values as the population contains.
The allocating forms return a vector.

Integer ranges have a direct allocating path without materializing the population.

```@example sampling
indices, rng = randsample_next(rng, 1:1_000_000, 8)
indices
```

Arrays use the order of `CartesianIndices(axes(population))`.
This matches ordinary column-major order and also defines positions for custom axes.

Finite non-array iterators are materialized. Do not pass infinite iterators.

## Shuffle and sample without replacement

```@example sampling
order, rng = randperm_next(rng, 6)
deck, rng = shuffle_next(rng, collect(1:10))
hand, rng = randsample_next(rng, 1:52, 5; replace = false)
hand
```

A permutation draws one uniform `UInt64` key per element at the held position and orders the elements by key, ties by index.
It consumes 64 bits per element, and it gives the same result on the CPU and on a GPU.
Equal keys have probability about `n^2 / 2^65`. They are shuffled within their run with draws from `subrng(rng, key)`, so the permutation stays exactly uniform and the consumption stays fixed.

`shuffle_next` moves elements in linear order by that permutation.
`randcycle_next` sends `p[i]` to `p[i + 1]`, which gives a uniform cyclic permutation.
`randsample(...; replace = false)` returns the leading elements of the shuffled population, so it consumes 64 bits per population element whatever the count.
`Random.randperm`, `randcycle`, `shuffle`, and their in-place forms accept a generator too, and `StatefulRNG` uses the same law.

The keys are ordered by a counting pass over their top bits, in expected linear time, on the CPU and on a GPU.
A GPU permutation of 2^24 elements takes 13.7 ms on an A100, against 104 ms for CUDA.jl's `sortperm!`.

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

Each weighted batch builds its cumulative `Float64` weights once per call and
selects samples by binary search, preserving draw order on CPU and GPU.
The preparation is shared within that call, not cached across calls. It allocates
weighted scratch space, so an in-place weighted fill does not promise zero
allocations. CPU weighted fills follow the `threaded` keyword like the other
fills.

```julia
table = WeightTable([1.0, 3.0, 0.0])
draws, rng = randsample_next(rng, population, table, 12)
```

Build a `WeightTable` once to reuse the cumulative table across calls.
A table of device weights folds on their device and serves generators on that device.

Each category's realized share is a whole number of `2^-53` cells of the
cumulative total. Shares below about `1e-16` of the total are not represented
faithfully and depend on weight order.

### Weighted sampling without replacement

```@example sampling
finalists, rng = randsample_next(rng, [:ann, :bo, :cy, :di], [4.0, 1.0, 2.0, 0.0], 2; replace = false)
finalists
```

With `replace = false`, each element gets one exponential draw `E`, and the sample is the population in increasing order of `E / w`, ties by index (Efraimidis and Spirakis, 2006).
The first element is `i` with probability `w[i] / sum(w)`, the next one is drawn in proportion to the remaining weights, and so on.
The sample consumes 52 bits per population element whatever the count, and the count may not exceed the number of positive weights.
The keys cover the whole `Float64` weight range without overflow, and no weight total is formed, so the total need not be finite.
Pass the weight vector: a `WeightTable` keeps only cumulative sums.
Device exponentials can differ from CPU exponentials in the last ulp, so a device sample can differ from the CPU sample when two keys are that close.

## Fill an existing destination

```@example sampling
expected, next_rng = randsample_next(rng, population, weights, 12)
destination = similar(expected)
_, rng = randsample_next!(rng, population, weights, destination)
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
been validated and privately prepared. On CPU, a fill uses the calling task
directly unless `threaded=true`; it does not look up or launch a backend.

## Keep data on the right device

An array population must match the generator's backend. The same rule applies to array weights.
A host array with a CUDA-bound generator throws instead of silently copying.

Device-agnostic inputs, such as integer ranges, can serve a device-bound generator.
GPU populations must also support device indexing and storage.

Results stay on the generator's backend. Weighted validation can require a small host result.
Weighted sampling does not have the same performance profile as primitive fills.
