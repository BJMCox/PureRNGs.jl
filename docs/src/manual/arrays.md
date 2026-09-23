# Arrays and performance

## Allocate or reuse a buffer

```@example arrays
using PureRNGs, Random

rng = Philox4x32(7)
matrix, rng = rand_next(rng, Float32, 3, 4)

buffer = similar(matrix)
result, rng = rand_next!(rng, buffer)
@assert result === buffer
```

Dimensions are positional integers or one `Dims` tuple, as in `Random`.
Arrays use Julia's native column-major order.

A bang modifies the destination, not the immutable generator.
`rand!` returns the destination. `rand_next!` also returns the advanced generator.

The same forms exist for `randn!`, `randn_next!`, `randexp!`, and `randexp_next!`.

## Destinations that take the packed CPU path

Any destination is filled with the same stream, but the speed depends on how it
indexes. A CPU destination whose `IndexStyle` is `IndexLinear` takes the packed
fill, which decodes whole generator blocks at once: an `Array`, a contiguous or
strided `view` of one, and a `reshape` of either. A destination whose style is
`IndexCartesian`, such as a `transpose` or a `view` with a strided second index,
falls back to one draw per element and runs several times slower. Copy such a
destination into an `Array`, fill the `Array`, and copy back when the fill is on
a hot path.

## Process a long stream in chunks

```@example arrays
function sum_batches(rng, buffer, batches)
    total = 0.0
    for _ in 1:batches
        _, rng = rand_next!(rng, buffer)
        total += sum(buffer)
    end
    return rng, total
end

buffer = Vector{Float64}(undef, 1024)
rng, total = sum_batches(Philox4x32(7), buffer, 4)
total
```

Buffer reuse avoids repeated allocation. Continuing the returned state preserves the stream across chunk boundaries.

## Control CPU threading

CPU fills run serially on the calling task by default, as `Random` fills do.
Pass `threaded=true` to split a large fill into chunks across Julia's threads:

```@example arrays
_, rng = rand_next!(rng, buffer; threaded=true)
```

Destination fills, allocating draws, sampling, and `splitrng(rng, n)` all take the keyword.
It never changes the values: a threaded fill writes the same stream as the serial one.
It affects CPU execution only. Fills too small to split stay on the calling task.
Leave it off inside your own threaded loop, so the outer loop owns the parallelism.

See [Parallel jobs](@ref) for a complete outer-threaded workflow.

## Packed booleans

Boolean draws use individual random bits. An ordinary `Array{Bool}` still stores one byte per element.

```@example arrays
packed = falses(128)
_, rng = rand_next!(rng, packed)
packed
```

`BitArray` destinations are CPU-only. GPU Boolean arrays use their backend's array storage.

## Measure the workload

Use the repository's `benchmark/throughput.jl` from a Julia session.
Select the generator, result type, and device there.

Warm compilation before timing. Compare allocating draws separately from buffer fills.
For GPU measurements, include synchronization and distinguish device generation from host transfers.

Scalar draws inside a kernel and bulk array fills are different workloads. Neither timing predicts the other.
