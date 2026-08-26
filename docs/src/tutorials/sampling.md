# Sampling

`randsample` samples with replacement and returns values only.
`randsample_next` also returns the next RNG.

```julia
using PureRNGs

root = Philox4x32(42)
rng, draws = randsample_next(root, 10:10:60, 8)

@assert length(draws) == 8
@assert all(in(10:10:60), draws)
@assert randsample(root, 10:10:60, 8) == draws
@assert root == Philox4x32(42)
```

Omit the count to request as many draws as the population contains.

```julia
values = ["low", "middle", "high"]
rng, draws = randsample_next(rng, values)

@assert length(draws) == length(values)
@assert all(in(values), draws)
```

## Weighted sampling

Pass plain real weights in population order. Weights must be finite and
non-negative, and their sum must be finite and positive.

```julia
population = [:low, :middle, :high]
weights = [1, 3, 12]
rng, weighted = randsample_next(rng, population, weights, 20)

@assert length(weighted) == 20
@assert all(in(population), weighted)
@assert randsample(root, population, weights, 5) ==
        last(randsample_next(root, population, weights, 5))
```

No weight wrapper or performance option is required. Batch generation and
backend-specific selection paths are automatic.

## Sample on CUDA

Keep the RNG, population, weights, and result on the same backend.

```julia
using CUDA
using PureRNGs
using MLDataDevices

CUDA.device!(0)
rng = MLDataDevices.CUDADevice()(Philox4x32(42))
population = CUDA.CuArray(Int32[10, 20, 30, 40])
weights = CUDA.CuArray(Float64[1, 2, 3, 4])
rng, draws = randsample_next(rng, population, weights, 10_000)

@assert draws isa CUDA.CuArray{Int32}
```

This optional GPU example is not run when the documentation builds. It needs a
working CUDA installation and device.
