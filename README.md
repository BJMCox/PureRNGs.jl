# PureRNGs.jl

PureRNGs provides counter-based random number generators for reproducible
CPU and accelerator code. Each continuing draw returns the advanced generator
with its result. Array allocation, device placement, and fast fill paths follow
the generator, with fast defaults and no required tuning.

## Quick start

`Philox4x32` is the recommended default for general use:

```julia
using PureRNGs
using Random

rng = Philox4x32(1234)
rng, values = rand_next(rng, Float32, 1_000_000)
```

`rand_next` returns `(next_rng, value)`. Pass `next_rng` to the next sequential
draw:

```julia
rng = Philox4x32(1234)
rng, uniform = rand_next(rng, Float64)
rng, normal = randn_next(rng, Float64)
rng, signed = rand_next(rng, Int64)
rng, exponential = randexp_next(rng, Float64)
```

Keeping an older generator is useful when you want to repeat a draw. Reusing it
by accident repeats the same stream position.

For an existing array, use `rand_next!`. CPU fills use the fast threaded path by
default:

```julia
destination = Vector{Float32}(undef, 1_000_000)
rng, destination = randexp_next!(rng, destination)
```

`threaded=false` is available as optional advanced CPU control. It is not needed
for normal use.

## Allocate on a device

Bind the generator to an MLDataDevices device before drawing. Allocating draws
then create their result directly on that device:

```julia
using CUDA
using MLDataDevices

device = MLDataDevices.CUDADevice()
rng = Philox4x32(1234) |> device
rng, values = randexp_next(rng, Float32, 1_000_000)
```

The same workflow applies to supported AMDGPU and Metal devices. Backend-specific
fast paths are selected automatically. Device-bound allocation and generation
stay on the device.

## Weighted sampling

Pass weights as a plain vector. No wrapper type or preparation step is required:

```julia
rng = Philox4x32(1234)
population = ["red", "green", "blue"]
weights = [1.0, 2.0, 7.0]

rng, samples = randsample_next(rng, population, weights, 1_000)
```

## Stateful interoperability

Use `StatefulRNG` when an existing host-side API requires a mutable
`Random.AbstractRNG`:

```julia
using Random

mutable_rng = StatefulRNG(Philox4x32(1234))
values = rand(mutable_rng, Float64, 1_000)
```

The bridge advances its held immutable generator after each draw.

## Learn more

- [Immutable workflows](docs/src/tutorials/immutable-workflows.md)
- [Splitting and devices](docs/src/tutorials/splitting-and-devices.md)
- [Sampling](docs/src/tutorials/sampling.md)
- [Stateful interoperability](docs/src/tutorials/stateful-interop.md)
- [API reference](docs/src/api.md)
