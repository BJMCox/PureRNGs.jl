# PureRNGs.jl

Counter-based random numbers for Julia, with explicit state and CPU/GPU array generation.

```julia
using PureRNGs, Random

rng = Philox4x32(123456)
rng, values = rand_next(rng, Float32, 1_000)
rng, more = rand_next(rng, Float32, 1_000)
```

Continuation draws return a new generator. The original stays unchanged. Bulk draws use the same stream as chained scalar draws.

Bind a generator to a GPU with MLDataDevices:

```julia
using CUDA, MLDataDevices

rng = Philox4x32(123456) |> CUDADevice()
rng, values = rand_next(rng, Float32, 1_000_000) # CuArray
```

Eight Philox and Threefry families support primitive draws, integer ranges, sampling with replacement, and purpose-based key derivation. Optional extensions cover Distributions, Enzyme, and Reactant.

Start with the [documentation](docs/src/index.md) and [getting started](docs/src/getting-started.md).
See [device support](docs/src/manual/devices.md) before choosing a backend.

The package is under development. Install this checkout with `Pkg.develop(path="/path/to/PureRNGs.jl")`.

For benchmarks, open [benchmark/throughput.jl](benchmark/throughput.jl) in a Julia session and choose the device there.

## Build the documentation

From the repository root, run these commands in Julia:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.activate("docs")
Pkg.instantiate()
include("docs/make.jl")
```

Open `docs/build/index.html`. The build runs CPU examples and checks exported docstrings and links.

Licensed under Apache-2.0.
