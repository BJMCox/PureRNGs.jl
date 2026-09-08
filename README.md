# PureRNGs.jl

[![CI](https://github.com/BJMCox/PureRNGs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BJMCox/PureRNGs.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://bjmcox.github.io/PureRNGs.jl/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

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

Start with the [documentation](https://bjmcox.github.io/PureRNGs.jl/) and [getting started](docs/src/getting-started.md).
See [device support](docs/src/manual/devices.md) before choosing a backend.

The package is under development. Install this checkout with `Pkg.develop(path="/path/to/PureRNGs.jl")`.

For benchmarks, open [benchmark/throughput.jl](benchmark/throughput.jl) in a Julia session and choose the device there.

[Documentation](https://bjmcox.github.io/PureRNGs.jl/) · [Development](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Apache 2.0 license](LICENSE)
