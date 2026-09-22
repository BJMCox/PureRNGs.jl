# PureRNGs.jl

[![CI](https://github.com/BJMCox/PureRNGs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BJMCox/PureRNGs.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/BJMCox/PureRNGs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BJMCox/PureRNGs.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://bjmcox.github.io/PureRNGs.jl/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Counter-based random numbers for Julia, with explicit state and CPU/GPU array generation.
Philox, Threefry, and ChaCha generators support uniform, normal, and exponential draws, integer ranges, weighted sampling, and key splitting.

Install from GitHub using Julia 1.10 or later:

```julia
using Pkg
Pkg.add(url="https://github.com/BJMCox/PureRNGs.jl")
```

```julia
using PureRNGs, Random

# Each draw returns the next generator without changing the original.
rng = Philox4x32(123456)
values, rng = rand_next(rng, Float32, 1_000)
more, rng = rand_next(rng, Float32, 1_000)

# Use the same interface on a GPU.
using CUDA, MLDataDevices

rng = Philox4x32(123456) |> CUDADevice()
values, rng = rand_next(rng, Float32, 1_000_000) # CuArray
```

Backend support differs by tier: CPU and CUDA are release gates, AMDGPU is a preview, and Metal is experimental. See the [Devices](https://bjmcox.github.io/PureRNGs.jl/manual/devices/) page.

Coverage reflects CPU CI only. GPU tests run separately and are not included because hosted CI has no GPU runner.

See the [BigCrush and PractRand results](https://github.com/BJMCox/PureRNGs.jl/releases/tag/statistical-evidence-2026-09-22) for statistical test logs and reproduction details.

Portions of the code in this package were generated with the assistance of LLMs.

[Documentation](https://bjmcox.github.io/PureRNGs.jl/) · [Apache 2.0 license](LICENSE)
