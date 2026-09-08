# PureRNGs.jl

[![CI](https://github.com/BJMCox/PureRNGs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BJMCox/PureRNGs.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/BJMCox/PureRNGs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BJMCox/PureRNGs.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://bjmcox.github.io/PureRNGs.jl/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Counter-based random numbers for Julia, with explicit state and CPU/GPU array generation.
Eight Philox and Threefry families support uniform, normal, and exponential draws, integer ranges, weighted sampling, and key splitting.

```julia
using PureRNGs, Random

# Each draw returns the next generator without changing the original.
rng = Philox4x32(123456)
rng, values = rand_next(rng, Float32, 1_000)
rng, more = rand_next(rng, Float32, 1_000)

# Use the same interface on a GPU.
using CUDA, MLDataDevices

rng = Philox4x32(123456) |> CUDADevice()
rng, values = rand_next(rng, Float32, 1_000_000) # CuArray
```

Portions of the code in this package were generated with the assistance of LLMs.

[Documentation](https://bjmcox.github.io/PureRNGs.jl/) · [Apache 2.0 license](LICENSE)
