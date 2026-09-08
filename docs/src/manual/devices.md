# Devices

## Bind a generator

Install the backend package in your environment, then load it with MLDataDevices.

```julia
using PureRNGs, Random
using CUDA, MLDataDevices

rng = Philox4x32(123456) |> CUDADevice()
rng, values = rand_next(rng, Float32, 1_000_000)
@assert values isa CuArray
```

Allocating draws create backend arrays. Fills require a destination on the same backend.
No unsupported operation silently falls back to the CPU.

Rebinding preserves the key and position:

```julia
cpu_rng = rng |> CPUDevice()
```

## Select a physical GPU

The generator stores a backend token, not a physical device handle.
Use `CUDA.device!` or the backend's equivalent to select the active GPU before allocation and execution.

Passing a named device object does not pin the generator to that physical device.
Keep arrays and the active device consistent.

## Know where scalar arithmetic runs

A scalar or addressed call made on the host runs on the host, even when the generator has a GPU token.
The same operation inside a GPU kernel runs on that GPU.

Use allocating draws, fills, or [GPU kernels](@ref) to generate values on the device.

## Support levels

| Backend | Status | Device-executing operations |
|:--|:--|:--|
| CPU | Required release gate | Full eager API |
| CUDA | Required local release gate | Primitive draws, ranges, sampling, supported distributions |
| AMDGPU | Preview | Backend extension, not a required release gate |
| Metal | Experimental | Restricted primitive draws and fills |

Metal device execution supports 32-bit families with `Bool`, `UInt32`, `Int32`, `UInt64`, `Int64`, and `Float32` results.
Normal and exponential results must be `Float32`.

Metal excludes allocating ranges, population sampling, and distribution fills.
Host scalar operations are not restricted by the Metal token.
Unsupported host-called operations throw even for empty outputs.

`BitArray` fills are CPU-only.
Unsupported MLDataDevices types throw `ArgumentError`.

Reactant uses its own conversion path. See [Differentiation and compilation](@ref).
