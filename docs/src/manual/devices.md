# Devices

## Bind a generator

Install the backend package in your environment, then load it with MLDataDevices.

```julia
using PureRNGs, Random
using CUDA, MLDataDevices

rng = Philox4x32(123456) |> CUDADevice()
values, rng = rand_next(rng, Float32, 1_000_000)
@assert values isa CuArray
```

KernelAbstractions is a weak dependency that carries the device kernels, and every
GPU backend package pulls it in, so a CPU-only environment installs neither.

Allocating draws create backend arrays. Fills require a destination on the same backend.
No unsupported operation silently falls back to the CPU. The one exception is
explicit: `StatefulRNG(rng)` rebinds a device generator to the CPU. See
[Random interoperability](@ref).

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
| AMDGPU | Preview | Every operation CUDA serves, through the portable kernel path |
| Metal | Experimental | Every operation without Float64 or 128-bit arithmetic |

AMDGPU is a backend extension, not a required release gate. It has no tuned
fill plan, so fills run the portable kernel path. See [Sampling](@ref) for
weighted preparation and device rules.

CUDA and AMDGPU fills reject the 128-bit integer element types, which run on the CPU only.

Metal has no Float64 or 128-bit integer arithmetic.
Its device execution serves every generator and every result type except `Float64`, `ComplexF64`, and the 128-bit integers, including ranges, unweighted sampling, permutations, and `Float32` distributions.
Weighted sampling, `WeightTable`, and `Categorical` fold Float64 weights, so a Metal generator rejects them.
Host scalar operations are not restricted by the Metal token.
Unsupported host-called operations throw even for empty outputs.

`BitArray` fills are CPU-only.
Unsupported MLDataDevices types throw `ArgumentError`.

Reactant uses its own conversion path. See [Differentiation and compilation](@ref).
