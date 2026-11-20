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

PureRNGs supports KernelAbstractions 0.9 and 0.10. Backend packages and other
dependencies determine which version your environment can resolve.
Ordinary CPU draws use PureRNGs' own CPU path, not KernelAbstractions' CPU backend.

Allocating draws create backend arrays. Fills require a destination on the same backend.
No unsupported operation silently falls back to the CPU. The one exception is
explicit: `StatefulRNG(rng)` rebinds a device generator to the CPU. See
[Random interoperability](@ref).

Rebinding preserves the key and position:

```julia
cpu_rng = rng |> CPUDevice()
```

`MLDataDevices.get_device(rng)` returns the generator's backend device.
`MLDataDevices.get_device_type(rng)` returns its device type.
With CUDA or AMDGPU loaded, `get_device` resolves the active physical GPU at the
time of the call, so device queries can combine a generator with arrays there.
Without the backend loaded, it returns an unnamed backend device.
`get_device_type` reports only the backend. Neither query pins the generator to a GPU.

## Select a physical GPU

The generator stores a backend token, not a physical device handle.
Use `CUDA.device!` or the backend's equivalent to select the active GPU before allocation and execution.

Passing a named device object does not pin the generator to that physical device.
Keep arrays and the active device consistent.

## Know where scalar arithmetic runs

Primitive scalar draws and scalar addressed draws made on the host run on the host, even when the generator has a GPU token.
Supported scalar operations called inside a GPU kernel execute there.

Use allocating draws, fills, or [GPU kernels](@ref) to generate values on the device.

## Support levels

| Backend | Status | Device-executing operations |
|:--|:--|:--|
| CPU | Required release gate | Full eager API |
| CUDA | Required local release gate | Primitive draws, ranges, sampling, supported distributions |
| AMDGPU | Preview | Portable paths for the documented CUDA operations |
| Metal | Experimental | Primitive fills, ranges, unweighted sampling, permutations, supported eager distributions |

AMDGPU is a backend extension, not a required release gate. It has no tuned
fill plan, so fills run the portable kernel path. See [Sampling](@ref) for
weighted preparation and device rules.

Use host scalar calls or CPU arrays for 128-bit integers. A host scalar call
preserves the backend binding of a GPU-bound generator.

Metal has no Float64 or 128-bit integer arithmetic.
Use supported device result types, including `Float32` and `ComplexF32`.
Device execution of weighted sampling, `WeightTable`, and `Categorical` requires Float64 weight arithmetic.
See [Distributions](@ref) for result and parameter restrictions.
Unsupported host-called operations throw even for empty outputs.

`BitArray` fills are CPU-only.
Unsupported MLDataDevices types throw `ArgumentError`.

Reactant uses its own conversion path. See [Differentiation and compilation](@ref).
