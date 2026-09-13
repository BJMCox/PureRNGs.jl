# GPU kernels

This tutorial requires CUDA, MLDataDevices, and KernelAbstractions in your Julia environment,
plus a working NVIDIA GPU.
The examples are optional and do not run in the CPU documentation build.

## Generate an array on the GPU

```julia
using PureRNGs, Random
using CUDA, MLDataDevices

rng = Philox4x32(123456) |> CUDADevice()
values, rng = rand_next(rng, Float32, 1_000_000)
@assert values isa CuArray

_, rng = randn_next!(rng, values)
host_values = Array(values) # Explicit transfer for host inspection
```

Keep subsequent computation on the GPU to avoid transfer costs.
Reuse the destination when the shape stays fixed.

## Fuse a draw into your computation

Addressed draws let each work item obtain its own value without an intermediate random array.

```julia
using KernelAbstractions

@kernel function centered_noise!(out, rng)
    i = @index(Global, Linear)
    out[i] = 2f0 * randat(rng, Float32, i) - 1f0
end

rng = Philox4x32(123456) |> CUDADevice()
out = CUDA.zeros(Float32, 1024)
backend = get_backend(out)
centered_noise!(backend)(out, rng; ndrange=length(out))
KernelAbstractions.synchronize(backend)
```

Each index selects a draw relative to the supplied generator's current position.
The kernel leaves that generator unchanged.

Repeated launches with the same generator repeat the noise.
Use a distinct purpose key for each logical launch, or advance the stream explicitly outside the kernel.

For example, `subrng(root, iteration)` gives a reproducible stream for each iteration.
Choose a wider-key generator when deriving many keys.

Bulk fill throughput and fused-kernel throughput measure different work.
Benchmark the whole consuming kernel before choosing between them.
