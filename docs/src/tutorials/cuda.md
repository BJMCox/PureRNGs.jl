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

## Batch fills before waiting

GPU fills enqueue work asynchronously.
Generate a batch across particles, then let the next operation consume that batch on the same CUDA stream:

```julia
function propagate_steps!(rng, particles, noise, steps)
    for _ = 1:steps
        _, rng = randn_next!(rng, noise)
        @. particles = 0.9f0 * particles + 0.3f0 * noise
    end
    return rng
end

particles = CUDA.zeros(Float32, 65_536)
noise = similar(particles)
rng = propagate_steps!(rng, particles, noise, 20)
CUDA.synchronize()
```

The fill and broadcast remain separate kernels, but need no host wait between them.
The final synchronization waits for all queued steps.
Include that wait when timing the complete workload.
Wait before host inspection or explicit cross-stream handoffs, not after every fill.

One large fill also avoids the launch overhead of many small fills.
Batch independent work when the algorithm permits it.
Time steps here depend on earlier states, so they still run in order.

The [Particle filter](@ref) tutorial adds likelihoods and resampling, then shows a fully fused observation loop.

## Fuse a draw into your computation

Addressed draws let each work item obtain its own value without an intermediate random array.

```julia
using KernelAbstractions

@kernel function centered_noise!(out, rng)
    i = @index(Global, Linear)
    out[i] = 2.0f0 * rand_at(rng, Float32, i) - 1.0f0
end

rng = Philox4x32(123456) |> CUDADevice()
out = CUDA.zeros(Float32, 1024)
backend = get_backend(out)
centered_noise!(backend)(out, rng; ndrange = length(out))
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
