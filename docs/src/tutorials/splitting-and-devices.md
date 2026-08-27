# Splitting and devices

Split by role. Continue within each role.

```julia
using PureRNGs

root = Philox4x64(91)
proposal_rng = subrng(root, 1)
resampling_rng = subrng(root, 2)

proposal_rng, proposals = randn_next(proposal_rng, Float64, 128)
resampling_rng, uniforms = rand_next(resampling_rng, Float64, 128)

@assert length(proposals) == length(uniforms) == 128
@assert root == Philox4x64(91)
```

Use stable purpose identifiers instead of deriving keys from task order. Use
`splitrng(root, n)` when a caller owns a fixed group of unnamed children.

```julia
children = splitrng(root, 4)
draws = map(children) do child
    last(rand_next(child, UInt32))
end

@assert length(draws) == 4
@assert eltype(draws) === UInt32
```

Read [Splitting keys](../guides/splitting.md) for collision guidance and the
static `Val` form.

## Run the same workflow on CUDA

Load the backend, select its active device, and apply the matching
`MLDataDevices` device to the RNG. Allocation and fast-path selection remain
automatic.

```julia
using CUDA
using PureRNGs
using MLDataDevices

CUDA.device!(0)
root = MLDataDevices.CUDADevice()(Philox4x32(91))
worker = subrng(root, 1)
worker, values = randexp_next(worker, Float32, 1_000_000)

@assert values isa CUDA.CuArray{Float32}
```

This optional GPU example is not run when the documentation builds. It needs a
working CUDA installation and device.

The device token names a backend, not a physical GPU. Read
[Device binding](../guides/devices.md) before using multiple devices in one
process.

Binding preserves the key and position. Signed integers, uniform values, and
raw exponential lattice inputs remain equal. The final exponential value may
change because the backend token selects its transform.
