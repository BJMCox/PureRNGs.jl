# Device binding

Applying a supported `MLDataDevices` device binds an immutable RNG to that
backend. The RNG stores only a zero-size backend token. It does not retain a
physical device handle or an element-type adaptor from the applied device.

Allocations and launches use the backend's active device at call time. Select
the active device before each operation when one process uses multiple devices.
For CUDA, call `CUDA.device!`:

```julia
using CUDA
using PureRNGs
using MLDataDevices

device = 0
CUDA.device!(device)
rng = MLDataDevices.CUDADevice()(Philox4x32(1234))
rng, values = rand_next(rng, Float32, 1_000_000)
```

This optional GPU example is not run when the documentation builds.

For AMDGPU and Metal, use that backend's active-device selection API before an
allocation or launch. Applying a named `MLDataDevices` device does not select
or retain its physical device.

The RNG and every device-bound input must use the same backend. Generated
arrays stay on that backend. PureRNGs does not silently copy results to the
host or fall back to CPU generation.
