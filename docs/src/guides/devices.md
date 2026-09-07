# Device binding

Applying a supported `MLDataDevices` device binds an immutable RNG to that
backend. The RNG stores only a zero-size backend token. It does not retain a
physical device handle or an element-type adaptor from the applied device.

## Backend tiers

CPU and CUDA are fully supported and their conformance suites block releases.
AMDGPU is a preview backend: its suite runs and failures are documented without
blocking releases. Metal support is experimental.

On Metal, device-executing allocating draws and destination fills are limited
to primitive `Bool`, `UInt32`, `UInt64`, `Int32`, and `Int64` results and
`Float32` uniform, normal, and exponential results on 32-bit-word RNG families.
Metal excludes `Float64` results, 64-bit-word RNG families, allocating integer
ranges, all fixed-distribution forms, and all sampling forms. These exclusions
also apply to zero-size requests.

Host-computing operations remain available on a Metal-bound generator for every
result type and RNG family. This includes scalar primitives, ranges, and fixed
distributions; scalar continuations; addressed draws; `splitrng`; and `subrng`.

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
rng, values = randexp_next(rng, Float32, 1_000_000)
```

This optional GPU example is not run when the documentation builds.

For AMDGPU and Metal, use that backend's active-device selection API before an
allocation or launch. A named device discards any physical device handle. It
does not select or retain that device.

The RNG and every device-bound input must use the same backend. Generated
arrays stay on that backend. PureRNGs does not silently copy results to the
host or fall back to CPU generation.

The backend token selects the exponential transform. CPU uses the package's
fixed table-free transform. CUDA, AMDGPU, and Metal use `Base.log`. Rebinding
preserves exponential input bits but may change final exponential values.
Signed integers and uniform values remain bitwise equal.

Destination fills require the destination and generator on the same backend.
A mismatch throws `ArgumentError` before generation or mutation. The
`threaded` keyword controls only CPU task use and does not change device paths.
