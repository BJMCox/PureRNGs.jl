# Device binding

Applying a supported MLDataDevices device binds an immutable RNG to that backend.
The RNG stores only a zero-size backend token. It discards any physical device
handle and element-type adaptor from the applied value.

Allocations and launches use the backend's active device at call time. Select the
active device before each operation when a process uses multiple devices. For
CUDA, call `CUDA.device!`:

```julia
CUDA.device!(device)
rng = MLDataDevices.CUDADevice()(Philox4x32(1234))
values = rand(rng, Float32, 1_000_000)
```

For AMDGPU and Metal, use that backend's active-device selection API before an
allocation or launch. Applying a named MLDataDevices device does not select or
retain its physical device.
