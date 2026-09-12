# Changelog

## Unreleased

Initial development release.

- Eight immutable Philox and Threefry generators with explicit continuation and purpose-based key derivation. Continuation functions return `(value, next_rng)`.
- Primitive, integer-range, and population sampling, including weighted sampling with replacement. Every draw kind reads one stream of bits at the generator's position.
- CPU and CUDA array generation, with preview AMDGPU and experimental Metal support.
- Generators carry their decoded output block, so consecutive scalar draws inside a block run the core once. Chained scalar draws and `StatefulRNG` draws run two to three times faster for the 64-bit and ChaCha generators.
- Round-reduced `Philox4x32R7` and `Threefry4x64R13`, the Random123 minimum round counts that pass BigCrush. The round count is a type parameter on every generator.
- `ChaCha` generators with a 256-bit key, 512-bit output blocks, and twelve rounds by default, with the `ChaCha8`, `ChaCha12`, and `ChaCha20` round-count aliases, on every backend including Reactant.
- CPU-bound 64-bit Philox generators use the host widening multiply. Kernel code keeps the portable four-product form.
- The Philox4x32 Float64 fill extracts 128 draws per 53 blocks with fixed shifts, with a bit buffer for the remainder.
- A host-only `Random.AbstractRNG` bridge.
- `rngkey` and `rngposition` accessors, and constructors that rebuild a generator at a saved position.
- Optional Distributions, Enzyme, and Reactant integrations.
- A workflow-led manual, CPU and GPU tutorials, and an exported API reference.
