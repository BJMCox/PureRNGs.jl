# Changelog

## Unreleased

Initial development release.

- Eight immutable Philox and Threefry generators with explicit continuation and purpose-based key derivation. Continuation functions return `(value, next_rng)`.
- Primitive, integer-range, and population sampling, including weighted sampling with replacement. Every draw kind reads one stream of bits at the generator's position.
- Fixed-work `Normal`, `Uniform`, `Exponential`, `LogNormal`, `Weibull`, `Rayleigh`, `Laplace`, `Logistic`, `Gumbel`, `Pareto`, `Frechet`, `Cauchy`, `TriangularDist`, `Bernoulli`, and `DiscreteUniform` draws, continuations, addressed draws, and fills, plus `Categorical` labels and in-place population sampling with `randsample!` and `randsample_next!`.
- CPU and CUDA array generation, with preview AMDGPU and experimental Metal support.
- Generators carry their decoded output block, so consecutive scalar draws inside a block run the core once. Chained scalar draws and `StatefulRNG` draws run two to three times faster for the 64-bit and ChaCha generators.
- Round-reduced `Philox4x32R7` and `Threefry4x64R13`, the Random123 minimum round counts that pass BigCrush. The round count is a type parameter on every generator.
- `ChaCha` generators with a 256-bit key, 512-bit output blocks, and twelve rounds by default, with the `ChaCha8`, `ChaCha12`, and `ChaCha20` round-count aliases, on every backend including Reactant.
- CPU-bound 64-bit Philox generators use the host widening multiply. Kernel code keeps the portable four-product form.
- The Philox4x32 Float64 fill extracts 128 draws per 53 blocks with fixed shifts, with a bit buffer for the remainder.
- The Philox4x32 normal and exponential fills decode whole groups of aligned draws with fixed shifts instead of walking a per-element bit cursor. Serial fills run 21% to 24% faster for `Float64` and 27% to 47% faster for `Float32`.
- Threaded CPU fills spread over the threads from three chunks instead of four. Fills that produce three chunks run up to 2.5 times faster, and no size measured between 1024 and 16384 elements gets slower.
- The packed CPU fills dispatch on `IndexStyle`, so every linearly indexed destination reaches them, not only `Array`. A contiguous or strided `view`, or a `reshape` of one, now fills about five times faster serially and matches an `Array` when threaded.
- A host-only `Random.AbstractRNG` bridge.
- `rngkey` and `rngposition` accessors, and constructors that rebuild a generator at a saved position.
- Array dimensions as one tuple, integer-range destination fills, and addressed draws over an index range.
- `randat(rng, range, i)` addresses a single integer-range draw.
- `WeightTable` for reusable weighted sampling preparation.
- Every GPU backend prepares weighted samples with a parallel fold kernel and selects them by binary search over the cumulative table. AMDGPU and Metal no longer run a single-work-item scan with a device `sortperm`.
- `splitrng(rng, n; threaded=true)` derives large child vectors on all CPU threads. The children do not depend on the keyword.
- `rand(m, T, n)` on `StatefulRNG` uses the package fill.
- Optional Distributions, Enzyme, and Reactant integrations. Reactant array fills of any size trace, with no per-element constants in the compiled module.
- `StreamExhausted`, thrown when a draw outruns the generator's stream, carrying the generator and the required bit span. Argument validation keeps throwing `ArgumentError`.
- A PrecompileTools workload over every public draw kind, in the package and in the Distributions extension. The first fill, scalar draw, and distribution draw of a session no longer compile.
- KernelAbstractions is a weak dependency. The device kernels moved to `PureRNGsKernelAbstractionsExt`, which every GPU backend package loads transitively. A CPU-only environment resolves 22 packages instead of 34 and loads in 0.04 s instead of 0.17 s.
- A workflow-led manual, CPU and GPU tutorials, and an exported API reference.
