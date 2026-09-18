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
- The packed CPU fills dispatch on `IndexStyle`, so every linearly indexed destination reaches them, not only `Array`. A contiguous or strided `view`, or a `reshape` of one, now fills about five times faster serially and matches an `Array` when threaded.
- A host-only `Random.AbstractRNG` bridge.
- `rngkey` and `rngposition` accessors, and constructors that rebuild a generator at a saved position.
- Array dimensions as one tuple, integer-range destination fills, and addressed draws over an index range.
- `WeightTable` for reusable weighted sampling preparation.
- `splitrng(rng, n; threaded=true)` derives large child vectors on all CPU threads. The children do not depend on the keyword.
- `rand(m, T, n)` on `StatefulRNG` uses the package fill.
- Optional Distributions, Enzyme, and Reactant integrations. Reactant array fills of any size trace, with no per-element constants in the compiled module.
- `StreamExhausted`, thrown when a draw outruns the generator's stream, carrying the generator and the required bit span. Argument validation keeps throwing `ArgumentError`.
- A workflow-led manual, CPU and GPU tutorials, and an exported API reference.
