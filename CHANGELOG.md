# Changelog

All notable changes to this project are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Initial development release.

### Added

- Eight immutable Philox and Threefry generators with explicit continuation and purpose-based key derivation. Continuation functions return `(value, next_rng)`.
- Primitive, integer-range, and population sampling, including weighted sampling with replacement. Every draw kind reads one stream of bits at the generator's position.
- Fixed-work `Normal`, `Uniform`, `Exponential`, `LogNormal`, `Weibull`, `Rayleigh`, `Laplace`, `Logistic`, `Gumbel`, `Pareto`, `Frechet`, `Cauchy`, `TriangularDist`, `Bernoulli`, and `DiscreteUniform` draws, continuations, addressed draws, and fills, plus `Categorical` labels and in-place population sampling with `randsample!` and `randsample_next!`.
- CPU and CUDA array generation, with preview AMDGPU and experimental Metal support.
- Round-reduced `Philox4x32R7`, `Philox2x64R6`, `Philox4x64R7`, `Threefry4x32R12`, and `Threefry4x64R13`. Each round count is the smallest that Salmon, Moraes, Dror, and Shaw (2011, Table 2) report as passing BigCrush for that shape. The round count is a type parameter on every generator.
- `ChaCha` generators with a 256-bit key, 512-bit output blocks, and twelve rounds by default, with the `ChaCha8`, `ChaCha12`, and `ChaCha20` round-count aliases, on every backend including Reactant.
- A host-only `Random.AbstractRNG` bridge.
- `rngkey` and `rngposition` accessors, and constructors that rebuild a generator at a saved position.
- Array dimensions as one tuple, integer-range destination fills, and addressed draws over an index range.
- `rand_at(rng, range, i)` addresses a single integer-range draw.
- `WeightTable` for reusable weighted sampling preparation.
- `splitrng(rng, n; threaded=true)` derives large child vectors on all CPU threads. The children do not depend on the keyword.
- Optional Distributions, Enzyme, and Reactant integrations. Reactant array fills of any size trace, with no per-element constants in the compiled module.
- `StreamExhausted{R}`, thrown when a draw outruns the generator's stream. `R` is the generator type and the single `bits` field is the required span. It holds no generator value, because keeping one alive across the capacity check cost 10% to 12% per chained `Philox4x64` draw. Argument validation keeps throwing `ArgumentError`.
- A PrecompileTools workload over every public draw kind, in the package and in the Distributions extension. The first fill, scalar draw, and distribution draw of a session no longer compile.
- A workflow-led manual, CPU and GPU tutorials, and an exported API reference.

### Changed

- CUDA packed fills reuse the aligned full-tile kernel across generator families. Shared staging skips redundant stream extraction for aligned fills. Shifted positions, partial tiles, and small-fill fallbacks are unchanged.
- The Philox cores unroll their rounds explicitly, as the Threefry cores already did. CUDA.jl 6.4 ships a ptxas that left the ten-round loop rolled, which cost the Philox fill kernels about a third of their throughput on an A100. With the unrolled cores a Philox4x32 `Float32` fill of 2^27 elements reaches 1.3 TiB/s on an A100 under CUDA.jl 6.4. Streams are unchanged.
- Exponential draws use the open midpoint lattice the normal transform already uses, and consume 23 bits for `Float32` and 52 for `Float64` instead of 24 and 53. This is stream-law version 12. The old closed lattice contained zero, so the smallest draw was a negative zero; every draw is now strictly positive. The reach is unchanged at 16.6355 and 36.7368. `Exponential`, `Weibull`, `Rayleigh`, and `Pareto` follow the new width, `Laplace` consumes 24 or 53 bits, and every draw positioned after an exponential draw in a mixed stream moves by one bit per exponential draw.
- Normal draws on a CUDA, AMDGPU, or Metal generator evaluate Giles' inverse error function instead of AS241, as exponential draws already choose their formula by device. This is stream-law version 13. CPU normals are unchanged, and so are the consumed widths, the bit stream, and the uniform value each normal draw is built from; only the final normal value on a device generator moves. The new formula is the more accurate of the two on both widths, and a `Float32` normal fill of 2^27 elements reaches 635 GiB/s on an A100 where it reached 384. A normal drawn on the host from a device-placed generator follows its device, so it no longer equals the same draw from a CPU generator.
- The addressed draws are `rand_at`, `randn_at`, and `randexp_at`, matching the `_next` continuation family. They were `randat`, `randnat`, and `randexpat`, and those spellings are gone.
- Generators carry their decoded output block, so consecutive scalar draws inside a block run the core once. Chained scalar draws and `StatefulRNG` draws run two to three times faster for the 64-bit and ChaCha generators.
- CPU-bound 64-bit Philox generators use the host widening multiply. Non-CUDA kernels keep the portable four-product form.
- CUDA kernels take the 64-bit Philox widening product from the device's high-multiply instruction, which halves the core instruction count. 2^27-element `UInt64` and `Float64` fills from Philox2x64 and Philox4x64 reach 1067 to 1103 GiB/s where they reached 714 to 751. These are best results from five-second BenchmarkTools trials on a quiet A100 PCIe 40GB with CUDA.jl 6.4. Streams are unchanged.
- CUDA integer fills from `ChaCha` use the cooperative 16-byte store kernel, and the 64-bit output tile of that kernel doubles. A 2^27-element `ChaCha` `UInt64` fill reaches 918 GiB/s on an A100 where it reached 336, and its `Float64` fill 874 where it reached 783.
- The Philox4x32 Float64 fill extracts 128 draws per 53 blocks with fixed shifts, with a bit buffer for the remainder.
- The Philox4x32 normal and exponential fills decode whole groups of aligned draws with fixed shifts instead of walking a per-element bit cursor. Serial fills run 21% to 24% faster for `Float64` and 27% to 47% faster for `Float32`.
- Threaded CPU fills spread over the threads from three chunks instead of four. Fills that produce three chunks run up to 2.5 times faster, and no size measured between 1024 and 16384 elements gets slower.
- The packed CPU fills dispatch on `IndexStyle`, so every linearly indexed destination reaches them, not only `Array`. A contiguous or strided `view`, or a `reshape` of one, now fills about five times faster serially and matches an `Array` when threaded.
- CUDA and AMDGPU prepare weighted samples with a cooperatively staged fold and select them by binary search over the cumulative table. They no longer sort thresholds. Metal still excludes device-executing sampling.
- `rand(m, T, n)` on `StatefulRNG` uses the package fill.
- KernelAbstractions is a weak dependency. The device kernels moved to `PureRNGsKernelAbstractionsExt`, which every GPU backend package loads transitively. A CPU-only environment resolves 22 packages instead of 34 and loads in 0.04 s instead of 0.17 s.
