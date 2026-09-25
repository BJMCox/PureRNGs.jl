# Changelog

All notable changes to this project are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Draws of every signed and unsigned integer type from 8 to 128 bits, `Float16`, and `Complex` values, with normal draws for `Float16` and complex types and exponential draws for `Float16`. Integer ranges accept 128-bit element types; a range with more than 2^64 values reduces a 192-bit candidate. The 128-bit and complex types run on the CPU only.
- `StatefulRNG` serves every `Random` element type, `Random.seed!(rng)` without a seed, `copy!`, `==`, and `hash`.
- Picks from collections: `rand`, `rand_next`, `rand_at`, `rand!`, and `rand_next!` accept any array or range, a tuple, a string, a dict, or a set. A pick consumes what one `randsample_next` draw does. `StatefulRNG` picks from tuples, strings, dicts, and sets through the same law.
- `Char` draws, uniform over the Unicode scalar values as in `Random`.
- Permutations: `randperm_next`, `randcycle_next`, `shuffle_next`, their in-place forms, and `Random.randperm`, `randcycle`, `shuffle` and their in-place forms for pure generators. A permutation orders one uniform `UInt64` key per element, so it is the same on the CPU and on a GPU. `StatefulRNG` uses the same law.
- `randsample(...; replace = false)` samples without replacement: the leading elements of the shuffled population. With weights, it orders the population by `E / w` with one exponential draw `E` per element, which is successive sampling proportional to the remaining weights.
- A StatsBase extension: `sample`, `sample!`, `wsample`, and `wsample!` accept pure generators and draw what `randsample` does, and `samplepair` makes two range draws.
- Sampling, permutation, and pick errors state the values that broke the rule, for example `cannot draw 5 elements without replacement from a population of 3`.
- CUDA fills of 8- and 16-bit integers and `Float16`, and `Float16` normal and exponential fills, store 16 bytes per work item. On an A100 a 2^26 `UInt8` fill runs at 1015 GiB/s instead of 183, `UInt16` at 1176 instead of 292, and `Float16` at 715 instead of 145.
- `Threefry2x64R13`, the thirteen-round `Threefry2x64` that Salmon, Moraes, Dror, and Shaw (2011, Table 2) report as the smallest passing BigCrush. It has no BigCrush run of its own in the evidence release yet.

### Changed

- CPU fills, allocating draws, sampling, and `splitrng(rng, n)` run serially by default, as `Random` does. Pass `threaded = true` to split a CPU fill across threads. Values are unchanged either way. Large fills that relied on the old threaded default run slower until they opt in.
- Allocating draws, addressed array draws, and allocating sampling now accept the `threaded` keyword.
- `threaded` is a typed `Bool` keyword. A non-`Bool` value throws a `TypeError` instead of an `ArgumentError`.

## 0.0.1 - 2026-09-22

Initial development release.

### Added

- Eight immutable Philox and Threefry generators with explicit continuation and purpose-based key derivation. Continuation functions return `(value, next_rng)`.
- Primitive, integer-range, and population sampling, including weighted sampling with replacement. Every draw kind reads one stream of bits at the generator's position.
- Fixed-work `Normal`, `Uniform`, `Exponential`, `LogNormal`, `Weibull`, `Rayleigh`, `Laplace`, `Logistic`, `Gumbel`, `Pareto`, `Frechet`, `Cauchy`, `TriangularDist`, `Bernoulli`, and `DiscreteUniform` draws, continuations, addressed draws, and fills, plus `Categorical` labels and in-place population sampling with `randsample!` and `randsample_next!`.
- CPU and CUDA array generation, with preview AMDGPU and experimental Metal support.
- Round-reduced `Philox4x32R7`, `Philox2x64R6`, `Philox4x64R7`, `Threefry4x32R12`, and `Threefry4x64R13`. The first four use the smallest round count that Salmon, Moraes, Dror, and Shaw (2011, Table 2) report as passing BigCrush for that shape; `Threefry4x64R13` uses one round more than that table's twelve, the reduced count in Random123's known-answer tests. The round count is a type parameter on every generator.
- `ChaCha` generators with a 256-bit key, 512-bit output blocks, and twelve rounds by default, with the `ChaCha8`, `ChaCha12`, and `ChaCha20` round-count aliases, on every backend including Reactant.
- A host-only `Random.AbstractRNG` bridge.
- `rngkey` and `rngposition` accessors, and constructors that rebuild a generator at a saved position.
- Array dimensions as one tuple, integer-range destination fills, and addressed draws over an index range.
- `rand_at(rng, range, i)` addresses a single integer-range draw.
- `WeightTable` for reusable weighted sampling preparation.
- `splitrng(rng, n; threaded=true)` derives large child vectors on all CPU threads. The children do not depend on the keyword.
- Optional Distributions, Enzyme, and Reactant integrations. Reactant array fills of any size trace, with no per-element constants in the compiled module.
- `StreamExhausted{R}`, thrown when a draw outruns the generator's stream. `R` is the generator type and the single `bits` field is the required span. It holds no generator value, because keeping one alive across the capacity check cost 10% to 12% per chained `Philox4x64` draw. Argument validation keeps throwing `ArgumentError`.
- PrecompileTools workloads for common core and Distributions calls.
- GitHub installation instructions, a workflow-led manual, CPU and GPU tutorials, and an exported API reference.
- Normal and exponential draws use an open midpoint lattice, so every exponential draw is strictly positive. CPU generators evaluate AS241 for normals; CUDA, AMDGPU, and Metal generators evaluate Giles' inverse error function, so host and device normals differ in the last few ulp.
- Generators carry their decoded output block, so consecutive scalar draws inside one block run the core once.
- Packed CPU fills for every linearly indexed destination, including views and reshapes, with fixed-shift group decoding for Philox4x32 `Float64`, normal, and exponential fills.
- CUDA fills with packed 16-byte stores and the device high-multiply for 64-bit Philox. CUDA and AMDGPU weighted sampling folds the weights on the device and selects by binary search. Metal does not run sampling on the device.
- KernelAbstractions is a weak dependency, loaded by the GPU backend packages, so a CPU-only environment stays small.
