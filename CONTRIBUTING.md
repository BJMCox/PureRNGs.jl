# Development

Use a Julia session opened at the repository root.
Keep package code and code documentation in Git.
Keep private plans, statistical logs, and generated reports outside Git.

## Architecture

Five fill scaffolds cover every array-producing draw: uniform, transformed
(normal and exponential), range, unweighted sampling, and weighted sampling.
Each scaffold repeats the same eight steps, so a new scaffold is written by
following an existing one rather than by inventing a shape.

1. Validate and reserve. Check the destination device and serviceability, then
   reserve the whole bit span up front so a fill either runs or throws
   `StreamExhausted` before writing anything.
2. CPU dense cursor fill. Walk a `_DenseBitCursor` over consecutive blocks and
   write elements without re-deriving a position for each draw.
3. Chunked CPU launcher. Split the destination on draw boundaries and hand the
   chunks to `_run_chunks`, so the threaded result equals the serial result.
4. Generic KernelAbstractions kernel. One work item per element, addressing its
   own position from its index, for every backend without a specialised path.
5. Device plan hook. A `_device_*_fill_plan` method returns `nothing` on the
   generic path; a backend extension returns a plan to select a tuned kernel.
6. Allocating entry. Allocate through `_allocate_draw_array` on the generator's
   device, then call the prevalidated fill.
7. Continuation entry. The `*_next` and `*_next!` forms return the advanced
   generator alongside the result.
8. Pure entry. The `Random` forms return only the result and never advance the
   caller's generator.

The transformed scaffold takes a codec instead of a result type. A codec is
`Val{:normal}`, an `_ExponentialCodec`, or an extension subtype of
`_MappedFillCodec`, and it supplies the per-element map from uniform bits. This
is the seam an extension uses to add a distribution without a new scaffold.

Bit extraction lives in two files. `_extract_bits_unchecked` and `_chain_bits`
in `bits.jl` serve scalar draws, while `_local_dense_bits` serves the kernels
and `_take_dense_bits_unchecked` in `uniform_fill.jl` serves the CPU cursor.
The `_fill_uniform_grouped_unchecked!` methods in `uniform_fill.jl` store a
whole block of draws at once for `Bool` and for the 32-bit and 64-bit integers.
`Float32` and `Float64` take the generic cursor on the CPU and the four-element
group given by `_device_uniform_fill_group` in the kernels.

All position arithmetic funnels through `_split_bit_advance` in `generators.jl`,
which is the single place a bit offset becomes a block and bit pair. Device and
keyword validation funnels through `validation.jl`.

One line per file in `src/`, in include order:

- `PureRNGs.jl` — module: exports, the `_ReactantRNG` declaration both Reactant extensions dispatch on, and the include list.
- `core_words.jl` — word arithmetic over `_CoreWord`, so one block function serves host words and device words.
- `philox.jl` — Philox round function, multipliers, and Weyl constants.
- `threefry.jl` — Threefry round function, rotation tables, and parity constants.
- `chacha.jl` — ChaCha quarter round, block function, and constants.
- `generators.jl` — `AbstractPureRNG`, the nine generator types, backend tokens, the two position types, reservation, and `StreamExhausted`.
- `allocation.jl` — `_allocate_draw_array` and the per-backend `_allocate_array`.
- `bits.jl` — block evaluation and the scalar bit extractors.
- `derive.jl` — `splitrng`, `subrng`, and the tagged key derivation behind them.
- `uniform_scalar.jl` — the generator and result-type unions, `_draw_bits`, `_from_bits`, scalar `rand`, and the untyped-draw guards.
- `validation.jl` — device, keyword, and serviceability checks for fills and sampling.
- `uniform_fill.jl` — `_DenseBitCursor`, the dense CPU uniform fill, and the grouped stores.
- `uniform_kernels.jl` — the KernelAbstractions uniform kernels and `_launch_device_fill!`.
- `cpu_scheduler.jl` — CPU chunk sizes and the `_run_chunks` work loop.
- `transformed_fill.jl` — the codec types and the transformed CPU fill and launchers.
- `uniform.jl` — the uniform fill scaffold and its allocating entries.
- `addressed.jl` — `randat` and the addressed-position arithmetic it shares with `randnat` and `randexpat`.
- `normal.jl` — the AS241 inverse normal CDF and the normal scaffold.
- `exponential.jl` — the exponential transform and the exponential scaffold.
- `integers.jl` — range span, range bits, and the multiply-shift range reduction.
- `range_fill.jl` — the range fill scaffold.
- `sampling.jl` — population preparation and the unweighted sampling scaffold.
- `weighted_sampling.jl` — `WeightTable`, weight folding, and the weighted sampling scaffold.
- `stateful.jl` — `StatefulRNG`, the mutable `Random.AbstractRNG` bridge.
- `docstrings.jl` — docstrings for the extended `Random` names, bound by signature; keep it after every file it documents.
- `precompile.jl` — the PrecompileTools workload; keep it last.

## Package tests

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

Start a separate Julia session with multiple threads to check parallel CPU execution.
For example, start Julia with `julia --threads=4 --project`, then run `Pkg.test()`.
Do not change the Julia version merely to bypass a failure.

Tests must preserve stream order, counter advancement, device placement, and fixed random-work contracts.
Prefer small public-behavior tests and independent mathematical oracles.

## Documentation

From the repository root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.activate("docs")
Pkg.develop(path=pwd())
Pkg.instantiate()
include("docs/make.jl")
```

Open `docs/build/index.html`.
The strict Documenter build runs CPU examples and checks exported docstrings and links.
Optional GPU and integration examples need their corresponding environments.

## Extensions and GPU checks

Separate projects live in `test/environments`.
Every environment is a package with its entry point at `test/runtests.jl`.
Activate the relevant project and develop this checkout before running its tests:

```julia
using Pkg
Pkg.activate("test/environments/distributions")
Pkg.develop(path=pwd())
Pkg.instantiate()
include("test/environments/distributions/test/runtests.jl")
```

`Pkg.test()` runs the same entry point in a fresh subprocess, which is the better
choice when the suite must not inherit the current session's loaded packages.

The MeasureBase and Turing environments verify host `StatefulRNG` conformance.
Run them with the same `Pkg.activate` recipe.

CUDA is a local release gate, not a hosted CI job.
Run it on a supported CUDA device and record the exact source revision.
AMDGPU remains preview support. Metal remains experimental.

Reactant and Enzyme have separate checks. Neither replaces the CPU or CUDA gates.
The Enzyme floor is 0.13.203, because 0.13 releases up to 0.13.199 assert on Julia 1.13
for differentiation paths outside this package's own rules.
RNGTest BigCrush over all nine families in both UInt32 and Float64 lanes is best-effort
release evidence, run on explicit request. The release notes record which cases ran.
The published evidence so far covers ChaCha, Philox4x32R7, and Threefry4x64R13 at commit
`f8e60b2`, in both lanes, with PractRand as a diagnostic.
Upload statistical logs as artifacts rather than committing them.

## Performance

Start Julia at the repository root with `julia --project=benchmark`.
Set the device, family, result type, and sample size, then run
`include("benchmark/throughput.jl")`.
Keep runs long enough to measure steady-state performance.

Compare the same workload and hardware against the exact base revision.
Separate allocation, fills, kernel-local draws, and device transfers.
Preserve benchmark results outside Git.

Load and first-call latency come from the PrecompileTools workload in
`src/precompile.jl` and the matching one at the end of
`ext/PureRNGsDistributionsExt.jl`. Extend the workload when a new public draw
kind appears. Check it with a fresh `julia --trace-compile=stderr` run over the
README quickstart and `docs/src/getting-started.md`, and add any method that
still compiles.

On an Apple M4 Pro the workload moved the first 128-element `Float64` fill from
0.50 s to 0.07 ms and the first `Normal` fill from 72 ms to 0.04 ms, at the cost
of `Base.compilecache` rising from 0.9 s to 9.4 s and `using PureRNGs` from
0.39 s to 0.65 s. These numbers are references for the trade on one machine, not
gates.

## Before pushing

Hosted CI is manual-dispatch only while the repository is private, so this recipe is the gate.

1. Run `Pkg.test()` on the package. It sets `--check-bounds=yes`, so allocation
   assertions must hold under bounds checking.
2. Run the Distributions environment suite.
3. Run JuliaFormatter 2.12.6 `format(["src", "ext", "test", "docs", "benchmark"])`
   and confirm `git diff` is empty.
4. Run `Aqua.test_all(PureRNGs)`.
5. Build the documentation with `include("docs/make.jl")`.
6. Run the CUDA environment on a CUDA host when `ext/PureRNGsCUDAExt.jl` or any
   kernel changes.

## Automation

The CI workflow checks Julia 1.10 and current stable Julia, serial and threaded execution,
Linux/macOS/Windows.

The current Julia/Linux core job and independent Distributions, Enzyme, and Reactant CPU jobs
each upload source and extension coverage to Codecov, which merges their reports.
Each job retains its `lcov.info` in a separate `coverage-*` artifact.
Extension suites run on current stable Julia only. The core job also covers the 1.10 LTS floor.
The conformance environments in `test/environments` declare and require current stable Julia,
because they use `[sources]`, which 1.10 does not support. The package itself still supports 1.10.
The Reactant job tests Philox4x32, Threefry4x64, Threefry4x32, and ChaCha.
Full-family Reactant and GPU validation remain separate release gates.
Every successful coverage run uploads to Codecov using GitHub OIDC, without an opt-in input.
The first upload must confirm that Codecov accepts the repository's OIDC identity.
Coverage measures executed lines, not statistical quality. No percentage target is set.

Automatic CI and documentation runs are disabled while the repository is private.
Both workflows support manual dispatch only.
TagBot is removed. The package is not registered and must not be registered
while the repository is private.

## Before the first release

- Check every required gate on the exact release revision.
- Review the exported interface and backend support levels.
- Upload and link the complete statistical evidence.
- Update the changelog with the chosen version and date.
- Match the version in `Project.toml`.
- Confirm repository visibility and documentation hosting before registration.
- Obtain maintainer approval before tagging, publishing, or registering.
