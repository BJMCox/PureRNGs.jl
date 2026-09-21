# Development

Use a Julia session opened at the repository root.
Keep package code and code documentation in Git.
Keep statistical logs and generated reports outside Git.

## Architecture

Two fill scaffolds cover every array-producing draw. The generic scaffold is
keyed by a codec and serves uniform, normal, exponential, range, unweighted
sampling and mapped extension distributions. The weighted scaffold is separate,
because it folds a cumulative table first and runs two device kernels over an
intermediate threshold array.

The generic path has five steps:

1. Validate and reserve. Fill entries validate their inputs, then call
   `_fill_prevalidated!` to reserve the whole bit span before writing.
   Primitive allocating and destination fills enter through
   `_rand_transformed_next_array` and `_rand_transformed_next_fill!`.
2. CPU body. `_fill_transformed_cpu!` walks a `_DenseBitCursor` over
   consecutive blocks and writes elements without re-deriving a position for
   each draw. `Val{:uniform}` has its own method, which routes to
   `_fill_uniform_cpu!`.
3. Chunked CPU launcher. `_launch_cpu!` splits the destination on draw
   boundaries and hands the chunks to `_run_chunks`, so the threaded result
   equals the serial result. `_fill_chunk_elements(codec, T)` sizes a chunk.
4. Generic KernelAbstractions kernel. `_transformed_fill_kernel!` runs one work
   item per element, addressing its own position from its index, for every
   backend without a specialised path. It lives in
   `ext/PureRNGsKernelAbstractionsExt.jl`, with the rest of the device
   launchers.
5. Device plan hook. `_device_fill_plan(backend, rng, codec, T)` returns
   `nothing` on the generic path; a backend extension dispatches on the codec
   and returns a plan, which selects a tuned kernel through
   `_launch_device_fill!`.

Normal and exponential scalars use `_draw_next`; addressed primitives use
`_draw_at`. Public pure forms discard the returned state. Their user-facing
contracts live in the [API reference](docs/src/api.md).

A codec says how wide a draw is and what the draw means. It is `Val{:uniform}`,
a `_NormalCodec`, an `_ExponentialCodec`, a `_RangeCodec`, a `_PopulationCodec`,
or an extension subtype of `_MappedFillCodec`. Its protocol is five hooks:

- `_fill_width(codec, T)` — the bits one draw consumes.
- `_cooperative_value(codec, T, raw)` — the value those bits mean.
- `_codec_take(codec, rng, cursor, T)` — one element off the dense cursor, with
  the advanced cursor. The default is one fixed-width draw through
  `_cooperative_value`, which is all a distribution needs, so that is the seam
  an extension uses to add one without a new scaffold. `_RangeCodec` and
  `_PopulationCodec` override it because their draw is 64 or 128 bits wide and
  their value is a reduced ordinal rather than a converted raw.
- `_transformed_draw_unchecked(codec, rng, position, T)` — one draw at a given
  position, which the addressed entry and the generic device kernel use.
- `_fill_chunk_elements(codec, T)` — the CPU chunk size, with a default from the
  codec's width.

`_fill_cursor!` writes consecutive draws, `_fill_group!` writes one work item's
group, and `_fill_group_elements(codec, rng, T)` sizes that group.
Keep the specialised uniform stores in `uniform_fill.jl`: the Float64
`_store_f64_group!` interleaves decoding and stores, while
`_decode_transformed_group` returns a tuple. Merging them caused tuple spills
and a measured regression. CUDA store layouts and launch sizes live in its
extension, not in the portable scaffold.

Define `CUDA.@device_override` methods in the extension before device code
compiles. An overlay added later in a REPL need not reach package internals.
The `_mulhilo64` override uses the device high product without introducing
128-bit integers into device code; host arithmetic remains unchanged.

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
- `uniform_fill.jl` — `_DenseBitCursor`, the uniform CPU fill, and the uniform grouped stores.
- `fill_hooks.jl` — the uniform codec accessors, `_fill_backend`, and the `_launch_device_fill!` declaration the KernelAbstractions extension fills in.
- `cpu_scheduler.jl` — the CPU chunk constants and the `_run_chunks` work loop.
- `transformed_fill.jl` — the codec protocol, the CPU fill bodies, `_launch_cpu!`, `_fill_prevalidated!`, and the four internal entries the public methods forward to.
- `uniform.jl` — the public uniform array entries over the transformed scaffold.
- `addressed.jl` — `rand_at` and the addressed-position arithmetic it shares with `randn_at` and `randexp_at`.
- `normal.jl` — the two inverse normal CDFs the backend token selects between,
  AS241 and Giles' erfinv, `_open_midpoint`, and the public normal entries.
- `exponential.jl` — the exponential transform and the public exponential entries.
- `integers.jl` — range span, range bits, and the multiply-shift range reduction.
- `range_fill.jl` — `_RangeCodec`, the shared candidate reduction, and the public range fill entries.
- `sampling.jl` — population preparation, `_PopulationCodec`, and the public unweighted sampling entries.
- `weighted_sampling.jl` — `WeightTable`, weight folding, and the weighted sampling scaffold.
- `stateful.jl` — `StatefulRNG`, the mutable `Random.AbstractRNG` bridge.
- `docstrings.jl` — docstrings for the extended `Random` names, bound by signature; keep it after every file it documents.
- `precompile.jl` — the PrecompileTools workload; keep it last.

## Dependencies

The package depends on MLDataDevices, PrecompileTools, and Random. Everything
else is a weak dependency. KernelAbstractions is one of them: it carries every
device kernel through `ext/PureRNGsKernelAbstractionsExt.jl`, and CUDA, AMDGPU
and Metal each depend on it, so a GPU user still gets it. Adapt is the second
trigger of that extension, because a codec that carries a device array must
convert that field for a kernel argument. Moving it out of the
normal dependencies keeps CPU-only loading independent of the kernel stack.

Adding a normal dependency needs the same justification. Prefer a weak
dependency and an extension whenever a CPU-only user does not execute the code.

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
Pkg.instantiate()
include("docs/make.jl")
```

The docs project points at this checkout through a `[sources]` entry, so no
`Pkg.develop` call is needed and nothing is written back into `docs/Project.toml`.

Open `docs/build/index.html`.
The strict Documenter build runs CPU examples and checks exported docstrings and links.
Optional GPU and integration examples need their corresponding environments.

## Extensions and GPU checks

Separate projects live in `test/environments`.
Every environment is a package with its entry point at `test/runtests.jl`.
Each environment points at this checkout through a `[sources]` entry. Activate it and run its tests:

```julia
using Pkg
Pkg.activate("test/environments/distributions")
Pkg.instantiate()
include("test/environments/distributions/test/runtests.jl")
```

`Pkg.test()` runs the same entry point in a fresh subprocess, which is the better
choice when the suite must not inherit the current session's loaded packages.

The MeasureBase and Turing environments verify host `StatefulRNG` conformance.
CUDA is a local gate, not a hosted CI job. Record the exact source revision
for each device run. See [Devices](docs/src/manual/devices.md) for support
levels and [Differentiation and compilation](docs/src/integrations/compilation.md)
for the distinct Enzyme and Reactant contracts.
The Enzyme floor is 0.13.203, because 0.13 releases up to 0.13.199 assert on Julia 1.13
for differentiation paths outside this package's own rules.
Run statistical batteries only on explicit request, separately from performance
measurements. Upload logs as artifacts rather than committing them. The tested
generators, source revisions and available artifacts belong in
[Statistical validation](docs/src/manual/reproducibility.md#statistical-validation).

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

Measure precompile cost, package load and first-call latency separately.

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

### Coverage

Measure line coverage from a scratch environment, at the repository root:

```julia
using Pkg
Pkg.activate(temp = true)
Pkg.develop(path = pwd())
Pkg.add("Coverage")
Pkg.test("PureRNGs"; coverage = true)
using Coverage
covered, total = Coverage.get_summary(Coverage.process_folder("src"))
Coverage.clean_folder("src")
```

`get_summary` returns the covered and total line counts; their ratio is the coverage.
`clean_folder` deletes the `*.cov` files the run leaves in the checkout. They are
gitignored, so a forgotten cleanup does not reach a commit.

Coverage measures executed lines, not statistical quality. No percentage target is set.

### Changelog

Every user-visible change gets one entry in `CHANGELOG.md`, under the single
`## Unreleased` heading, in the `### Added`, `### Changed`, `### Removed`, or `### Fixed`
subsection that fits. Do not add a version heading or a date before a release.

## Automation

The CI workflow checks Julia 1.10, current stable Julia, and current prerelease Julia, serial and
threaded execution, Linux/macOS/Windows. The prerelease row runs on Linux only, with 4 threads,
and does not block the workflow when it fails.

A separate downgrade job installs Julia 1.10, downgrades every non-stdlib dependency to its
declared `[compat]` floor, and runs the test suite. This checks the floors in `[compat]`,
not only the latest releases.

The current Julia/Linux core job and five independent CPU extension jobs (Distributions, Enzyme,
Reactant, MeasureBase, Turing) each upload source and extension coverage to Codecov, which merges
their reports.
Each job retains its `lcov.info` in a separate `coverage-*` artifact.
Extension suites run on current stable Julia only. The core job also covers the 1.10 LTS floor,
and the downgrade job covers the compat floors of the package's own dependencies.
The conformance environments in `test/environments` declare and require current stable Julia,
because they use `[sources]`, which 1.10 does not support. The package itself still supports 1.10.
The Reactant job tests Philox4x32, Threefry4x64, Threefry4x32, and ChaCha.
The Turing job runs only on manual dispatch, and later on schedule; it is skipped on pull requests.
Full-family Reactant and GPU validation remain separate release gates.
Every successful coverage run uploads to Codecov using GitHub OIDC, without an opt-in input.
The first upload must confirm that Codecov accepts the repository's OIDC identity.
Coverage measures executed lines, not statistical quality. No percentage target is set.

Automatic CI and documentation runs are disabled while the repository is private.
Every job above is authored now, but both workflows support manual dispatch only until publication.
TagBot is removed. The package is not registered and must not be registered
while the repository is private. Release-ready means merged and pushed to the default branch,
with no tag and no registration.

## Before the first release

- Check every required gate on the exact release revision.
- Review the exported interface and backend support levels.
- Upload and link the complete statistical evidence.
- Update the changelog with the chosen version and date.
- Match the version in `Project.toml`.
- Confirm repository visibility and documentation hosting before registration.
- Obtain maintainer approval before tagging, publishing, or registering.
