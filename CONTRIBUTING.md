# Development

Use a Julia session opened at the repository root.
Keep package code and code documentation in Git.
Keep statistical logs and generated reports outside Git.

## Branches and merging

Open development pull requests against `dev`.
Use a `dev`-to-`main` pull request to promote tested work for release.
This is a contributor convention, not an automated source-branch restriction.
Maintainers may use another source branch when needed.
Required CI and documentation checks still apply.

## Architecture

The generic fill scaffold is keyed by a codec and serves uniform, normal,
exponential, range, unweighted sampling and mapped extension distributions.
Weighted sampling prepares a cumulative table. Its device path uses two
kernels over an intermediate threshold array. Multivariate methods fill
columns or apply a distribution transform to a primitive fill.

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

Normal and exponential scalars use `_draw_next`. Addressed primitives use
`_draw_at`. Continuation calls return the next generator. Addressed calls and
`randsample` return only values. `StatefulRNG` retains the continuation for
standard `Random` calls. See the [API reference](docs/src/api.md).

A codec says how wide a draw is and what the draw means. It is `Val{:uniform}`,
a `_NormalCodec`, an `_ExponentialCodec`, a `_RangeCodec`, a `_PopulationCodec`,
or an extension subtype of `_MappedFillCodec`. Its protocol is five hooks:

- `_fill_width(codec, T)` — the bits one draw consumes.
- `_cooperative_value(codec, T, raw)` — the value those bits mean.
- `_codec_take(codec, rng, cursor, T)` — one element off the dense cursor, with
  the advanced cursor. The default is one fixed-width draw through
  `_cooperative_value`, which is all a distribution needs, so that is the seam
  an extension uses to add one without a new scaffold. `_RangeCodec` and
  `_PopulationCodec` override it to reduce a candidate to an ordinal.
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

`_split_bit_advance` in `generators.jl` splits a bit offset into a block and bit
pair. Device and keyword validation live in `validation.jl`.

Start at [the module entry point](src/PureRNGs.jl) for exports and include order.

- `generators.jl`, `bits.jl`, and `derive.jl` define generator state, the packed
  stream and key derivation. The Philox, Threefry, ChaCha and PCG4D cores each
  have their own file.
- `uniform_scalar.jl`, `uniform_fill.jl`, `transformed_fill.jl`, and
  `addressed.jl` implement primitive draws and their shared fill and addressing
  paths. `engine.jl` defines the hooks used by external engines.
- `normal.jl`, `exponential.jl`, `gamma.jl`, and `dirichlet.jl` provide
  transform and codec implementations. Optional distribution entry points
  live in `ext/`.
- `integers.jl`, `collections.jl`, `sampling.jl`, `weighted_sampling.jl`,
  and `permutations.jl` implement ranges, picks, resampling and permutations.
- `stateful.jl` implements the mutable bridge. `unified.jl` implements the
  `!!` adapters for pure and mutable generators.

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

Tests must preserve stream order, counter advancement, device placement, and each sampler's random-work contract.
Gamma-family rejection can continue on child streams without changing the fixed parent advancement.
Prefer small public-behavior tests and independent mathematical oracles.

## Documentation

The docs environment requires Julia 1.11 or later. The package supports Julia 1.10.

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
Documenter runs CPU examples and checks exported docstrings and internal links.
It reports external link failures as warnings.
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
The KernelAbstractions environment runs the device fill and weighted-scan kernels on
the KernelAbstractions CPU backend, so they are checked without a GPU.
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
Warm compilation, then time each case for at least ten seconds.

Compare the same workload and hardware against the exact base revision.
Separate allocation, fills, kernel-local draws, and device transfers.
Preserve benchmark results outside Git, except the regression pins below.

`benchmark/regression.jl` compares each case's minimum time with its pin in
`benchmark/baselines/<name>.toml`. Use it from the benchmark REPL:

```julia
include("benchmark/regression.jl")
report = run_regression("apple-cpu"; extra_cases = distribution_cases(),
    report_path = "/tmp/purerngs-performance.toml")
```

Load CUDA first. Select `"a100"` for large fills or `"gh200"` for small and
medium fills, unaligned normal draws, and a fused normal consumer on the
corresponding GPU. Use one Julia thread for the CUDA pins and `--threads=auto`
for the Mac pins.
For a regression decision, the pin needs a source revision, a trial of at least
ten seconds, and matching recorded controls such as the host, device, Julia and
dependency versions, and thread counts. Inconclusive runs still report timings
and list the differences in `report["comparison_issues"]`.
The command-line form `regression.jl <name> [--update]` exits with 1 for a regression
beyond tolerance and 2 for an inconclusive comparison. Update pins from a clean
package checkout after explaining and accepting the measured change.

Reports identify the loaded PureRNGs checkout and benchmark script separately,
with their paths, Git revisions and dirty states when available. They record
Julia/package versions, device, threads, host load, minima, medians, and host
allocations. GPU timings include synchronization but exclude copying results
back to the host. The allocation counts cover host memory. Check competing
CPU/GPU usage before each run.
Alternate exact base and candidate revisions, and repeat close results.

The optional distribution sweep covers Gamma shapes below, at, and above one,
Beta, narrow/wide Dirichlet columns, and diagonal/dense MvNormal. It times
in-place public calls with prebuilt distributions, including each call's
validation and device preparation. These cases have no pins until measured.
Set `cached = true` on a weighted case to measure reused `WeightTable` input.
Set `storage = "packed"` on a CPU Bool fill case to measure a `BitArray`.
Use small, cache-sized, and large cases. Keep cold load/compilation and AD
profiles separate from steady-state throughput.

Load and first-call latency come from the PrecompileTools workload in
`src/precompile.jl` and the matching one at the end of
`ext/PureRNGsDistributionsExt.jl`. Extend the workload when a new public draw
kind appears. Check it with a fresh `julia --trace-compile=stderr` run over the
README quickstart and `docs/src/getting-started.md`, and add any method that
still compiles.

Measure precompile cost, package load and first-call latency separately.

## Before pushing

Hosted CI runs on every push to `main` and `dev` and every pull request, but GPU and
statistical checks stay outside it, so this recipe is the gate.

1. Run `Pkg.test()` on the package. It sets `--check-bounds=yes`, so allocation
   assertions must hold under bounds checking.
2. Run the Distributions environment suite.
3. Run JuliaFormatter 2.12.6 `format(["src", "ext", "test", "docs", "benchmark"])`
   and confirm `git diff` is empty.
4. Run `Aqua.test_all(PureRNGs)`.
5. Build the documentation with `include("docs/make.jl")`.
6. Run the CUDA environment on a CUDA host when `ext/PureRNGsCUDAExt.jl` or any
   kernel changes.
7. Run `benchmark/regression.jl` on each pinned host when a fill, sampler, or
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

Record user-visible changes once under `## Unreleased` in `CHANGELOG.md`.
Put incompatible behavior and migration steps under `### Breaking changes`.
Use `### Added`, `### Changed`, `### Removed`, or `### Fixed` for the rest.
Add the version heading and date at release time.

## Automation

The CI workflow checks Julia 1.10, current stable Julia, and current prerelease Julia, serial and
threaded execution, Linux/macOS/Windows. The prerelease row runs on Linux only, with 4 threads,
and does not block the workflow when it fails.

A separate downgrade job installs Julia 1.10, downgrades every non-stdlib dependency to its
declared `[compat]` floor, and runs the test suite. This checks the floors in `[compat]`,
not only the latest releases.

The current Julia/Linux core job and the CPU extension jobs retain their
`lcov.info` files as separate `coverage-*` artifacts. Codecov publication uses
GitHub OIDC and is best-effort.
See the [CI matrix](.github/workflows/CI.yml) for the nine CPU extension
environments, both KernelAbstractions series, the Enzyme Julia pin, and the
Reactant generator selection. The conformance projects in `test/environments`
use `[sources]` and declare their own Julia versions.
Full-family Reactant and GPU validation remain separate release gates.

CI runs on pushes to `main` and `dev` and on pull requests.
Documentation runs on pushes to `main` and on pull requests. Builds on `main`
deploy the unversioned site to GitHub Pages.
Both workflows also support manual dispatch.
Dependabot checks GitHub Actions weekly and groups updates into one pull request.

## Release checklist

- Match the version in `Project.toml`, `CITATION.cff`, and the changelog.
  Use the same release date in the citation and changelog. Confirm repository
  visibility and the documentation deployment.
- Check every required gate on the exact release revision.
- Review the exported interface and backend support levels.
- Check the [statistical release gate](docs/src/manual/reproducibility.md#statistical-release-gate)
  for every named configuration. Carry earlier evidence forward only for
  unchanged streams, recording the tested revisions and why the results apply.
  Publish and link the evidence for changed streams. Missing or failed cases
  block release.
- With maintainer approval, invoke Registrator on the validated release commit.
  Include the release's changelog section after the literal `Release notes:` line.
  Each `0.0.x` increment crosses a Julia compatibility boundary. Explain its
  breaking changes in the trigger and check that they appear in the General PR.
  [RegistryCI release notes](https://juliaregistries.github.io/RegistryCI.jl/stable/guidelines/#Providing-and-updating-release-notes)
- After General accepts the version, [TagBot](.github/workflows/TagBot.yml)
  creates the version tag and GitHub release.
  Dispatch the workflow manually if the registration notification is missed.
  If the default token cannot tag a workflow-changing commit, a maintainer
  creates the tag and release at the registered revision. See
  [TagBot's manual fallback](https://github.com/JuliaRegistries/TagBot#commits-that-modify-workflow-files).
