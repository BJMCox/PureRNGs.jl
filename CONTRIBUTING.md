# Development

Use a Julia session opened at the repository root.
Keep package code and code documentation in Git.
Keep private plans, statistical logs, and generated reports outside Git.

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
The CUDA environment requires Julia 1.11, so it does not run on the 1.10 floor the
other suites keep.
AMDGPU remains preview support. Metal remains experimental.

Reactant and Enzyme have separate checks. Neither replaces the CPU or CUDA gates.
RNGTest BigCrush covers all nine families in both UInt32 and Float64 lanes before release.
Upload statistical logs as artifacts rather than committing them.

## Performance

Use `benchmark/throughput.jl` from a Julia session.
Set the device, family, result type, and sample size before including it.
Keep runs long enough to measure steady-state performance.

Compare the same workload and hardware against the exact base revision.
Separate allocation, fills, kernel-local draws, and device transfers.
Preserve benchmark results outside Git.

## Automation

The CI workflow checks Julia 1.10 and current stable Julia, serial and threaded execution,
Linux/macOS/Windows.

The current Julia/Linux core job and independent Distributions, Enzyme, and Reactant CPU jobs
each upload source and extension coverage to Codecov, which merges their reports.
Each job retains its `lcov.info` in a separate `coverage-*` artifact.
The Enzyme test environment requires version 0.13.203 or later for its Julia 1.13 compiler fixes.
The Distributions suite also runs on Julia 1.10.
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
