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
Pkg.instantiate()
include("docs/make.jl")
```

Open `docs/build/index.html`.
The strict Documenter build runs CPU examples and checks exported docstrings and links.
Optional GPU and integration examples need their corresponding environments.

## Extensions and GPU checks

Separate projects live in `test/environments`.
Activate the relevant project and develop this checkout before running its tests:

```julia
using Pkg
Pkg.activate("test/environments/distributions")
Pkg.develop(path=pwd())
Pkg.instantiate()
include("test/environments/distributions/runtests.jl")
```

CUDA is a local release gate, not a hosted CI job.
Run it on a supported CUDA device and record the exact source revision.
AMDGPU remains preview support. Metal remains experimental.

Reactant and Enzyme have separate checks. Neither replaces the CPU or CUDA gates.
RNGTest BigCrush covers all eight families in both UInt32 and Float64 lanes before release.
Upload statistical logs as artifacts rather than committing them.

## Performance

Use `benchmark/throughput.jl` from a Julia session.
Set the device, family, result type, and sample size before including it.
Keep runs long enough to measure steady-state performance.

Compare the same workload and hardware against the exact base revision.
Separate allocation, fills, kernel-local draws, and device transfers.
Preserve benchmark results outside Git.

## Automation

CPU test CI remains manual, with only a `workflow_dispatch` trigger.
Do not run it or enable automatic test runs without maintainer approval.

The CI workflow checks Julia 1.10 and current stable Julia, serial and threaded execution,
Linux/macOS/Windows.

The current Julia/Linux job collects source coverage and retains `lcov.info` as a `coverage` artifact.
Every successful coverage run uploads to Codecov using GitHub OIDC, without an opt-in input.
The first upload must confirm that Codecov accepts the repository's OIDC identity.
Coverage measures executed lines, not statistical quality. No percentage target is set.

Documentation builds automatically when docs, source, extensions, the package project,
or the docs workflow change on `main`. It also supports manual dispatch.
The strict build runs examples and doctests, then retains a `documentation` artifact.
Successful builds on `main` deploy to
[GitHub Pages](https://bjmcox.github.io/PureRNGs.jl/).
Docs deployment does not run or wait for the package test matrix.

TagBot responds automatically to JuliaTagBot's registry notifications and also supports manual dispatch.
It creates tags and releases for registered versions. It does not register the package.
Registration still requires maintainer approval.

## Before the first release

- Check every required gate on the exact release revision.
- Review the exported interface and backend support levels.
- Upload and link the complete statistical evidence.
- Update the changelog with the chosen version and date.
- Match the version in `Project.toml`.
- Confirm repository visibility and documentation hosting before registration.
- Obtain maintainer approval before tagging, publishing, or registering.
