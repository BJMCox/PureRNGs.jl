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

## Manual automation

Automatic CI remains disabled. The workflows have only `workflow_dispatch` triggers.
Do not run them or add push, pull-request, or tag triggers without maintainer approval.

The CI workflow checks Julia 1.10 and current stable Julia, serial and threaded execution,
Linux/macOS/Windows, and strict documentation builds.
It retains the built docs as a `documentation` artifact.

The current Julia/Linux job collects source coverage and retains `lcov.info` as a `coverage` artifact.
Its optional Codecov upload uses GitHub OIDC and requires repository setup in Codecov.
Coverage measures executed lines, not statistical quality. No percentage target is set.

The optional Pages job requires GitHub Pages configuration and deployment approval.
It runs only from `main`, after successful tests and docs.
Until hosting is configured, README links point to the documentation source.

TagBot is also manual-only. Do not dispatch it before release approval.

## Before the first release

- Check every required gate on the exact release revision.
- Review the exported interface and backend support levels.
- Upload and link the complete statistical evidence.
- Update the changelog with the chosen version and date.
- Match the version in `Project.toml`.
- Confirm repository visibility and documentation hosting before registration.
- Obtain maintainer approval before tagging, publishing, or registering.
