## What changed

Describe the change and why it is needed.

## Pre-push checklist

Hosted CI is manual-dispatch only, so this recipe is the
gate. See the "Before pushing" section of `CONTRIBUTING.md`.

- [ ] `Pkg.test()` on the package, which sets `--check-bounds=yes`.
- [ ] The Distributions environment suite.
- [ ] JuliaFormatter 2.12.6 `format(["src", "ext", "test", "docs", "benchmark"])`, with an empty `git diff`.
- [ ] `Aqua.test_all(PureRNGs)`.
- [ ] The documentation build, `include("docs/make.jl")`.
- [ ] The CUDA environment on a CUDA host, when `ext/PureRNGsCUDAExt.jl` or any kernel changes.
- [ ] A changelog entry under `## Unreleased`, in the matching subsection.
