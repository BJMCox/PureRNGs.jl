# Reproducibility

## Preserve the stream inputs

Record the package version, generator type, seed or key, purpose IDs, and draw sequence.

Pure calls repeat their result. Continuation calls advance explicit state.
Bulk calls follow chained scalar consumption, so changing chunk sizes does not change the underlying stream.

Randomness belongs to logical work, not thread IDs or scheduling order.
See [Parallel jobs](@ref).

PureRNGs implements the Philox and Threefry keyed bijections, but its public streams need not match other packages built on them.

## Statistical validation

The Philox and Threefry cores reproduce the Random123 known-answer vectors, and the ChaCha core reproduces the ChaCha8, ChaCha12, and ChaCha20 test vectors.
RNGTest BigCrush over the packed uniform stream of every generator in both the `UInt32` and `Float64` lanes is best-effort release evidence, run on explicit request, and the release notes record which cases ran.
PractRand also runs as a diagnostic on the native word stream and on the split and `subrng` children.
The [published logs](https://github.com/BJMCox/PureRNGs.jl/releases/tag/bigcrush-practrand-f8e60b2-r1)
cover ChaCha, Philox4x32R7, and Threefry4x64R13 at commit `f8e60b2`, in both
lanes, with PractRand alongside. The same checks passed for Philox2x64R6,
Philox4x64R7, and Threefry4x32R12 at commit `c887861`; those logs are not yet
uploaded. Logs record the tested revision and remain outside Git.
Hosted CI has CPU tests only, including separate extension jobs.

## Separate bits from floating-point transforms

Primitive integer and uniform floating-point draws agree across supported execution sites.

Normal and exponential draws can differ in their final floating-point transformation across backends.
The generator's device selects the formula. A CPU generator evaluates AS241 for normals and a reduced-argument series for exponentials; a CUDA, AMDGPU, or Metal generator evaluates Giles' inverse error function for normals and `log` for exponentials.
The two normal formulas read the same random bits and the same uniform value, and both stay within 5 ulp for `Float32` and 6 ulp for `Float64` of the true quantile, but they do not return the same value.
A device-placed generator keeps its device when you draw from it on the host, so `randn(cuda_rng, T)` does not equal `randn(cpu_rng, T)`.
Within one formula, matching random bits still do not force different `log` or `sqrt` implementations to return identical values.

Reactant permits further compiler-dependent transformations.
Its exact limits appear in [Differentiation and compilation](@ref).

## Fixed random work

Package-owned samplers consume a fixed number of random bits for each result.
They do not retry rejected candidates.

This does not promise constant runtime. Allocation, table preparation, backend scheduling, and validation still have costs.
Foreign consumers of `StatefulRNG` may use rejection algorithms.

All draw kinds read one stream of bits at the generator's position.
Two draws of different kinds at the same position read the same bits. Advance the generator or derive a new key between them.

See [Distributions](@ref) for each mapping's bit span and finite-grid limits,
and [Generators and streams](@ref) for primitive widths and per-key capacities.

## Failure behavior

Eager package-owned fills check deterministic contract errors before writing.
Exhaustion does not silently wrap the state.

A draw that outruns the stream throws [`StreamExhausted`](@ref), which argument validation never throws.
Catch it to tell exhaustion from a bad argument: its type parameter names the generator type and its `bits` field gives the span the draw needed.
It carries no generator value, because holding one would slow every draw. You still hold the generator you passed in.
Derive a fresh key with [`splitrng`](@ref) or [`subrng`](@ref) and restart the work there, rather than reusing the exhausted position.

Foreign `Random` methods reaching `StatefulRNG` hooks can write partially before exhaustion.
Backend and resource failures do not carry the same atomicity guarantee.

Compiled Reactant carriers omit exhaustion checks. The caller must stay within capacity.
