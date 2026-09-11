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
Before a release, RNGTest BigCrush runs on the packed uniform stream of every generator in both the `UInt32` and `Float64` lanes.
The logs are release artifacts and are not committed. Hosted CI runs the CPU unit tests only.

## Separate bits from floating-point transforms

Primitive integer and uniform floating-point draws agree across supported execution sites.

Normal and exponential draws can differ in their final floating-point transformation across backends.
Matching random bits does not force different `log` or `sqrt` implementations to return identical values.

Reactant permits further compiler-dependent transformations.
Its exact limits appear in [Differentiation and compilation](@ref).

## Fixed random work

Package-owned samplers consume a fixed number of random bits for each result.
They do not retry rejected candidates.

This does not promise constant runtime. Allocation, sorting, backend scheduling, and validation still have costs.
Foreign consumers of `StatefulRNG` may use rejection algorithms.

All draw kinds read one stream of bits at the generator's position.
Two draws of different kinds at the same position read the same bits. Advance the generator or derive a new key between them.

## Failure behavior

Eager package-owned fills check deterministic contract errors before writing.
Exhaustion does not silently wrap the state.

Foreign `Random` methods reaching `StatefulRNG` hooks can write partially before exhaustion.
Backend and resource failures do not carry the same atomicity guarantee.

Compiled Reactant carriers omit exhaustion checks. The caller must stay within capacity.
