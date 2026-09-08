# Reproducibility

## Preserve the stream inputs

Record the package version, generator family, seed or key, purpose IDs, and draw sequence.

Pure calls repeat their result. Continuation calls advance explicit state.
Bulk calls follow chained scalar consumption, so changing chunk sizes does not change the underlying stream.

Randomness belongs to logical work, not thread IDs or scheduling order.
See [Parallel jobs](@ref).

PureRNGs implements Philox and Threefry cores, but its public streams need not match other packages using those cores.

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

Uniforms, normals, exponentials, ranges, and weighted sampling use separate counter-address regions.
Address separation does not prevent equal output values by chance.

## Failure behavior

Eager package-owned fills check deterministic contract errors before writing.
Exhaustion does not silently wrap the state.

Foreign `Random` methods reaching `StatefulRNG` hooks can write partially before exhaustion.
Backend and resource failures do not carry the same atomicity guarantee.

Compiled Reactant carriers omit exhaustion checks. The caller must stay within capacity.
