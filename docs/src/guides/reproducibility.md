# Reproducibility

Integer and uniform draws are bitwise identical across supported backends for
the same family, key, family region, and logical-bit position. Normal input
bits and midpoint values have the same guarantee.

Complete software identity records the Julia version and every used backend
package, compiler, toolkit, and math-library version. Final normal values are
bitwise reproducible within one backend, architecture, and complete software
identity. They may vary across any of these only through AS241's `log` and
`sqrt` evaluations.

Exponential raw integers and lattice inputs are bitwise identical across
supported backends. The CPU-token atanh transform has no architecture or
software identity exception because its written IEEE operations define the
exact result. CUDA, AMDGPU, and Metal tokens use `Base.log`. Their final values
may vary by backend, architecture, and complete software identity. Derived
exponential values may differ only through that primitive difference.

Reproduce a run by keeping the family, seed or key, derivation identifiers, and
draw order fixed. Use addressed draws when work order may change. See
[Immutable workflows](../tutorials/immutable-workflows.md) and
[Splitting and devices](../tutorials/splitting-and-devices.md) for examples.

## Reactant

Reactant may bake an immutable generator's key and position into a compiled
executable. A changed key or position may require a new compilation. Compile
or cache functions outside key- or position-varying loops. Reuse an executable
only for the key and position used to compile it.
