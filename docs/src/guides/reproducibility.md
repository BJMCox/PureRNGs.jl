# Reproducibility

Integer and uniform draws are bitwise identical across supported backends for
the same family, key, family region, and logical-bit position. Normal input
bits and midpoint values have the same guarantee.

Final normal values are bitwise reproducible on one backend and architecture.
Across backends or architectures, only the AS241 `log` and `sqrt` evaluations
may change the final bits.

Exponential raw integers and lattice inputs are bitwise identical across
supported backends. The CPU-token atanh transform has no architecture or
software identity exception because its written IEEE operations define the
exact result. CUDA, AMDGPU, and Metal tokens use `Base.log`. Their final values
may vary by backend, architecture, and complete software identity. That
identity includes Julia and every used backend package, compiler, toolkit, and
math library. Derived exponential values may differ only through that
primitive difference.

Reproduce a run by keeping the family, seed or key, derivation identifiers, and
draw order fixed. Use addressed draws when work order may change. See
[Immutable workflows](../tutorials/immutable-workflows.md) and
[Splitting and devices](../tutorials/splitting-and-devices.md) for examples.

## Reactant

Reactant may bake an immutable generator's key and position into a compiled
executable. A changed key or position may require a new compilation. Compile
or cache functions outside key- or position-varying loops. Reuse an executable
only for the key and position used to compile it.
