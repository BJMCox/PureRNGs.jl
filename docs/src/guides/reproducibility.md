# Reproducibility

Integer and uniform draws are bitwise identical across supported backends for
the same family, key, family region, and logical-bit position. Normal input
bits and midpoint values have the same guarantee.

Final normal values are bitwise reproducible on one backend and architecture.
Across backends or architectures, only the AS241 `log` and `sqrt` evaluations
may change the final bits.

Exponential raw integers and lattice inputs are bitwise identical across
supported backends. Final exponential values are reproducible for one backend
token. A CPU token uses the fixed table-free transform. CUDA, AMDGPU, and Metal
tokens use `Base.log`, so rebinding may change only the final exponential value
and values derived from it.

Reproduce a run by keeping the family, seed or key, derivation identifiers, and
draw order fixed. Use addressed draws when work order may change. See
[Immutable workflows](../tutorials/immutable-workflows.md) and
[Splitting and devices](../tutorials/splitting-and-devices.md) for examples.

## Reactant

Reactant may bake an immutable generator's key and position into a compiled
executable. A changed key or position may require a new compilation. Compile
or cache functions outside key- or position-varying loops. Reuse an executable
only for the key and position used to compile it.
