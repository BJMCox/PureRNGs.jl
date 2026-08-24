# Reproducibility

Integer and uniform draws are bitwise identical across supported backends for
the same family, key, family region, and logical-bit position. Normal input
bits and midpoint values have the same guarantee.

Final normal values are bitwise reproducible on one backend and architecture.
Across backends or architectures, only the AS241 `log` and `sqrt` evaluations
may change the final bits.

## Reactant

Reactant may bake an immutable generator's key and position into a compiled
executable. A changed key or position may require a new compilation. Compile
or cache functions outside key- or position-varying loops, and reuse an
executable only for the key and position used to compile it.
