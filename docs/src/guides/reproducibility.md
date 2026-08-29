# Reproducibility

Integer and uniform draws are bitwise identical across supported backends for
the same family, key, family region, and logical-bit position. Normal input
bits and midpoint values have the same guarantee.

Complete software identity records the Julia version and every used backend
package, compiler, toolkit, and math-library version. Final normal values are
bitwise reproducible within one backend, architecture, and complete software
identity. They may vary across any of these only through AS241's `log` and
`sqrt` evaluations, except under the Reactant rules below.

Exponential raw integers and lattice inputs are bitwise identical across
supported backends. Outside Reactant execution, the CPU-token atanh transform
has no architecture or software identity exception because its written IEEE
operations define the exact result. CUDA, AMDGPU, and Metal tokens use
`Base.log`. Their final values may vary by backend, architecture, and complete
software identity. Outside the Reactant mapping rule below, derived exponential
values may differ only through that primitive difference.

Reproduce a run by keeping the family, seed or key, derivation identifiers, and
draw order fixed. Use addressed draws when work order may change. See
[Immutable workflows](../tutorials/immutable-workflows.md) and
[Splitting and devices](../tutorials/splitting-and-devices.md) for examples.

## Reactant

Convert an immutable generator with `Reactant.to_rarray`. The result stores the
key and position as runtime state. One executable accepts other converted
generators with the same family, backend token, Reactant execution backend,
state type, shape, and sharding while their keys and positions differ.

Reactant CPU remains bitwise exact against eager CPU arithmetic for generator
input and returned generator state, extracted and signed integers, Bool values,
ranges, child keys, uniform values, normal inputs and midpoints, final normal
values, exponential lattice values, final exponential values, and their
addressed variants. On Reactant CUDA, the same list remains bitwise exact except
for the two final transformed values. CUDA lowering may shift final normal and
exponential values without a cross-eager equality or ULP bound. The compiled
and eager results must both be zero or both nonzero. Their signs must match,
including the sign of zero. Conformance rejects non-finite values, and repeated
execution remains bitwise stable under one recorded software identity.

Uniform, Normal, Exponential, and Bernoulli mapping arithmetic is native to
Reactant. The corresponding formulas are a separately rounded multiplication
then addition for Uniform, `muladd` for Normal, multiplication for Exponential,
and comparison for Bernoulli. Each compiled mapping must match its formula
evaluated directly in the same executable and repeat bitwise, but has no
cross-eager mapping guarantee or ULP bound. DiscreteUniform remains bitwise
exact against eager execution.

Compile once, then pass the returned carrier back into the same executable:

```julia
using PureRNGs
using Reactant

function step(rng)
    rng, value = rand_next(rng, Float64)
    return rng, value
end

function run_compiled()
    carrier = Reactant.to_rarray(Philox4x32(1))
    compiled_step = Reactant.@compile sync = true step(carrier)

    for seed in 1:4
        carrier = Reactant.to_rarray(Philox4x32(seed))
        for _ in 1:100
            carrier, value = compiled_step(carrier)
        end
    end
    return carrier
end

carrier = run_compiled()
```

This loop changes the key between seeds and advances the position within each
seed without another `Reactant.@compile` call.

The compiled carrier performs no capacity or exhaustion checks. The caller
must prove that every requested draw span fits before invoking the executable.

Compiled integer-range draws accept static `OrdinalRange` values and integer
`LinRange` values. Other `AbstractRange` types remain available to eager draws
but are outside the compiled carrier API.

A plain Julia integer passed to `subrng` is static during compilation. The
parent key remains dynamic, so the child key changes with each supplied
carrier. Changing the purpose may compile a different executable.
