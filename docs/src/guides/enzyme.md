# Enzyme

Loading Enzyme activates differentiation rules for the package-owned uniform,
normal, and exponential fills. Immutable RNGs are inactive values and must be
passed as `Const`. Enzyme differentiates code around the generated values.

```julia
using Enzyme
using PureRNGs
using Random

function sample_sum!(rng, destination, scale)
    randn!(rng, destination; threaded = false)
    return scale * sum(destination)
end

rng = Philox4x32(1234)
values = zeros(Float64, 16)
shadow = fill(1.0, 16)

derivative = only(
    autodiff(
        Reverse,
        sample_sum!,
        Active,
        Const(rng),
        Duplicated(values, shadow),
        Active(2.0),
    ),
)

@assert derivative[3] ≈ sum(values)
@assert all(iszero, shadow)
```

The fill is the primal operation. It runs exactly once. Because the fill
overwrites its destination, the rule zeros every destination shadow lane after
a successful primal. Arithmetic that uses the generated values keeps its
ordinary derivatives.

## Supported fills

The immutable rules cover `Random.rand!`, `Random.randn!`, `Random.randexp!`,
`rand_next!`, `randn_next!`, and `randexp_next!` for `Float32` and `Float64`
destinations. Continuation fills return one advanced inactive RNG with the
primal destination and its matching shadow alias.

The same three `Random` fills support concrete host `Array` destinations through
`StatefulRNG`. The bridge remains effectful and advances its held RNG exactly
once. It is not an inactive type.

Scalar draws, ranges, sampling, and fixed-distribution methods are outside the
Enzyme rule surface. A failed primal leaves destination shadows unchanged.

CPU and CUDA are conformance targets. AMDGPU support is preview. Metal and
Reactant are outside this Enzyme contract. Device fills retain the ordinary
same-backend placement rules.
