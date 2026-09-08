# API reference

The manual explains workflows and backend limits. These docstrings describe exported entry points.

## Generator families

```@docs
AbstractPureRNG
Philox2x32
Philox4x32
Philox2x64
Philox4x64
Threefry2x32
Threefry4x32
Threefry2x64
Threefry4x64
```

## Continuation draws

```@docs
rand_next
rand_next!
randn_next
randn_next!
randexp_next
randexp_next!
```

## Addressed draws

```@docs
randat
randnat
randexpat
```

## Key derivation

```@docs
splitrng
subrng
```

## Population sampling

```@docs
randsample
randsample_next
```

## Mutable interoperability

```@docs
StatefulRNG
```
