# API reference

The manual explains workflows and backend limits. These docstrings describe exported entry points.

## Generators

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
Philox4x32R7
Threefry4x64R13
ChaCha
ChaCha8
ChaCha12
ChaCha20
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

## State access

```@docs
rngkey
rngposition
```

## Population sampling

```@docs
randsample
randsample_next
randsample!
randsample_next!
WeightTable
```

## Mutable interoperability

```@docs
StatefulRNG
```

## Errors

```@docs
StreamExhausted
```
