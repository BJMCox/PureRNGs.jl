# Splitting keys

Use counter continuation for sequential draws. Derive a new key only when a
job needs an independent role or an explicit chunk.

```julia
using PureRNGs

root = Philox4x64(123)
proposal_rng = subrng(root, 1)
resampling_rng = subrng(root, 2)
chunk_id = 17
chunk_rng = subrng(root, chunk_id)
children = splitrng(root, 8)
```

Keep purpose identifiers stable. The same parent key and purpose always
produce the same child. Derivation ignores the parent position, preserves its
device, and starts every child at position zero. It never changes the parent.

`splitrng(rng, n)` is the ordinary vector API. Use
`splitrng(rng, Val(N))` only when a static tuple helps allocation-free or GPU
code.

Child keys are core output and can collide. Across `n` program-wide
derivations with `k` key bits, the collision probability is about
`n^2 / 2^(k+1)`. A collision makes both child subtrees identical.

Use `Philox4x64`, `Threefry4x32`, `Threefry2x64`, or `Threefry4x64` for
per-particle or per-proposal derivation at scale. Each has at least 128 key
bits. The 32-bit key space of `Philox2x32` makes collisions likely beyond a
few thousand derivations.

Use `subrng(root, chunk_id)` for explicit large-job chunks. Counter exhaustion
does not create a chunk or derive a key automatically.
