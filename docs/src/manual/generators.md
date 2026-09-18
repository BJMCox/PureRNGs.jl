# Generators and streams

## Choose a generator

A generator holds a key, a bit position, a zero-size backend token, and the decoded output block at its position. It is an immutable, isbits value.
Consecutive scalar draws inside one block reuse that decoded block.

`Philox4x32` is a useful starting point. Benchmark your workload before choosing another generator.

| Generator | Key bits | Output block | Stream capacity, bits per key |
|:--|--:|:--|:--|
| `Philox2x32` | 32 | 2 × 32 | 2^62 |
| `Philox4x32` | 64 | 4 × 32 | 2^71 |
| `Philox2x64` | 64 | 2 × 64 | 2^71 |
| `Philox4x64` | 128 | 4 × 64 | 2^136 |
| `Threefry2x32` | 64 | 2 × 32 | 2^62 |
| `Threefry4x32` | 128 | 4 × 32 | 2^71 |
| `Threefry2x64` | 128 | 2 × 64 | 2^71 |
| `Threefry4x64` | 256 | 4 × 64 | 2^136 |
| `ChaCha` | 256 | 16 × 32 | 2^73 |

`Philox4x32R7` and `Threefry4x64R13` are the same generators with the smallest round counts that pass BigCrush in Random123, seven and thirteen instead of ten and twenty.
They share every method with the full-round types, produce different streams, and trade statistical margin for speed.

`ChaCha` runs the ChaCha stream cipher core with twelve rounds, the common choice for random number generation. `ChaCha12` names that default explicitly, `ChaCha8` is the faster reduced count, and `ChaCha20` is the cipher's full round count.
The 64-bit block counter occupies the two counter words of the ChaCha state, and the nonce words carry the key derivation tag.

The word width is not the counter width. A 32-bit generator does not stop after 2^32 draws.

Construct a generator from a nonnegative integer fitting its key, or an exact tuple of key words.
New generators start at position zero on the CPU.

An integer seed splits into little-endian key words with no mixing, so adjacent
seeds give adjacent keys. This is safe because the cores are keyed bijections
whose outputs decorrelate adjacent keys: across all twelve configurations, the
mean pairwise Hamming distance of the first outputs for seeds 1 to 4 lies
between 31.0 and 32.9 of 64 bits. The mapping is part of the reproducibility
contract, so a given seed always yields the same key.

## Derive streams for separate purposes

```@example keys
using PureRNGs, Random

root = Threefry4x32(123456)
initialization = subrng(root, 1)
measurement = subrng(root, 2)

children = splitrng(root, 8)       # Vector
left, right = splitrng(root)      # Two-element tuple
fixed = splitrng(root, Val(3))    # Tuple with compile-time length
@assert subrng(root, 1) == initialization
```

Derivation leaves the parent unchanged and ignores its current position. Children start at position zero and retain the backend.

`splitrng(root, n)` takes `threaded=true` by default and derives large child vectors in parallel, with the same children as `threaded=false`.

Repeated derivation with the same parent and purpose returns the same key. It does not allocate a fresh stream automatically.

Splitting in a loop is the common mistake. The parent advances, the derivation
ignores its position, and every iteration gets the same child:

```@example keys
function repeated_children(rng, n)
    firsts = Float64[]
    for _ in 1:n
        child, _ = splitrng(rng)
        push!(firsts, randat(child, Float64, 1))
        _, rng = rand_next(rng, Float64)
    end
    return firsts
end

repeated_children(Threefry4x32(123456), 3)
```

Derive from the loop index instead:

```@example keys
function indexed_children(root, n)
    return [randat(subrng(root, i), Float64, 1) for i in 1:n]
end

children_of_root = indexed_children(Threefry4x32(123456), 3)
@assert allunique(children_of_root)
children_of_root
```

Use stable integer purpose IDs. `subrng` reduces them modulo 2^64, so IDs differing by 2^64 alias.
Splitting and purpose IDs use separate derivation namespaces.

Child keys are core output. Across `n` program-wide derivations with `k` key
bits, two derivations collide with probability about `n^2 / 2^(k+1)`, and a
collision makes both child subtrees identical. For many children, prefer a
generator with at least 128 key bits. Distinct keys do not guarantee distinct
output values.

Use integer counts for ordinary splitting. `Val(N)` exposes a fixed tuple length to the compiler and suits small, statically known counts.

## Address a draw directly

```@example keys
values = rand(root, Float32, 8)
@assert values == [randat(root, Float32, i) for i in 1:8]
```

Indices start at one, relative to the generator's current position.
`randnat` and `randexpat` provide the corresponding normal and exponential draws.

Addressed draws do not change state. Keep the operation and result type fixed when assigning indices to parallel work.

## Save and restore a position

```@example keys
key = rngkey(root)
position = rngposition(root)
restored = Threefry4x32(key, position)
@assert restored == root
```

`rngkey` returns the key words. `rngposition` returns the number of consumed bits.
The two-argument constructor rebuilds the generator on the CPU at that position. Apply a device afterwards if needed.

## Exhaustion

Continuation advances a bit position. It does not split keys or silently wrap.

Package-owned operations check capacity before writing their destination. Exhaustion throws an error.
Choose a new key explicitly when a stream ends.

Uniform draws consume one bit for `Bool`, 24 for `Float32`, and 53 for `Float64`.
Integer draws consume their type's width. Normal draws consume 23 or 52 bits.
Exponential draws consume 24 or 53 bits.

Mixed result types share one cursor without alignment padding. Chunk large workloads to limit memory, not to reset the stream.
