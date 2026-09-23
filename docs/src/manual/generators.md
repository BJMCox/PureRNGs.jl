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

Six aliases name the same generators at a reduced round count. Five use the smallest count that Salmon, Moraes, Dror, and Shaw (2011, Table 2) report as passing BigCrush for that shape.
`Threefry4x64R13` uses thirteen rounds, one more than the twelve in that table and the reduced round count in Random123's known-answer tests.
`Philox2x32` and `Threefry2x32` have no published minimum and so have no alias.

| Alias | Family | Rounds | Default rounds |
|:--|:--|--:|--:|
| `Philox4x32R7` | `Philox4x32` | 7 | 10 |
| `Philox2x64R6` | `Philox2x64` | 6 | 10 |
| `Philox4x64R7` | `Philox4x64` | 7 | 10 |
| `Threefry2x64R13` | `Threefry2x64` | 13 | 20 |
| `Threefry4x32R12` | `Threefry4x32` | 12 | 20 |
| `Threefry4x64R13` | `Threefry4x64` | 13 | 20 |

They share every method with the full-round types, produce different streams, and trade statistical margin for speed.

`ChaCha` runs the ChaCha stream cipher core with twelve rounds, the common choice for random number generation. `ChaCha12` names that default explicitly, `ChaCha8` is the faster reduced count, and `ChaCha20` is the cipher's full round count.
The 64-bit block counter occupies the two counter words of the ChaCha state, and the nonce words carry the key derivation tag.

The word width is not the counter width. A 32-bit generator does not stop after 2^32 draws.

Construct a generator from a nonnegative integer fitting its key, or an exact tuple of key words.
New generators start at position zero on the CPU.

An integer seed splits into little-endian key words with no mixing, so adjacent
seeds give adjacent keys. The Random123 and ChaCha designs accept every key value
and do not require random keys, so adjacent seeds are safe. The mapping is part
of the reproducibility contract, so a given seed always yields the same key.

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

`splitrng(root, n; threaded=true)` derives large child vectors in parallel, with the same children as the serial default.

Repeated derivation with the same parent and purpose returns the same key. It does not allocate a fresh stream automatically.

Splitting in a loop is the common mistake. The parent advances, the derivation
ignores its position, and every iteration gets the same child:

```@example keys
function repeated_children(rng, n)
    firsts = Float64[]
    for _ in 1:n
        child, _ = splitrng(rng)
        push!(firsts, rand_at(child, Float64, 1))
        _, rng = rand_next(rng, Float64)
    end
    return firsts
end

repeated_children(Threefry4x32(123456), 3)
```

Derive from the loop index instead:

```@example keys
function indexed_children(root, n)
    return [rand_at(subrng(root, i), Float64, 1) for i in 1:n]
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
@assert values == [rand_at(root, Float32, i) for i in 1:8]
```

Indices start at one, relative to the generator's current position.
`randn_at` and `randexp_at` provide the corresponding normal and exponential draws.

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

Uniform draws consume one bit for `Bool`, 11 for `Float16`, 24 for `Float32`, and 53 for `Float64`.
Integer draws consume their type's width, from 8 to 128 bits. A complex draw consumes a real draw and then an imaginary draw.
Normal and exponential draws consume 23 bits for `Float16` and `Float32` and 52 for `Float64`; a `Float16` value is the `Float32` value on the same bits, rounded once.
Integer ranges consume 64, 128, or 192 bits, depending on the span.

Mixed result types share one cursor without alignment padding. Chunk large workloads to limit memory, not to reset the stream.
