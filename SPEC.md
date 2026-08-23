# PureRNGs version 0 specification

Status: normative specification, revision 9
Date: 2026-08-23

## 1. Reading rules

This document specifies the complete version 0 package. It is ordered so
that every section depends only on earlier sections.

Normative keywords MUST, MUST NOT, and MAY have their RFC 2119 meanings.
Every normative statement carries an id like [R12]. A conforming
implementation satisfies every requirement. Section 12 maps requirements to
acceptance tests.

- [R1] Closed world. The package contains exactly what this document
  states. The implementation MUST NOT add any exported name, public method,
  default value, or convenience overload beyond those stated here. Section
  12 lists every exported symbol. Section 11 lists every public error.
- [R2] Value oracle. `PureRNGsTestbed.jl` (local checkout,
  `~/Code/scratch/PureRNGsTestbed.jl`) at commit
  `7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d` fixes stream values. Every
  construct this document shares with the testbed MUST produce bitwise
  identical values to that commit. Oracle vectors captured for the test
  suite record this hash beside them. The governing testbed files are
  `src/philox.jl`, `src/threefry.jl`, `src/derive.jl`, `src/families.jl`,
  `src/uniform.jl`, `src/normal.jl`, `src/integers.jl`,
  `src/cursor.jl`, `src/cursor_sampling.jl`, `src/cursor_fill.jl`,
  `src/packed_bits.jl`, `src/step.jl`.
- [R3] Escalation. If a requirement seems to contradict another
  requirement or the testbed, stop and report the conflict. Do not resolve
  it by judgment.

## 2. The model

A generator is an immutable `isbits` value holding a key, one raw-word
counter position, and a device binding. The counter position is functional
state: pure draws read it, continuation draws return a new generator with an
advanced position.

- [R4] `AbstractPureRNG` is the sole immutable-generator supertype.
  The eight family types directly subtype it; no intermediate Philox or
  Threefry abstract type exists. Every family type is an immutable `isbits`
  struct parameterized by device. Its contents are the native key words, the
  [R53] counter position, and the [R38] `isbits` MLDataDevices device value.
  Construction uses `CPUDevice()`. The binding never influences stream
  values, and serialization drops it ([R45]).
- [R5] Draws are pure: the same call on the same generator returns the same
  values on every call and every backend, subject only to [R43]. A pure draw
  reads the held position and leaves the generator unchanged.
- [R6] Counter continuation is the default source of fresh sequential
  randomness. `rand_next`, `randn_next`, and `randsample_next` return the
  advanced generator. `splitrng` and `subrng` explicitly derive new keys.
- [R61] Fixed-work law. Every version 0 random-generation call consumes a
  raw-word count determined only by its method, result types, shapes, range
  spans, population size, weight presence, and `k`. Random values never
  control the word count or trigger a retry. The implementation contains no
  rejection sampler. Weighted sorting and population iteration remain
  bounded by their input sizes.
- [R7] Tier 1 counter partition: key-derivation counters carry the
  derivation tag in the top counter word, draw counters never do. Child
  keys therefore come from counter addresses no draw ever reads. This is
  address separation: equal output words at distinct addresses remain
  possible by chance ([R22]).
- [R8] Tier 2 counter partition: each draw family (bits/uniform, normal,
  integer range)
  reads a disjoint counter region selected by a family word. `rand` and
  `randn` on one key are therefore domain-separated pseudorandom streams.
  All families read the same held raw-word position.
- [R9] Tag values, subtag values, family words, and region layout equal the
  testbed values: `DERIVE_TAG = 0xC0FFEE00`, `SPLIT_SUBTAG = 0x00000000`,
  `FOLD_SUBTAG = 0x00000001`, `THREEFRY_FOLD_INDEX = 0xffffffff` (the
  narrow-layout fold namespace), family words as in `src/families.jl`.
  These values are frozen for the life of stream-law version 1.
- [R10] Family words and derivation-region indices not assigned by this
  document are reserved by the stream law and MUST stay unassigned in
  version 0, so later stream-law-compatible extensions can claim them.

- [R51] The logical address of a drawn word is (key, family, raw-word
  position). The address involves no Julia task, thread, GPU lane, launch
  shape, worker, or storage slot.

## 3. Families, construction, seeding

Version 0 ships the eight standard Random123 shapes:

| Family | Key type | Key bits | Counter bits | Rounds |
| --- | --- | ---: | ---: | ---: |
| `Philox2x32` | `NTuple{1,UInt32}` | 32 | 64 | 10 |
| `Philox4x32` | `NTuple{2,UInt32}` | 64 | 128 | 10 |
| `Philox2x64` | `NTuple{1,UInt64}` | 64 | 128 | 10 |
| `Philox4x64` | `NTuple{2,UInt64}` | 128 | 256 | 10 |
| `Threefry2x32` | `NTuple{2,UInt32}` | 64 | 64 | 20 |
| `Threefry4x32` | `NTuple{4,UInt32}` | 128 | 128 | 20 |
| `Threefry2x64` | `NTuple{2,UInt64}` | 128 | 128 | 20 |
| `Threefry4x64` | `NTuple{4,UInt64}` | 256 | 256 | 20 |

- [R11] Core algorithms are the Random123 reference implementation,
  D. E. Shaw Research release v1.14.0, files `philox.h` and `threefry.h`,
  with that release's standard multiplier, Weyl, and rotation constants
  and its key-bump schedule. This pins every output word of all eight
  cores. Every core reproduces the release's `kat_vectors` known-answer
  entries for its shape at the listed round count. These are the only
  round counts in the package. For `Philox4x32` and `Threefry2x32` the
  testbed ([R2]) realizes the same cores; a disagreement between testbed
  and KATs is an [R3] stop.
- [R12] Layout classes. Families with counter bits >= 128 use the wide
  layout, families with 64 counter bits (`Philox2x32`, `Threefry2x32`) the
  narrow layout. `Philox2x32` copies the testbed `Threefry2x32` narrow
  rules exactly.
- [R12a] Wide draw counters by word width, low word first (testbed
  `Philox4x32` placement: family in word 3, zero in word 4):
  four 32-bit words: `(block_lo32, block_hi32, family, 0)`;
  four 64-bit words: `(block_lo64, block_hi64, family, 0)`;
  two 64-bit words: `(block, UInt64(family))` — the block index fills the
  low word, the family word sits zero-extended in the high word with its
  top 32 bits zero. Wide derivation counters by word width:
  four 32-bit words: `(i_lo32, i_hi32, subtag, DERIVE_TAG)`;
  four 64-bit words: `(i, subtag, 0, DERIVE_TAG)` zero-extended;
  two 64-bit words: low word `i`, high word
  `(UInt64(DERIVE_TAG) << 32) | subtag`. Draw counters keep the top 32
  bits of their high word zero, so derivation and draw addresses stay
  disjoint in every wide family.
- [R12b] Narrow layout (testbed `Threefry2x32` rules): draws carry the
  family word in the top byte of the high counter word and the block index
  below it, capping draw block indices at `2^56 - 1` per family.
  Derivation uses the whole high word: `derive_child(rng, i)` runs one
  core block with counter `(UInt32(i), DERIVE_TAG)` and no subtag. Valid
  child indices are 0 through `2^32 - 2`; index `2^32 - 1` is the reserved
  fold namespace (`THREEFRY_FOLD_INDEX`) and throws `ArgumentError`.
  `subrng` chains two blocks: the first derives the namespace key with
  counter `(THREEFRY_FOLD_INDEX, DERIVE_TAG)`, the second consumes the
  full 64 data bits as its counter.
  When a narrow family's key is narrower than its block output —
  `Philox2x32`, one key word from a two-word block — every derivation
  child key is output word 1, the low word: the `derive_child` child,
  the `subrng` namespace key, and the final `subrng` child. Output word
  2 is discarded. The narrow rule stays one child per block; a narrow
  family never packs two children into one block.
- [R13] Oracle scope. `Philox4x32` and `Threefry2x32` streams equal the
  testbed value for value ([R2]). The other six families are fully
  determined by [R11], [R12a], [R17]-[R19], and [R27]; their golden vectors,
  frozen in-repo before release, confirm the specification rather than
  define it.
- [R14] Constructors, for every family `F` with key type `K`: `F(seed::
  Integer)` and `F(key::K)`. Every generator names its family at
  construction; the package defines no default generator. Both constructors
  create a CPU-bound generator at raw-word position zero.
- [R15] An integer seed satisfies `0 <= seed < 2^key_bits` for the family's
  key bits. Violations throw `ArgumentError` (section 11). Seeds are never
  truncated or reduced.
- [R16] Seed-to-key mapping: the seed splits into native key words from low
  to high by numeric significance. `Philox4x32(1234)` has key
  `(0x000004d2, 0x00000000)`. The tuple constructor stores the tuple as
  given.

## 4. Key derivation

```julia
splitrng(rng::R)              :: NTuple{2,R}    # = splitrng(rng, Val(2))
splitrng(rng::R, ::Val{N})    :: NTuple{N,R}
splitrng(rng::R, n::Integer)  :: Vector{R}
subrng(rng::R, purpose::Integer) :: R
```

- [R17] `splitrng` derives child `i` (zero-based internally) in the tagged
  derivation region. Wide families use `SPLIT_SUBTAG` in the [R12a]
  derivation counter with derivation index `i >> c`, where `2^c` child keys
  fit one output block; children take consecutive key-width word groups of
  the block output, low group first. For `Philox4x32` this equals testbed
  `derive_child`: one block per two children, index `i >> 1`, even `i`
  takes output words 1-2, odd `i` words 3-4. Narrow families follow
  [R12b], one child per block, and for `Threefry2x32` equal the testbed.
- [R18] `subrng(rng, purpose)` derives one child from `purpose % UInt64`. Wide
  families use `FOLD_SUBTAG` in the [R12a] derivation counter with `purpose`
  as the derivation index; the child key is the low key-width word group
  of the output. For `Philox4x32` this equals testbed `subrng`. Narrow
  families use the [R12b] two-block chain, equal to the testbed for
  `Threefry2x32`.
- [R19] [R11], [R12a], [R12b], [R17], and [R18] fully determine derivation
  for all eight families. The six non-testbed families carry in-repo golden
  vectors ([R13]).
- [R20] Derivation uses only the parent key. It ignores the parent position,
  leaves the parent unchanged, preserves its device binding, and creates
  children at position zero. The same parent key and purpose always produce
  the same `subrng` child. The docstrings state these facts and show stable
  purpose ids for independent roles.
- [R21] `splitrng(rng, n)` throws `ArgumentError` for `n < 0` and returns
  an empty `Vector{R}` for `n = 0`. The `Val{N}` form obeys the same
  bounds: `N` is a non-negative `Int`, and every request form on a narrow
  family throws `ArgumentError` once a child index reaches `2^32 - 1`
  ([R12b]). Documentation presents the integer form as the ordinary API.
  The `Val` form exists for compile-time tuple size, allocation-free code,
  and GPU kernels.
- [R22] Collision statement. The `splitrng` and `subrng` docstrings and the
  documentation's splitting page state: child keys are core output, two
  derivations collide with probability about `n^2 / 2^(k+1)` for `n`
  derivations program-wide and `k` key bits, and a collision makes two
  subtrees identical. They direct at-scale derivation (per particle, per
  proposal) to the four families with key bits >= 128. The `Philox2x32`
  docstring warns that its 32-bit key space makes derivation collisions
  likely beyond a few thousand derivations. The documentation shows
  `subrng(root, chunk_id)` for explicit large-job chunks. No exhaustion path
  creates such a chunk automatically.

## 5. Draws

Result types are exactly `Bool`, `UInt32`, `UInt64`, uniform `Float32`,
uniform `Float64`, normal `Float32`, and normal `Float64`.

The draw API consists of `Base` and `Random` method extensions on
package-owned types:

```julia
Random.rand(rng, ::Type{T})                   :: T
Random.rand(rng, ::Type{T}, dim1::Integer,
          dims::Integer...)                   :: Array{T}
Random.rand!(rng, dest::AbstractArray{T})     :: typeof(dest)
Random.rand(rng, range::AbstractRange{T})     :: T
Random.rand(rng, range::AbstractRange{T},
          dim1::Integer, dims::Integer...)     :: Array{T}
Random.randn(rng, ::Type{T})                  :: T   # T float
Random.randn(rng, ::Type{T}, dim1::Integer,
           dims::Integer...)                  :: Array{T}
Random.randn!(rng, dest::AbstractArray{T})    :: typeof(dest)
rand_next(rng::R)                             :: Tuple{R,Float64}
rand_next(rng::R, dim1::Integer,
          dims::Integer...)                   :: Tuple{R,Array{Float64}}
rand_next(rng::R, ::Type{T})                  :: Tuple{R,T}
rand_next(rng::R, ::Type{T}, dim1::Integer,
          dims::Integer...)                   :: Tuple{R,Array{T}}
rand_next(rng::R, range::AbstractRange{T})    :: Tuple{R,T}
rand_next(rng::R, range::AbstractRange{T},
          dim1::Integer, dims::Integer...)     :: Tuple{R,Array{T}}
rand_next!(rng::R, dest::AbstractArray{T})    :: Tuple{R,typeof(dest)}
randn_next(rng::R)                            :: Tuple{R,Float64}
randn_next(rng::R, dim1::Integer,
           dims::Integer...)                  :: Tuple{R,Array{Float64}}
randn_next(rng::R, ::Type{T})                 :: Tuple{R,T}
randn_next(rng::R, ::Type{T}, dim1::Integer,
           dims::Integer...)                  :: Tuple{R,Array{T}}
randn_next!(rng::R, dest::AbstractArray{T})   :: Tuple{R,typeof(dest)}
randat(rng, ::Type{T}, i::Integer)           :: T
randnat(rng, ::Type{T}, i::Integer)          :: T
```

- [R23] Every primitive draw names its result type explicitly or through a
  typed destination. An integer range supplies its element type. All
  overloads on immutable generators are typed, and the package defines two
  untyped guard methods, `Random.rand(rng)` and
  `Random.randn(rng)` on the immutable generator supertype, that throw
  `ArgumentError` naming the typed form (section 11), so Base's fallbacks
  are never reached. `rand_next(rng, dims...)` and
  `randn_next(rng, dims...)` are the explicit continuation conveniences and
  default to `Float64`, including their scalar zero-dimension forms.
- [R24] `rand!` and `randn!` mutate only the destination and return the
  destination. `rand_next!` and `randn_next!` return
  `(next_rng, destination)`. Every other continuation draw returns
  `(next_rng, value)` with the generator first.
- [R25] Integer and uniform values equal the testbed bits and uniform
  families. Uniform floats lie in [0, 1) and use the testbed's exact bit
  conversion. A `Bool` consumes one logical 32-bit word and is true exactly
  when that word's low bit is one. This rule governs scalar values,
  `Array{Bool}`, and `BitArray` fills.
- [R26] Shape stability: for every generator, result type, and `m <= n`,
  `rand(rng, T, n)[1:m] == rand(rng, T, m)` holds bitwise, and
  `vec(rand(rng, T, a, b)) == rand(rng, T, a*b)`. Same for `randn` and
  integer ranges. A batch continuation equals chained scalar continuation
  draws and advances by the same raw-word count: one [R53] alignment at the start,
  then consecutive elements, which chained same-width scalars reproduce
  because each element leaves the position aligned for the next.
  Pure array draws have the same values but leave the input generator unchanged.
- [R27] Element packing equals the testbed `_draw` packing rule. A core
  block of `N` words of `W` bits holds `(N * W) / e` elements of bit width
  `e`; each element occupies consecutive `e`-bit lanes of the block
  output, word-minor, low lanes first, inside the result's family region.
  Within one element, the lower-indexed lane supplies the high-order bits,
  as the testbed assembles `UInt64` values
  (`(UInt64(blk[2j-1]) << 32) | UInt64(blk[2j])`).
  Hence `Philox4x32` packs four 32-bit or two 64-bit elements per block,
  `Threefry2x32` two or one, and the 64-bit-word families pack `N` 64-bit
  or `2N` 32-bit elements per block.
- [R62] Logical word order. The stream of a family region is a sequence
  of logical 32-bit words. In 32-bit-word families the logical words are
  the block output words in order. In 64-bit-word families each native
  output word contributes two logical words: the earlier logical word is
  the native word's high 32 bits, the later its low 32 bits. This is the
  [R27] rule read in reverse: the lower-indexed lane supplies the
  high-order bits, so reassembling the two logical words as
  `(UInt64(w1) << 32) | UInt64(w2)` returns the native word. Every [R27]
  element occupies exactly `e / 32` consecutive logical words, a `UInt64`
  or `Float64` element of a 64-bit-word family is exactly one native
  word, and a `UInt32`-class element of a 64-bit-word family at an even
  logical position is the native word's high half. [R28] normal word
  counts are counted in logical 32-bit words and equal the testbed
  counts: one for `Float32`, two for `Float64`.
- [R28] Normal generation consumes a fixed number of native words per value
  with no cache and no rejection, so [R26] holds for `randn`. Algorithm and
  word count equal the testbed normal family.
- [R29] `randat(rng, T, i) == rand(rng, T, n)[i]` for every `n >= i`, and
  `randnat` likewise for `randn`. The index is one-based: element `i`
  occupies the span starting at the
  [R53]-aligned position derived from the generator's held position,
  advanced by `i - 1` element word counts.
  These addressed operations never advance the generator. They throw
  `ArgumentError` for `i < 1` or when the addressed
  word span exceeds [R53].
- [R30] `randat`, `randnat`, `subrng`, `splitrng` with `Val`, and scalar
  pure and continuation draws compile in GPU kernels without allocation,
  dynamic dispatch, or host state, and run on the host with bitwise
  identical results for integer and uniform types.
- [R53] The counter position is the zero-based index of the next logical
  32-bit word, represented as a core block and a lane, plus one terminal
  exhausted value. `Bool`, `UInt32`, and `Float32` consume one logical word.
  `UInt64` and `Float64` consume two logical words. Normal word counts
  equal [R28]. Every draw reserves its span at an alignment equal to its
  logical word count — one, two, or four logical words — computed on the
  zero-based global logical position (testbed `_reserve` with
  `alignment = words`). Alignment padding words are consumed and never
  drawn. A span never crosses its own alignment unit, so a two-word
  element always starts at an even position and, in 64-bit-word
  families, coincides with one native word ([R62]).
  Each draw family applies its own family word to the same position.
  `rand` and `randn` read at the held position without advancing.
  Continuation draws reserve their alignment padding plus their exact
  span and return the advanced generator.
  The last valid reservation returns the exhausted value. A
  later nonempty draw throws `ArgumentError`. No operation wraps the
  position or derives a new key automatically.
- [R54] A fixed-size draw computes its full reservation, including [R53]
  alignment padding, with widened checked arithmetic before generating a
  value or mutating a destination. It throws `ArgumentError` if the
  reservation exceeds the family region. A
  zero-size draw succeeds at every position, including exhaustion,
  provided its result type and family are serviceable on the generator's
  device: the [R41] Metal exclusion is checked before size and throws
  even for a zero-size request. The region contains
  `(max_block + 1) * logical_words_per_block` words, where `max_block` is
  `2^56 - 1` for narrow families and the full [R12a] draw-block range for
  wide families.
- [R55] Integer-range draws support nonempty unit and stepped ranges whose
  element type is a signed or unsigned integer of at most 64 bits, excluding
  `Bool`. They use `FAMILY_RANGE`. Unit ranges equal the fixed-work testbed
  algorithm in `src/integers.jl`: spans through `2^32` consume one `UInt64`
  slot and use its pinned 64-bit multiply-high reduction; wider spans consume
  two slots and use the pinned 128-by-64 reduction. Stepped ranges apply the
  same reduction to their length and map the selected zero-based offset by
  range indexing without materialization. The testbed's documented maximum
  relative preimage bias applies. Empty ranges throw `ArgumentError`.
- [R50] Statistical quality: the bits and uniform streams of every family
  pass TestU01 SmallCrush at minimum on the release architecture, run on
  sequential output and on `splitrng`-child interleavings. A failure
  blocks release.

## 6. Sampling with replacement

```julia
randsample(rng::AbstractPureRNG, iter)             :: AbstractVector
randsample(rng::AbstractPureRNG, iter, k::Integer) :: AbstractVector
randsample(rng::AbstractPureRNG, iter,
           weights::AbstractVector{<:Real})             :: AbstractVector
randsample(rng::AbstractPureRNG, iter,
           weights::AbstractVector{<:Real},
           k::Integer)                   :: AbstractVector

randsample_next(rng::R, iter)                  :: Tuple{R,AbstractVector}
randsample_next(rng::R, iter, k::Integer)      :: Tuple{R,AbstractVector}
randsample_next(rng::R, iter,
                weights::AbstractVector{<:Real}) :: Tuple{R,AbstractVector}
randsample_next(rng::R, iter,
                weights::AbstractVector{<:Real},
                k::Integer)              :: Tuple{R,AbstractVector}
```

- [R56] These eight methods are the complete sampling surface. Sampling is
  with replacement. A form without `k` returns as many samples as the
  population contains. A form with `k` returns exactly `k` samples.
  `randsample` reads the held position and leaves the generator unchanged.
  `randsample_next` returns `(next_rng, values)`.
- [R57] The population is any finite iterable for which
  `MLDataDevices.get_device(iter)` equals the generator device or returns
  `nothing`. `nothing` means device-agnostic and is compatible with every
  generator device; `AbstractRange` has this status. Any other device result
  throws `ArgumentError` before generation. Vectors are indexed directly.
  Integer unit and stepped ranges are mapped by position without
  materialization. Every other device-agnostic iterable is materialized
  exactly once on the generator device. The result is allocated on that
  device and its element type comes from the indexed or materialized
  population. Population cardinality MUST fit a positive
  `UInt64` when a sample is requested. A no-`k` form additionally requires
  cardinality to fit `Int` because it returns that many elements.
- [R58] Unweighted sampling uses the fixed-work [R55] range reduction to
  select a zero-based population index in O(k). Population sizes through
  `2^32` consume one `UInt64` slot per sample and use the pinned 64-bit
  multiply-high reduction. Wider populations consume two slots and use the
  pinned 128-by-64 reduction. The [R55] preimage-bias bound applies. This
  path never constructs weights and never retries.
- [R59] Weighted forms accept only a raw `AbstractVector{<:Real}` aligned
  with the population. Its MLDataDevices device must equal the generator
  device or be device-agnostic. Any other device result throws
  `ArgumentError` before generation. They do not accept a weight wrapper.
  They convert weights to `Float64`, validate them, and compute the total in
  population order. A batch draws `k` `Float64` thresholds from
  `FAMILY_RANGE`, records
  their original indices, maps each uniform `u` to
  `min(u * total, prevfloat(total))`, sorts by
  `(threshold, original_index)`, scans the weights once, and restores draw
  order. Selection chooses the first cumulative total strictly greater than
  the threshold. This algorithm is normative.
- [R60] Sampling validates the complete population, `k`, and weights before
  drawing. Device validation occurs first. It throws `ArgumentError` for
  negative `k`, an empty population with positive `k`, a weight-length
  mismatch, a non-finite or negative weight, or a non-finite or non-positive
  total. Zero weights are valid.
  Empty unweighted sampling with `k == 0` returns an empty vector. Batch
  results equal chained one-sample continuation calls and obey prefix
  stability. Every sampling call has a fixed raw-word count, preflights its
  full reservation under [R54], and never retries. Counter exhaustion throws
  `ArgumentError` without returning a partial result or a continuation.

## 7. The Julia bridge

- [R31] The immutable generator types do not subtype `Random.AbstractRNG`.
  A mutable-RNG consumer that receives an immutable generator gets a
  `MethodError`, which is the designed failure: drawing twice from an
  immutable key would silently repeat values.
- [R32] `StatefulRNG` is the package's single mutable type and single
  `Random.AbstractRNG` subtype. `StatefulRNG(rng)` wraps any immutable
  generator. Its sole field is one concrete immutable generator. The bridge
  is host-only: construction preserves the key and position while rebinding
  the held generator to the CPU, every bridge draw runs on the host, and
  results are host values. Device work goes through the immutable API of
  section 8.
- [R33] Every bridge draw uses the corresponding counter continuation from
  section 5. It replaces the held generator with `next_rng` only
  after a successful call and returns the value. Thus bridge draws equal a
  chained `rand_next` or `randn_next` sequence. They do not split keys. The
  cursor-form `nextrand` chain in testbed `src/step.jl` is the
  reference-family oracle.
- [R34] `StatefulRNG` supports the `Random` surface Distributions.jl needs
  on CPU: `rand(m)`, `rand(m, T)`, `rand(m, T, dims...)`, `rand!(m, A)`,
  `rand(m, range)`, `rand(m, range, dims...)`, `randn(m)`, `randn(m, T)`,
  `randn(m, T, dims...)`, `randn!(m, A)`, and `Random.seed!(m, seed)`.
  Seeding replaces the held generator with a fresh zero-position family
  constructor call. Untyped calls use Julia's standard `Float64` default.
  Generic `Random` dispatch reaches these methods through exactly five
  primitive hook methods:
  `Random.rand(m::StatefulRNG, ::Random.SamplerType{Bool})`,
  `Random.rand(m::StatefulRNG, ::Random.SamplerType{UInt32})`,
  `Random.rand(m::StatefulRNG, ::Random.SamplerType{UInt64})`,
  `Random.rand(m::StatefulRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float32}})`,
  `Random.rand(m::StatefulRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float64}})`.
  The package also defines only the required
  `Random.Sampler(::Type{<:StatefulRNG}, range::AbstractRange{T},
  ::Random.Repetition)` forwarding for [R55] integer types and the matching
  `Random.rand(m::StatefulRNG, sampler)` method. Normal draws bypass
  `Sampler`. These hooks are exempt from [R1] and MUST NOT widen behavior
  beyond this section.
- [R35] `copy(m)` returns an independent wrapper with the same held
  generator, giving exact replay. Serialization stores the held generator
  as in section 10.

- [R52] The bridge's complete method set is [R34] plus `copy` ([R35]) and
  serialization (section 10). Pure code creates a `StatefulRNG` and hands
  it derived children; keys stay pure-side.

## 8. Device placement

Dependencies: MLDataDevices (device selection) and KernelAbstractions (fill
kernels) are normal dependencies [R36]. CUDA.jl, AMDGPU.jl, and Metal.jl
are weak dependencies with package-owned extensions [R37].

```julia
dev = gpu_device(; force=true)
rng = dev(Philox4x32(1234))
xs  = rand(rng, Float32, 1_000_000)   # device array
```

- [R38] Applying an MLDataDevices device binds a generator to that device.
  The generator is `isbits`, so binding moves no data. `splitrng`, `subrng`,
  and every continuation preserve the binding. Applying another device
  preserves the key and position and changes only the binding.
- [R39] Placement invariant: device-closed for data, host-transparent for
  logical state. Section 5 allocating draws on a device-bound generator
  return arrays allocated on that device and generate every value there.
  `rand!`/`randn!` require the destination on the generator's device and
  throw `ArgumentError` on mismatch. Scalar draws, `randat`, `randnat`,
  `splitrng`, `subrng`, and scalar continuation draws compute where they are
  called — kernel registers or host arithmetic — and no generator operation
  performs a device-to-host copy or forces synchronization. Section 6
  sampling requires device-aligned inputs and allocates its result on the
  generator device.
- [R40] Fills launch asynchronously with the ordering semantics of the
  KernelAbstractions backend queue and do nothing for empty destinations.
- [R41] Backend tiers. CPU and CUDA pass the full section 13 suite and
  block release. AMDGPU is a preview: the suite runs and failures are
  documented without blocking. Metal is experimental and serves the 32-bit
  families with `Bool`, `UInt32`, `UInt64`, and `Float32` results. On a
  Metal-bound generator, allocating draws, destination fills, and
  Metal-kernel draws throw `ArgumentError` (section 11) for `Float64`
  results or 64-bit-word families: Metal has no `Float64`, the 64-bit
  Philox multiplies need 128-bit emulation Metal.jl lacks, and the 64-bit
  Threefry integer paths are unvalidated on Metal.jl and excluded pending
  validation. Host-side scalar draws follow [R39] regardless of binding.
  The exclusion applies before size checks: a zero-size allocating draw,
  fill, or kernel draw for an excluded result type or 64-bit-word family
  throws the same `ArgumentError`.
- [R42] Reactant: the generator representation traces under
  `Reactant.@compile` as data, so a changed key or position does not force a
  recompile. A compilation test gates release. If a plain `isbits` value
  fails the test, the Reactant extension provides a traced carrier with
  unchanged stream values.

## 9. Cross-backend guarantees

- [R43] Integer and uniform results are bitwise identical across all
  supported backends for the same family, key, family region, and raw-word
  positions, with no tolerance. Normal results are bitwise reproducible within
  one backend; across backends they MAY differ through transcendental
  functions, and the documentation states this single exception.

## 10. Stream law and serialization

The stream law is the set of value-determining rules: [R9] tags and
regions, [R11] cores and rounds, [R12] layouts, [R16] seed mapping,
[R17]-[R19] derivation, [R25] conversion, [R27] packing, [R28] normal
algorithm, [R53] counter continuation, [R62] logical word order,
[R55] integer ranges, [R58]-[R59]
sampling, and the fixed-work rule [R61].

- [R44] The package defines a stream-law identifier, one integer constant.
  Version 0 ships identifier 1. Any change to a value-determining rule
  increments it.
- [R45] The serialized form of an immutable generator is exactly (law
  identifier, family tag, key words, counter position). The position
  encoding distinguishes every block and lane and the exhausted value.
  Family tags, frozen: `Philox2x32` =
  1, `Philox4x32` = 2, `Philox2x64` = 3, `Philox4x64` = 4, `Threefry2x32`
  = 5, `Threefry4x32` = 6, `Threefry2x64` = 7, `Threefry4x64` = 8. A
  `StatefulRNG` serializes the same payload for its held generator. Deserialized
  generators are CPU-bound; callers rebind explicitly. The hooks are the
  stdlib methods
  `Serialization.serialize(s::Serialization.AbstractSerializer, x::F)` and
  `Serialization.deserialize(s::Serialization.AbstractSerializer,
  ::Type{F})` for the eight family types and `StatefulRNG`. `serialize`
  writes the stdlib type framing via `Serialization.serialize_type(s, F)`
  and then the payload, so a plain `deserialize(io)` dispatches to the
  package's `deserialize(s, ::Type{F})`, which reads the payload, validates
  it ([R46]), and returns the generator — never a raw tuple. The stdlib
  framing carries the Julia type; the tuple is the logical payload. These
  are the only serialization methods in the package.
- [R46] Deserialization of an unsupported law identifier throws
  `ArgumentError` naming the stored and supported identifiers. A family tag,
  key payload, or counter position invalid for the framed type also throws
  `ArgumentError`. Restoration never resumes with a silently changed stream.

## 11. Errors, closed list

The complete set of public-API throws:

| Call | Condition | Error |
| --- | --- | --- |
| `F(seed)`, `Random.seed!(m, seed)` | `seed < 0` or `seed >= 2^key_bits` | `ArgumentError` |
| `splitrng(rng, n)` or `splitrng(rng, Val(N))` | `n < 0`, or `N` is not a non-negative `Int` | `ArgumentError` |
| `splitrng` on a narrow family | child index at or beyond `2^32 - 1` (the reserved fold namespace, [R12b]) | `ArgumentError` |
| `randat`/`randnat` | `i < 1` or addressed span exceeds the remaining family region ([R29]) | `ArgumentError` |
| any nonempty pure, continuation, or destination draw | required span exceeds the remaining family region, including an exhausted generator ([R53], [R54]) | `ArgumentError` |
| integer-range draw | range is empty | `ArgumentError` |
| `rand`/`randn` or their continuation forms with dims | any negative dim | `ArgumentError` (Base array semantics) |
| `rand(rng)` or `randn(rng)` untyped on an immutable generator | always | `ArgumentError` naming the typed form ([R23]) |
| `randsample`/`randsample_next` | `k < 0` | `ArgumentError` |
| `randsample`/`randsample_next` | empty population and `k > 0` | `ArgumentError` |
| `randsample`/`randsample_next` | positive population cardinality exceeds `typemax(UInt64)`, or a no-`k` result length exceeds `typemax(Int)` | `ArgumentError` |
| weighted `randsample`/`randsample_next` | weight-length mismatch, non-finite or negative weight, or non-finite or non-positive total | `ArgumentError` |
| weighted `randsample`/`randsample_next` | weights are not a raw `AbstractVector{<:Real}` | `MethodError` (no method) |
| `randsample`/`randsample_next` | population has a non-agnostic device differing from the generator device | `ArgumentError` |
| weighted `randsample`/`randsample_next` | weights have a non-agnostic device differing from the generator device | `ArgumentError` |
| `rand!`/`randn!` and continuation forms | destination device differs from generator device | `ArgumentError` |
| primitive draw or fill | result type or destination eltype is not a result type | `MethodError` (no method) |
| allocating draw, fill, or kernel draw on Metal | `Float64` result or 64-bit-word family, any size including zero ([R41]) | `ArgumentError`, names Metal |
| deserialization | unsupported law identifier | `ArgumentError`, names stored and supported identifiers |
| deserialization | family tag, key payload, or counter position is invalid for the framed type | `ArgumentError` |
| `Random.AbstractRNG` consumer on an immutable generator | any such call | `MethodError` (designed, [R31]) |

- [R47] The public API throws exactly these deterministic contract errors.
  Outside the list, and outside this requirement, are failures the package
  does not decide: resource exhaustion such as `OutOfMemoryError`, errors
  raised by backend libraries or foreign code, and errors from user
  arguments inside foreign calls. A new contract error condition is a
  specification change: stop and report ([R3]).

## 12. Exports, closed list

- [R48] The package exports exactly nineteen symbols: the eight family
  types `Philox2x32`, `Philox4x32`, `Philox2x64`, `Philox4x64`,
  `Threefry2x32`, `Threefry4x32`, `Threefry2x64`, `Threefry4x64`, and the
  eleven names `splitrng`, `subrng`, `rand_next`, `rand_next!`,
  `randn_next`, `randn_next!`, `randsample`, `randsample_next`, `randat`,
  `randnat`, `StatefulRNG`.
- [R49] Non-exported public surface: the `Base` and `Random` draw methods
  of section 5,
  the `Random` methods of [R34], `copy(::StatefulRNG)`, the serialization
  hooks of section 10, and MLDataDevices device application. Every method
  extension of a foreign function has a package-owned type in a dispatch
  position.

## 13. Conformance

Backend applicability: every row runs on CPU unless its Backend entry says
otherwise. Rows marked CPU+CUDA also run on CUDA. Rows marked with one
backend run only there. Release requires every CPU, CPU+CUDA, CUDA, and
Reactant row green ([R41], [R42]). The Metal row follows the R41
experimental tier and does not block. AMDGPU reruns of the suite follow
the R41 preview tier and do not block.

| Test | Verifies | Backend |
| --- | --- | --- |
| Random123 KATs, all eight shapes, listed rounds | R11 | CPU |
| Testbed oracle: bits, `Bool`, uniform, normal, integer ranges, cursor continuation, `splitrng`, and `subrng` for `Philox4x32` and `Threefry2x32` | R2, R9, R12a, R12b, R13, R17, R18, R25, R27, R28, R33, R53, R55 | CPU |
| Golden vectors for the six non-testbed families, frozen in-repo | R12, R13, R19, R62 | CPU |
| Seed mapping values and bounds sweep | R15, R16 | CPU |
| Construction gives CPU binding and zero position; flat concrete `isbits` hierarchy | R4, R14 | CPU |
| `Bool` scalar, array, `BitArray`, and continuation agreement | R25, R26 | CPU+CUDA |
| Shape and prefix stability: several dims, all families and result types | R26 | CPU+CUDA |
| Batch continuation equals chained scalar continuation with exact mixed-width counter advances | R24, R26, R53 | CPU+CUDA |
| Fixed-work audit: every random path has input-determined raw-word use and no random retry | R61 | CPU+CUDA |
| Default `rand_next(rng, dims...)` and `randn_next(rng, dims...)` return `Float64` and the generator first | R23, R24 | CPU |
| Last valid reservation returns exhausted; later draw throws; zero-size draw succeeds; destination remains unchanged after failed fixed reservation | R53, R54 | CPU+CUDA |
| `randat`/`randnat` equal indexed fills | R29 | CPU+CUDA |
| Unit and stepped integer ranges: scalar, arrays, continuation, capacity, and no materialization | R55 | CPU+CUDA |
| Interleaved bits, ranges, and normals use one position and separate family regions | R8, R51, R53 | CPU |
| Derivation ignores parent position, resets child position, preserves device, and repeats stable purposes | R17-R21, R38 | CPU+CUDA |
| Unweighted sampling: four population shapes, all `k` forms, fixed-work index reduction, integer ranges, O(k) path, prefix stability | R56-R58, R60 | CPU+CUDA |
| Weighted sampling: raw weights, one sorted threshold batch, zero weights, restored order, chained-scalar equality | R56, R57, R59, R60 | CPU+CUDA |
| Sampling validation and counter exhaustion produce no partial result | R60 | CPU+CUDA |
| Purity: repeated calls identical; pure draws and derivation leave parent unchanged | R5, R20, R56 | CPU |
| Distributions.jl smoke on `StatefulRNG`: `rand(m, dist)`, `rand(m, dist, n)` | R34 | CPU |
| `StatefulRNG` matches continuation draws for primitive, `Bool`, normal, and range calls; hooks equal the pinned method set | R32-R34, R52 | CPU |
| `copy` replay | R35 | CPU |
| CPU/CUDA bitwise equality, integer and uniform draws | R43 | CPU+CUDA |
| GPU kernel compiles addressed, derivation, scalar pure, and scalar continuation draws with zero allocation | R30 | CUDA |
| Launch-shape and lane independence for addressed draws | R51 | CUDA |
| Method-surface audit: typed pure draws, continuation defaults, no default generator, dynamic and `Val` splits, return order, bridge hooks, and no extra foreign methods | R1, R14, R21-R24, R34, R49, R52 | CPU |
| Wrong-device destination, population, and weights throw before generation; device-agnostic ranges work | R39, R57, R59 | CUDA |
| Empty fill launches no kernel | R40 | CPU+CUDA |
| Metal exclusion errors, including zero-size requests | R41, R54 | Metal |
| Serialization round-trip preserves key, exact position, and exhaustion; device resets; malformed payloads reject | R44-R46 | CPU |
| Reactant: changed keys and positions use one compilation | R42 | Reactant |
| Export list equals [R48] exactly | R48 | CPU |
| Dependency and extension audit equals R36-R37 | R36, R37 | CPU |
| Reserved-tag audit: every assigned tag and family word equals R9-R10 | R9, R10 | CPU |
| Error audit: deterministic public throws equal section 11 exactly | R47 | CPU |
| Statistical suite: SmallCrush minimum, bits and uniform, per family, sequential and split-interleaved | R50 | CPU |

## 14. References

- `PureRNGsTestbed.jl`, local checkout at
  `~/Code/scratch/PureRNGsTestbed.jl` — value oracle ([R2])
- [Random123](https://github.com/DEShawResearch/random123) — cores, KATs
- [JAX PRNG design](https://docs.jax.dev/en/latest/jep/263-prng.html) — model precedent, informative
- [MLDataDevices](https://lux.csail.mit.edu/stable/api/Accelerator_Support/MLDataDevices)
- [KernelAbstractions](https://juliagpu.github.io/KernelAbstractions.jl/stable/)
