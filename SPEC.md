# PureRNGs version 0 specification

Status: normative specification, revision 16
Date: 2026-08-24

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
  12 lists every exported symbol. Section 11 lists every public error. The
  sole destination-fill keyword is `threaded::Bool = true`, accepted only by
  the four `rand!`, `rand_next!`, `randn!`, and `randn_next!` forms in
  section 5.
- [R2] Reference scope. `PureRNGsTestbed.jl` (local checkout,
  `~/Code/scratch/PureRNGsTestbed.jl`) at commit
  `7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d` remains authoritative only
  for the unchanged `Philox4x32` and `Threefry2x32` core, counter-layout,
  family-word, and key-derivation constructs shared with this document.
  The governing testbed files are `src/philox.jl`, `src/threefry.jl`,
  `src/derive.jl`, and `src/families.jl`. The testbed's word-aligned draw,
  conversion, cursor, fill, range, sampling, and continuation values do not
  define the packed stream. In-repo vectors record this hash for every
  unchanged construct they capture.
- [R3] Escalation. If a requirement seems to contradict another
  requirement or the testbed within [R2]'s scope, stop and report the
  conflict. Do not resolve it by judgment.

## 2. The model

A generator is an immutable `isbits` value holding a key, one logical-bit
counter position, and a device binding. The counter position is functional
state: pure draws read it, continuation draws return a new generator with an
advanced position.

- [R4] `AbstractPureRNG` is the sole immutable-generator supertype.
  The eight family types directly subtype it; no intermediate Philox or
  Threefry abstract type exists. Every family type is an immutable `isbits`
  struct parameterized by device. Its contents are the native key words,
  the [R53] counter position, and the [R38] package-owned `isbits`
  backend token. No foreign device object is ever stored, so the struct
  is `isbits` for every backend. Construction binds the CPU token. The
  binding never influences stream values.
- [R5] Draws are pure: the same call on the same generator returns the same
  values on every call and every backend, subject only to [R43]. A pure draw
  reads the held position and leaves the generator unchanged.
- [R6] Counter continuation is the default source of fresh sequential
  randomness. `rand_next`, `randn_next`, and `randsample_next` return the
  advanced generator. `splitrng` and `subrng` explicitly derive new keys.
- [R61] Fixed-work law. Every version 0 random-generation call consumes a
  logical-bit count determined only by its method, result types, shapes, range
  spans, population size, weight presence, and `k`. Random values never
  control the bit count or trigger a retry. The implementation contains no
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
  All families read the same held logical-bit position.
- [R9] Tag values, subtag values, family words, and region layout equal the
  testbed values: `DERIVE_TAG = 0xC0FFEE00`, `SPLIT_SUBTAG = 0x00000000`,
  `FOLD_SUBTAG = 0x00000001`, `THREEFRY_FOLD_INDEX = 0xffffffff` (the
  narrow-layout fold namespace), family words as in `src/families.jl`.
  These values are frozen for the life of stream-law version 4.
- [R10] Family words and derivation-region indices not assigned by this
  document are reserved by the stream law and MUST stay unassigned in
  version 0, so later stream-law-compatible extensions can claim them.

- [R51] The logical address of a drawn bit is (key, family, logical-bit
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
  cores. Together, [R9], [R12]-[R12b], and [R62] pin every bit at every
  draw-family address. Every core
  reproduces the release's `kat_vectors` known-answer
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
- [R13] Value definition. For all eight families, [R11] fixes core output,
  [R62] fixes normative bit order, [R27] fixes extraction, and the other
  value-determining rules listed in section 10 fix every public result.
  Normal floating results remain subject to the sole [R43] exception. The
  pinned testbed has only the narrow [R2] authority. Golden vectors for every
  family are frozen in-repo before release and confirm this specification;
  no golden vector defines the stream.
- [R14] Constructors, for every family `F` with key type `K`: `F(seed::
  Integer)` and `F(key::K)`. Every generator names its family at
  construction; the package defines no default generator. Both constructors
  create a CPU-bound generator at logical-bit position zero.
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
  for all eight families. The six families outside [R2]'s derivation scope
  carry in-repo golden vectors ([R13]).
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
Random.rand!(rng, dest::AbstractArray{T};
             threaded::Bool=true)             :: typeof(dest)
Random.rand(rng, range::AbstractRange{T})     :: T
Random.rand(rng, range::AbstractRange{T},
          dim1::Integer, dims::Integer...)     :: Array{T}
Random.randn(rng, ::Type{T})                  :: T   # T float
Random.randn(rng, ::Type{T}, dim1::Integer,
           dims::Integer...)                  :: Array{T}
Random.randn!(rng, dest::AbstractArray{T};
              threaded::Bool=true)            :: typeof(dest)
rand_next(rng::R)                             :: Tuple{R,Float64}
rand_next(rng::R, dim1::Integer,
          dims::Integer...)                   :: Tuple{R,Array{Float64}}
rand_next(rng::R, ::Type{T})                  :: Tuple{R,T}
rand_next(rng::R, ::Type{T}, dim1::Integer,
          dims::Integer...)                   :: Tuple{R,Array{T}}
rand_next(rng::R, range::AbstractRange{T})    :: Tuple{R,T}
rand_next(rng::R, range::AbstractRange{T},
          dim1::Integer, dims::Integer...)     :: Tuple{R,Array{T}}
rand_next!(rng::R, dest::AbstractArray{T};
           threaded::Bool=true)                :: Tuple{R,typeof(dest)}
randn_next(rng::R)                            :: Tuple{R,Float64}
randn_next(rng::R, dim1::Integer,
           dims::Integer...)                  :: Tuple{R,Array{Float64}}
randn_next(rng::R, ::Type{T})                 :: Tuple{R,T}
randn_next(rng::R, ::Type{T}, dim1::Integer,
           dims::Integer...)                  :: Tuple{R,Array{T}}
randn_next!(rng::R, dest::AbstractArray{T};
            threaded::Bool=true)               :: Tuple{R,typeof(dest)}
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
- [R25] Primitive widths and conversion. `Bool` consumes one bit and is true
  exactly when that bit is one. `UInt32` consumes 32 bits and `UInt64`
  consumes 64 bits; their values are the unsigned integers represented by
  those bits under [R27]. Uniform `Float32` consumes 24 bits and returns
  `Float32(k) * Float32(0x1p-24)`. Uniform `Float64` consumes 53 bits and
  returns `Float64(k) * 0x1p-53`. Here `k` is the extracted unsigned integer.
  Uniform floats therefore lie in [0, 1). The `Bool` rule governs scalar
  values, `Array{Bool}`, and `BitArray` fills. `BitArray` is host
  storage, so a `BitArray` fill requires a CPU-bound generator; on a
  device-bound generator the [R39] destination-device check throws.
- [R26] Shape stability: for every generator, result type, and `m <= n`,
  `rand(rng, T, n)[1:m] == rand(rng, T, m)` holds bitwise, and
  `vec(rand(rng, T, a, b)) == rand(rng, T, a*b)`. Same for `randn` and
  integer ranges. A batch continuation equals chained scalar continuation
  draws and advances by the same logical-bit count. Elements consume
  consecutive widths with no padding. A mixed continuation sequence starts
  each draw at the exact bit after its predecessor, independent of type; it
  equals the same calls performed through `StatefulRNG`.
  Pure array draws have the same values but leave the input generator unchanged.
  `threaded = true` and `threaded = false` produce identical values and
  continuation positions.
- [R27] Bit extraction. A width-`w` draw takes exactly the next `w` bits of
  its [R62] family stream. The first consumed bit is bit `w - 1` of the
  extracted unsigned integer and the last is bit zero. Extraction crosses
  native-word and core-block boundaries when needed. It discards no bit,
  repeats no bit, and adds no alignment or padding. The next draw starts at
  the first unconsumed bit.
- [R62] Logical bit order. Each (key, draw-family word) pair has one bit
  stream. Core blocks appear in increasing block-index order. Within a block,
  native output words appear in tuple order. Within each native word, bits appear
  from most significant to least significant. Concatenating those bits with
  no gap defines the stream. All result types assigned to one draw family
  share this bit stream and the generator's one held logical-bit position.
- [R28] Normal generation uses `FAMILY_NORMAL`, no cache, and no rejection.
  A normal `Float32` consumes 23 bits; a normal `Float64` consumes 52 bits.
  Let `k32::UInt32` and `k64::UInt64` be the respective extracted integers.
  The executable midpoint conversions are
  `u32 = Float32((k32 << UInt32(1)) | UInt32(1)) * Float32(0x1p-24)`
  and
  `u64 = Float64((k64 << UInt64(1)) | UInt64(1)) * Float64(0x1p-53)`.
  Their endpoint grids are exactly `Float32(0x1p-24)` through
  `one(Float32) - Float32(0x1p-24)` and `Float64(0x1p-53)` through
  `one(Float64) - Float64(0x1p-53)`. The following AS241 definition is
  normative.
  `H(x, (c1, ..., cn))` evaluates `p = c1; p = fma(p, x, ci)` for
  `i = 2:n`. Every floating arithmetic operation and floating literal has
  result type `T`; comparisons return `Bool`.

  ```text
  q = u - T(0.5)
  if abs(q) <= T(0.425)
      r = T(0.180625) - q*q
      z = q * (H(r, A) / H(r, B))
  else
      r = sqrt(-log(q < zero(T) ? u : one(T)-u))
      if r <= T(5)
          r = r - T(1.6); z = H(r, C) / H(r, D)
      else
          r = r - T(5);   z = H(r, E) / H(r, F)
      end
      z = q < zero(T) ? -z : z
  end
  ```

  For `Float32`, the coefficient tuples in Horner order are:

  ```text
  A = (5.9109374720f1, 1.5929113202f2, 5.0434271938f1, 3.3871327179f0)
  B = (6.7187563600f1, 7.8757757664f1, 1.7895169469f1, 1.0f0)
  C = (1.7023821103f-1, 1.3067284816f0, 2.7568153900f0, 1.4234372777f0)
  D = (1.2021132975f-1, 7.3700164250f-1, 1.0f0)
  E = (1.7337203997f-2, 4.2868294337f-1, 3.0812263860f0, 6.6579051150f0)
  F = (1.2258202635f-2, 2.4197894225f-1, 1.0f0)
  ```

  For `Float64`, the coefficient tuples in Horner order are:

  ```text
  A = (2.5090809287301226727e3, 3.3430575583588128105e4,
       6.7265770927008700853e4, 4.5921953931549871457e4,
       1.3731693765509461125e4, 1.9715909503065514427e3,
       1.3314166789178437745e2, 3.3871328727963666080)
  B = (5.2264952788528545610e3, 2.8729085735721942674e4,
       3.9307895800092710610e4, 2.1213794301586595867e4,
       5.3941960214247511077e3, 6.8718700749205790830e2,
       4.2313330701600911252e1, 1.0)
  C = (7.74545014278341407640e-4, 2.27238449892691845833e-2,
       2.41780725177450611770e-1, 1.27045825245236838258,
       3.64784832476320460504, 5.76949722146069140550,
       4.63033784615654529590, 1.42343711074968357734)
  D = (1.05075007164441684324e-9, 5.47593808499534494600e-4,
       1.51986665636164571966e-2, 1.48103976427480074590e-1,
       6.89767334985100004550e-1, 1.67638483018380384940,
       2.05319162663775882187, 1.0)
  E = (2.01033439929228813265e-7, 2.71155556874348757815e-5,
       1.24266094738807843860e-3, 2.65321895265761230930e-2,
       2.96560571828504891230e-1, 1.78482653991729133580,
       5.46378491116411436990, 6.65790464350110377720)
  F = (2.04426310338993978564e-15, 1.42151175831644588870e-7,
       1.84631831751005468180e-5, 7.86869131145613259100e-4,
       1.48753612908506148525e-2, 1.36929880922735805310e-1,
       5.99832206555887937690e-1, 1.0)
  ```
- [R29] `randat(rng, T, i) == rand(rng, T, n)[i]` for every `n >= i`, and
  `randnat` likewise for `randn`. The index is one-based: element `i`
  starts at the generator's held position plus `i - 1` times that element's
  bit width.
  These addressed operations never advance the generator. They throw
  `ArgumentError` for `i < 1` or when the addressed
  bit span exceeds [R53].
- [R30] `randat`, `randnat`, `subrng`, `splitrng` with `Val`, and scalar
  pure and continuation draws compile in GPU kernels without allocation,
  dynamic dispatch, or host state, and run on the host with bitwise
  identical results for integer and uniform types. All wide integer
  arithmetic in device-reachable bit extraction, position, addressing,
  reservation, capacity, and range-candidate paths uses `UInt64` limbs. A
  quantity wider than one limb uses `(lo::UInt64, hi::UInt64)`, low limb
  first. Native core words, [R53] bit offsets, and final `UInt32` values
  retain their specified narrower types. No device-reachable typed IR contains
  a `BigInt` or `UInt128` value, instruction, or call.
- [R53] The counter position is the zero-based index of the next logical
  bit. A 64-bit block-index position is exactly
  `(block::UInt64, bit::UInt16)`. A 128-bit block-index position is exactly
  `(lo::UInt64, hi::UInt64, bit::UInt16)`, low block limb first. Each form
  has one terminal exhausted value. For a core block with `N` native output
  words of `W` bits, `B = N * W`. Valid bit offsets are `UInt16(0)` through
  `UInt16(B - 1)`. Terminal is represented
  by the maximum draw-block index and the one reserved offset
  `typemax(UInt16)`. Offsets `UInt16(B)` through
  `typemax(UInt16) - UInt16(1)` are invalid. The widths are
  [R25] for primitive draws, [R28] for normals, [R55] for integer ranges,
  [R58] for unweighted sampling, and [R59] for weighted thresholds.
  A nonempty reservation consumes exactly the sum of its result widths,
  with no alignment or padding. It may cross native-word and core-block
  boundaries. A zero-element request reserves nothing and changes no
  position, including at terminal.
  Each draw family applies its own family word to the same position.
  `rand` and `randn` read at the held position without advancing.
  Continuation draws reserve their exact bit span and return the advanced
  generator.
  The last valid reservation returns the exhausted value. A
  later nonempty draw throws `ArgumentError`. A failed reservation leaves
  a nonterminal position unchanged even when fewer bits remain than the
  requested width. No operation wraps the position or derives a new key
  automatically.
- [R54] A fixed-size draw computes its full [R53] bit reservation with
  checked [R30] `UInt64`-limb arithmetic before generating a value or
  mutating a destination. This binds every method the package owns. A
  foreign `Random` fill method reaching the [R34] bridge hooks is a
  chain of scalar draws, each preflighting only its own span; the [R34]
  partial-write statement is the sole exemption. A fixed-size draw
  throws `ArgumentError` if the reservation
  exceeds the family region. A zero-size draw succeeds at every position,
  including exhaustion,
  provided its result type and family are serviceable on the generator's
  device: the [R41] Metal exclusion is checked before size and throws
  even for a zero-size request. The region contains
  `(max_block + 1) * core_block_bits` bits. `max_block` is `2^56 - 1` for
  `Philox2x32` and `Threefry2x32`; `2^64 - 1` for `Philox4x32`,
  `Philox2x64`, `Threefry4x32`, and `Threefry2x64`; and `2^128 - 1` for
  `Philox4x64` and `Threefry4x64`. Device-reachable preflight never
  materializes the total capacity. It computes counts, block advances, bit
  remainders, and checked carries with the [R30] `UInt64` limb form. It uses
  no `BigInt` or `UInt128` value, instruction, or call.
- [R55] Integer-range draws support nonempty unit and stepped ranges whose
  element type is a signed or unsigned integer of at most 64 bits, excluding
  `Bool`. They use `FAMILY_RANGE`. For a range length `s` through `2^32`,
  extract a 64-bit unsigned integer `u` and select
  `floor(u * s / 2^64)`. For a wider length, including the `s == 2^64`
  full-width case, extract a 128-bit unsigned candidate as
  `(lo::UInt64, hi::UInt64)`, low limb first, and select the mathematical
  value `floor(u * s / 2^128)`, where mathematically
  `u = hi * 2^64 + lo`. Device-reachable candidate extraction and both
  reductions use fixed `UInt64` limb pairs and 32-bit sub-limb products.
  They use no `BigInt` or `UInt128` value, instruction, or call. Stepped
  ranges apply the same reduction to their length and map the selected
  zero-based offset by range indexing without materialization. The maximum
  relative preimage bias is below `2^-32` for the 64-bit path and below
  `2^-64` for the 128-bit path.
  Empty ranges throw `ArgumentError`.
- [R50] Statistical quality: the complete pinned TestU01 SmallCrush matrix
  passes the release rule below. TestU01 version: 1.2.3. Root generator:
  `F(12345)`. Bits stream: the `UInt32` draw sequence, fed to TestU01 as
  32-bit integers. Uniform stream: the uniform `Float64` draw sequence, fed
  as doubles. Sequential run: the root generator's continuation chain.
  Interleaved run: the eight children of `splitrng(root, 8)`, round-robin,
  one value per child per turn in child order.

  The matrix is every family times {bits, uniform} times {sequential,
  interleaved}: 32 cases. SmallCrush 1.2.3 returns exactly 15 p-values per
  case, hence 480 p-values for the complete matrix.

  The inclusive TestU01 summary interval `[0.001, 0.999]` is diagnostic. The
  inclusive release interval is `[α, 1 - α]`, where `α = 0.001 / 480`,
  computed in `Float64`. This per-tail Bonferroni allocation bounds
  matrix-wide two-tail false rejection by `0.002` without assuming
  independence.

  A non-finite p-value, a case returning other than 15 p-values, an incomplete
  case, or any p-value outside the release interval blocks release. A finite
  p-value outside the diagnostic interval but inside the release interval is
  recorded as suspect and does not block release.

  Release architecture: the x86-64 Linux CPU release runner. The committed
  run log records the driver package and version, configuration, every
  p-value, both interval classifications, every case count, and matrix
  completion. Runs at other root seeds are nonnormative diagnostics and do
  not affect release.

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
  population contains. A form with `k` returns exactly `k` samples when
  `0 <= k <= typemax(Int)` and throws `ArgumentError` otherwise.
  `randsample` reads the held position and leaves the generator unchanged.
  `randsample_next` returns `(next_rng, values)`.
- [R57] The population's device is compatible under the sampling equality
  rule in [R38], or when `MLDataDevices.get_device(iter)` returns `nothing`.
  `nothing` means device-agnostic and is compatible with every generator
  device; `AbstractRange` has this status. Any other device result throws
  `ArgumentError` before generation. Integer unit and stepped ranges are
  mapped directly by position without materialization. Every other
  `AbstractArray` on a CPU generator or on the same non-CPU backend as the
  generator is indexed directly by canonical ordinal: population position
  `p` selects `A[I_p]`, where `I_p` is the `p`-th element of
  `CartesianIndices(axes(A))` in its iteration order. This defines values for
  any dimensionality and axis offsets and does not require materializing the
  Cartesian indices. On a non-CPU generator, every other device-agnostic
  population is collected exactly once on CPU — in canonical Cartesian order
  for an array and native iteration order for a non-array — then transferred
  to the generator device exactly once. On CPU, a device-agnostic non-array
  is collected exactly once and a device-agnostic array is indexed directly
  in canonical Cartesian order. A population that is neither an
  `AbstractArray` nor device-agnostic throws `ArgumentError` before
  generation. The result is allocated on the generator device and its
  element type comes from the indexed or materialized population.
  Population cardinality MUST fit a positive `UInt64` when a sample is
  requested. A no-`k` form additionally requires cardinality to fit
  `Int` because it returns that many elements.
- [R58] Unweighted sampling uses the fixed-work [R55] range reduction to
  select a zero-based population index in O(k). Population sizes through
  `2^32` consume 64 bits per sample and use the [R55] 64-bit multiply-high
  reduction. Wider populations consume 128 bits per sample and use the
  [R55] 128-by-64 reduction. The [R55] preimage-bias bound applies. This
  path never constructs weights and never retries.
- [R59] Weighted forms accept only a raw `AbstractVector{<:Real}` aligned
  with the population. Its device must be compatible under the sampling
  equality rule in [R38], or device-agnostic. Any other device result throws
  `ArgumentError` before generation. They do not accept a weight wrapper.
  Weight ordinal `p` is its canonical Cartesian ordinal and aligns with
  population ordinal `p`. Weights are converted to `Float64` in ordinal
  order. On a non-CPU generator, device-agnostic weights are converted and
  elementwise validated exactly once on CPU and transferred as one dense
  `Float64` vector to the generator device exactly once. The total and
  cumulative scan are `Float64` left folds on the generator device. Each
  starts from `+0.0`, and each addition is rounded to `Float64` before the
  next addition. Each threshold consumes 53 bits from `FAMILY_RANGE` and uses
  `u = Float64(j) * 0x1p-53` for the extracted integer `j`. A batch records
  their original indices, maps each uniform `u` to
  `min(u * total, prevfloat(total))`, sorts by
  `(threshold, original_index)`, scans the weights once, and restores draw
  order. Selection chooses the first cumulative total strictly greater than
  the threshold. This algorithm is normative.
- [R60] Sampling validates the complete population, `k`, and weights before
  drawing. Device validation occurs first. It throws `ArgumentError` for
  negative `k`, `k > typemax(Int)`, an empty population with positive `k`, a
  weight-length mismatch, a non-finite or negative weight, or a non-finite or
  non-positive total. Zero weights are valid.
  Empty unweighted sampling with `k == 0` returns an empty vector. Batch
  results equal chained one-sample continuation calls and obey prefix
  stability. Every sampling call has a fixed logical-bit count, preflights its
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
  chained `rand_next` or `randn_next` sequence. They do not split keys.
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
  The package also defines exactly two range `Sampler` methods for the
  [R55] integer element types —
  `Random.Sampler(::Type{<:StatefulRNG}, range::AbstractRange{T},
  ::Random.Repetition)` and its dispatch-disambiguating intersection
  `Random.Sampler(::Type{<:StatefulRNG}, range::AbstractUnitRange{T},
  ::Random.Repetition)`, with identical behavior — and the matching
  `Random.rand(m::StatefulRNG, sampler)` method. Normal draws bypass
  `Sampler`. The owned fill methods are
  `Random.rand!(m::StatefulRNG, A::Array{T})` and
  `Random.randn!(m::StatefulRNG, A::Array{T})` for the section 5 result
  types, plus `Random.rand!(m::StatefulRNG, B::BitArray)`, which
  overrides Base's `UInt64`-chunk `BitArray` specialization and
  preserves the [R25] one-bit-per-`Bool` rule. Concrete destination
  types foreclose ambiguity with foreign `AbstractRNG` fill methods.
  These owned fills preflight their full reservation atomically ([R54])
  and equal chained scalar continuations. Any other destination type
  dispatches through `Random`'s public machinery: uniform, `Bool`, and
  integer fills reach the five hooks, and normal fills loop the owned
  scalar `randn(m, T)`. Every bit such a foreign method consumes is a
  bridge draw obeying the stream law, but the package does not promise
  chained-scalar consumption order for foreign fill methods, and on
  counter exhaustion mid-fill the destination MAY be partially written
  while the held generator stays valid at the position after the last
  successful draw (the sole [R54] exemption). The documentation states
  this. The bridge method set MUST be free of dispatch ambiguities
  against Base and Random. These hooks are exempt from [R1] and MUST
  NOT widen behavior beyond this section.
- [R35] `copy(m)` returns an independent wrapper with the same held
  generator, giving exact replay.

- [R52] The bridge's complete method set is [R34] plus `copy` ([R35]). Pure
  code creates a `StatefulRNG` and hands it derived children; keys stay
  pure-side.

## 8. Device placement

Dependencies: MLDataDevices (device selection) and KernelAbstractions (fill
kernels) are normal dependencies [R36]. CUDA.jl, AMDGPU.jl, and Metal.jl
are weak dependencies with package-owned extensions [R37].

```julia
dev = gpu_device(; force=true)
rng = dev(Philox4x32(1234))
xs  = rand(rng, Float32, 1_000_000)   # device array
```

- [R38] The binding is a package-owned, zero-size, `isbits` backend
  token with exactly four values: CPU, CUDA, AMDGPU, Metal. Applying an
  MLDataDevices `CPUDevice`, `CUDADevice`, `AMDGPUDevice`, or
  `MetalDevice` value — named or unnamed, any type parameters — sets the
  token for that backend. A physical device handle or eltype adaptor
  inside the applied value is discarded: physical placement of
  allocations and launches follows the backend's active device at call
  time, and the documentation directs multi-GPU ordinal selection to the
  backend's own mechanism (for example `CUDA.device!`). Applying any
  other MLDataDevices device type throws `ArgumentError` (section 11).
  Destination equality under [R39] remains backend equality decided by
  `MLDataDevices.get_device_type`. Sampling equality for [R57] populations
  and [R59] weights first calls `MLDataDevices.get_device(input)`. A result
  of `nothing` is device-agnostic. Every returned supported device instance
  is mapped to its unparameterized `CPUDevice`, `CUDADevice`,
  `AMDGPUDevice`, or `MetalDevice` wrapper, then compared with
  `MLDataDevices.get_device_type(rng.device)`. Sampling therefore supports
  inputs that define only `get_device`. The generator is `isbits`, so
  binding moves no data. `splitrng`, `subrng`, and every continuation
  preserve the binding. Applying another supported device preserves the key
  and position and changes only the token.
- [R39] Placement invariant: device-closed for data, host-transparent for
  logical state. Section 5 allocating draws on a device-bound generator
  return arrays allocated on that device and generate every value there.
  `rand!`/`randn!` require the destination on the generator's device and
  throw `ArgumentError` on mismatch. Scalar draws, `randat`, `randnat`,
  `splitrng`, `subrng`, and scalar continuation draws compute where they are
  called — kernel registers or host arithmetic — and these operations perform
  no device-to-host copy or synchronization. Section 6 sampling allocates its
  result on the generator device. The sole sampling host-staging exceptions
  are the device-agnostic populations in [R57] and device-agnostic weights in
  [R59]. Sampling may copy one validation-status `Bool` from the device to
  the host. Thresholds, sort and order indices, selections, and results
  remain on the generator device. The `threaded` fill keyword controls CPU
  task use only. On non-CPU backends both values preserve the ordinary
  backend launch path.
- [R40] The typed `threaded` keyword is validated before fill preflight.
  After it passes, fills validate in fixed order: destination device ([R39]),
  [R41] serviceability, then size. A fill that passes validation with an
  empty destination performs no backend lookup and no kernel launch and
  leaves the position unchanged. On CPU, `threaded = false` performs the
  full preflight and reservation, then fills serially on the calling task with
  no KernelAbstractions backend lookup or task launch; it returns after the
  fill completes. Every other nonempty fill uses the KernelAbstractions
  backend. Completion follows that backend's semantics: the CPU call returns
  after its synchronous work, while an asynchronous backend may return after
  enqueueing work with its queue ordering.
- [R41] Backend tiers. CPU and CUDA pass the full section 13 suite and
  block release. AMDGPU is a preview: the suite runs and failures are
  documented without blocking. Metal is experimental. On a Metal-bound
  generator the device-executing operations — allocating draws,
  destination fills, and every `randsample` form — are served only as
  primitive `Bool`, `UInt32`, `UInt64`, and `Float32` draws and fills,
  uniform and normal, on 32-bit families. Every other device-executing
  request throws `ArgumentError` (section 11): `Float64` results,
  64-bit-word families, every allocating integer-range draw, and every
  `randsample` form. Metal has no `Float64`, the 64-bit Philox
  multiplies need 128-bit emulation Metal.jl lacks, the 64-bit Threefry
  integer paths and the [R55] range reductions are unvalidated on
  Metal.jl, and [R59] weighted sampling requires `Float64` — all
  excluded pending validation. Host-computing operations are never
  excluded by binding: scalar draws, scalar range draws, scalar
  continuations, `randat`, `randnat`, `splitrng`, and `subrng` follow
  [R39] and run as host arithmetic on a Metal-bound generator for every
  result type and family. The exclusion applies before size checks: a
  zero-size excluded device-executing request throws the same
  `ArgumentError`. The `ArgumentError` contract binds host-called API.
  A scalar draw compiled inside a user device kernel cannot throw a
  host error; an excluded type there fails through the backend compiler
  or `Metal.KernelException`, which [R47] classifies as a backend
  failure, not a contract error.
- [R42] Reactant: a function taking a generator argument compiles under
  `Reactant.@compile`, and the compiled draws, continuations, and
  derivations equal their eager values bitwise, subject only to [R43].
  This gates release. Version 0 does not require keys or positions to
  trace as data: Reactant MAY close over them as compile-time constants,
  so a changed key or position MAY trigger recompilation. The
  documentation states this limitation and recommends hoisting
  compilation outside key- or position-varying loops.

## 9. Cross-backend guarantees

- [R43] Integer and uniform results are bitwise identical across all
  supported backends for the same family, key, family region, and logical-bit
  positions, with no tolerance. Normal input bits and midpoint values obey
  the same guarantee. Final normal values are bitwise reproducible within one
  backend and architecture. Across backends or architectures, AS241 floating
  evaluation MAY differ through `log` and `sqrt`. The
  documentation states this sole cross-backend or cross-architecture
  exception.

## 10. Stream law

The closed stream law is the set of value-determining rules: [R9] tags and
regions, [R11] cores and rounds, [R12]-[R12b] layouts and narrow
derivation, [R14] and [R20] initial position zero, [R16] seed mapping,
[R17]-[R19] derivation, [R25] primitive widths and conversion, [R27] bit
extraction, [R28] normal generation, [R53] bit continuation and capacity,
[R62] logical bit order, [R55] integer ranges, [R57]-[R59] sampling, and
the fixed-work rule [R61]. Rules that equate one public operation with a
composition of stream-law rules — [R26] batch and mixed-type sequencing,
[R29] addressed indexing, [R33] the bridge, and [R60] sampling order and
prefix stability — are consistency laws. They introduce no value of their
own. A change to one that changes any value changes a listed stream-law rule.

- [R44] This closed set is stream-law version 4. Any change to a
  value-determining rule requires a new stream-law version.

## 11. Errors, closed list

The complete set of public-API throws:

| Call | Condition | Error |
| --- | --- | --- |
| `F(seed)`, `Random.seed!(m, seed)` | `seed < 0` or `seed >= 2^key_bits` | `ArgumentError` |
| `splitrng(rng, n)` or `splitrng(rng, Val(N))` | `n < 0`, or `N` is not a non-negative `Int` | `ArgumentError` |
| `splitrng` on a narrow family | child index at or beyond `2^32 - 1` (the reserved fold namespace, [R12b]) | `ArgumentError` |
| `randat`/`randnat` | `i < 1` or addressed bit span exceeds the remaining family region ([R29]) | `ArgumentError` |
| any nonempty pure, continuation, or destination draw | required bit span exceeds the remaining family region, including an exhausted generator ([R53], [R54]) | `ArgumentError` |
| integer-range draw | range is empty | `ArgumentError` |
| `rand`/`randn` or their continuation forms with dims | any negative dim | `ArgumentError` (Base array semantics) |
| `rand(rng)` or `randn(rng)` untyped on an immutable generator | always | `ArgumentError` naming the typed form ([R23]) |
| `randsample`/`randsample_next` | `k < 0` or `k > typemax(Int)` | `ArgumentError` |
| `randsample`/`randsample_next` | empty population and `k > 0` | `ArgumentError` |
| `randsample`/`randsample_next` | positive population cardinality exceeds `typemax(UInt64)`, or a no-`k` result length exceeds `typemax(Int)` | `ArgumentError` |
| weighted `randsample`/`randsample_next` | weight-length mismatch, non-finite or negative weight, or non-finite or non-positive total | `ArgumentError` |
| weighted `randsample`/`randsample_next` | weights are not a raw `AbstractVector{<:Real}` | `MethodError` (no method) |
| `randsample`/`randsample_next` | population has a non-agnostic device differing from the generator device, or is neither an `AbstractArray` nor device-agnostic | `ArgumentError` |
| weighted `randsample`/`randsample_next` | weights have a non-agnostic device differing from the generator device | `ArgumentError` |
| `rand!`/`randn!` and continuation forms | destination device differs from generator device | `ArgumentError` |
| `rand!`/`randn!` and continuation forms | `threaded` is not a `Bool` | `TypeError` |
| primitive draw or fill | result type or destination eltype is not a result type | `MethodError` (no method) |
| device-executing operation on Metal: allocating draw, destination fill, allocating range draw, or sampling | excluded by the [R41] Metal served set, any size including zero | `ArgumentError`, names Metal |
| MLDataDevices device application to a generator | device type outside `CPUDevice`, `CUDADevice`, `AMDGPUDevice`, `MetalDevice` | `ArgumentError` |
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
  `randnat`, `StatefulRNG`. The `threaded` keyword adds no exported name.
- [R49] Non-exported public surface: the `Base` and `Random` draw methods
  of section 5, including the exact `threaded::Bool = true` destination-fill
  keyword surface,
  the `Random` methods of [R34], `copy(::StatefulRNG)`, and MLDataDevices
  device application. Every method extension of a foreign function has a
  package-owned type in a dispatch position.

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
| Pinned testbed agreement for unchanged cores, layouts, family words, `splitrng`, and `subrng` on `Philox4x32` and `Threefry2x32` | R2, R9, R11, R12a, R12b, R17, R18 | CPU |
| Packed-stream golden vectors for every result class and all eight families confirm MSB-first extraction, canonical sampling ordinals, and exact weighted `Float64` folds without defining them, subject only to the normal exception | R13, R25, R27, R28, R43, R53, R55, R57-R59, R62 | CPU |
| Derivation golden vectors for the six families outside the pinned testbed scope | R13, R19 | CPU |
| Bit extraction starts MSB-first at varied `UInt16` offsets and crosses native-word and core-block boundaries without gaps | R27, R53, R62 | CPU+CUDA |
| Width audit: primitive 1/24/32/53/64, normal 23/52, range and unweighted 64/128, weighted 53 | R25, R28, R55, R58, R59 | CPU |
| AS241 audit: coefficient tuples, exact typed midpoint endpoints, and central, moderate-tail, and extreme-tail branches | R28, R43 | CPU+CUDA |
| Seed mapping values and bounds sweep | R15, R16 | CPU |
| Construction gives CPU binding and zero position; flat concrete `isbits` hierarchy | R4, R14 | CPU |
| `Bool` scalar, array, and continuation agreement | R25, R26 | CPU+CUDA |
| `BitArray` fill agreement and device-bound rejection | R25, R39 | CPU |
| Shape and prefix stability: several dims, all families and result types | R26 | CPU+CUDA |
| Batch continuation equals chained scalar continuation with exact mixed-type bit advances and no gaps | R24, R26, R53, R62 | CPU+CUDA |
| Ordinary and serial CPU fills agree for every family and result type; a write-task probe confirms serial fills run on the caller task; serial fills preserve preflight, avoid backend lookup and tasks, infer, and allocate zero where viable | R1, R26, R39, R40, R49 | CPU |
| Fixed-work audit: every random path has input-determined bit use and no random retry | R61 | CPU+CUDA |
| Default `rand_next(rng, dims...)` and `randn_next(rng, dims...)` return `Float64` and the generator first | R23, R24 | CPU |
| Exact 64-bit and low-limb-first 128-bit position representations, 2^62/2^71/2^136-bit capacity classes, reserved `typemax(UInt16)` terminal, invalid-offset rejection internally, cross-block draws, last valid reservation, later failure, zero-size success, and unchanged state and destination after failed preflight | R53, R54 | CPU+CUDA |
| `randat`/`randnat` equal indexed fills | R29 | CPU+CUDA |
| Unit and stepped integer ranges: scalar, arrays, continuation, 64/128-bit work, capacity, and no materialization | R55 | CPU+CUDA |
| Interleaved primitives, ranges, and normals use one bit position and separate family regions | R8, R26, R51, R53 | CPU |
| Derivation ignores parent position, resets child position, preserves device, and repeats stable purposes | R17-R21, R38 | CPU+CUDA |
| Unweighted sampling: canonical Cartesian array ordinals, native iterable order, direct integer ranges, all `k` forms, 64/128-bit fixed-work reduction, O(k) path, and prefix stability | R56-R58, R60 | CPU+CUDA |
| Weighted sampling: canonical population-weight alignment, ordinal `Float64` conversion, exact left-fold totals and scans, 53-bit thresholds, one sorted batch, zero weights, restored order, and chained-scalar equality | R56, R57, R59, R60 | CPU+CUDA |
| Sampling validation, including both `k` bounds, and counter exhaustion produce no partial result | R56, R60 | CPU+CUDA |
| Purity: repeated calls identical; pure draws and derivation leave parent unchanged | R5, R20, R56 | CPU |
| Distributions.jl smoke on `StatefulRNG`: `rand(m, dist)`, `rand(m, dist, n)` | R34 | CPU |
| `StatefulRNG` matches continuation draws for primitive, `Bool`, normal, and range calls; hooks equal the pinned method set | R32-R34, R52 | CPU |
| `copy` replay | R35 | CPU |
| CPU/CUDA bitwise equality, integer and uniform draws at matching bit positions | R43 | CPU+CUDA |
| GPU kernel compiles addressed, derivation, scalar pure, and scalar continuation draws with zero allocation | R30 | CUDA |
| Typed-IR audit: all wide integer arithmetic in device-reachable extraction, position, address, reservation, capacity, and range-candidate paths uses low-limb-first `UInt64` pairs and contains no `BigInt` or `UInt128` value, instruction, or call | R30, R53-R55 | CPU+CUDA |
| Launch-shape and lane independence for addressed draws | R51 | CUDA |
| Method-surface audit: typed pure draws, continuation defaults, exact destination-fill keyword, no default generator, dynamic and `Val` splits, return order, bridge hooks, and no extra foreign methods | R1, R14, R21-R24, R34, R49, R52 | CPU |
| Wrong-device inputs throw before generation; agnostic population and weight staging occurs exactly once as specified; only one validation `Bool` returns to the host; thresholds, ordering, selections, and results remain device-resident | R38, R39, R57, R59 | CUDA |
| Validated empty fill launches no kernel and keeps the position; validation order is `threaded` type, device, serviceability, size | R40 | CPU+CUDA |
| Metal exclusion errors, including zero-size requests | R41, R54 | Metal |
| Reactant-compiled draws, continuations, and derivations equal eager values | R42 | Reactant |
| Export list equals [R48] exactly | R48 | CPU |
| Dependency and extension audit equals R36-R37 | R36, R37 | CPU |
| Reserved-tag audit: every assigned tag and family word equals R9-R10 | R9, R10 | CPU |
| Stream-law closed-list audit identifies version 4 and every value-determining rule | R44 | CPU |
| Error audit: deterministic public throws equal section 11 exactly | R47 | CPU |
| Statistical suite: complete 32-case, 480-p-value SmallCrush matrix under the [R50] diagnostic and Bonferroni release intervals, with the committed complete run log | R50 | CPU |

## 14. References

- `PureRNGsTestbed.jl`, local checkout at
  `~/Code/scratch/PureRNGsTestbed.jl` — narrow reference scope ([R2])
- [Random123](https://github.com/DEShawResearch/random123) — cores, KATs
- [JAX PRNG design](https://docs.jax.dev/en/latest/jep/263-prng.html) — model precedent, informative
- [MLDataDevices](https://lux.csail.mit.edu/stable/api/Accelerator_Support/MLDataDevices)
- [KernelAbstractions](https://juliagpu.github.io/KernelAbstractions.jl/stable/)
