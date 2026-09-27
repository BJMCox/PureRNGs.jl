# Engine interface

This page is for maintainers of a package that brings its own random number
engine and wants PureRNGs' distribution, normal, and exponential draws on it.
Every name below is internal. The page is not part of the user manual.

## Scope

The interface serves:

- `randn_next`, `randn_next!`, `randn_at` and the `randexp_*` counterparts.
- `rand_next`, `rand_next!`, and `rand_at` with the fixed-map distributions,
  DiscreteUniform, the Gamma family (Gamma, Chisq, InverseGamma, Beta, TDist),
  Dirichlet, MvNormal, and their ForwardDiff dual forms.

It does not serve uniform draws, ranges, collections, sampling, permutations,
Categorical, StaticArrays, StatsBase, or Reactant. An engine keeps its own
uniform draws.

## Division of work

PureRNGs owns the mathematics: each distribution maps to a *codec*, and a codec
turns raw stream bits into a value. The engine owns its stream: where bits come
from, how positions align and advance, when the stream is exhausted, and how
bulk fills generate bits efficiently.

A codec reads its bits as `count` *takes* of `W` bits each, W in 1:64. Every
draw of a codec has the same take signature, whatever values it produces:

```julia
count, width = PureRNGs._codec_takes(codec, T)   # width isa Val{W}
value, cursor = PureRNGs._codec_take(codec, rng, cursor, T)
```

`_codec_take` reads exactly the draw's `count` takes from `cursor` and returns
the cursor past them, including takes a rejection sampler leaves unread. Codecs
are isbits values, so they run inside device kernels.

## Hooks an engine defines

`rng` is the engine's generator value and `cursor` is any type the engine
chooses. Methods dispatch on the engine's own types.

| Hook | Returns | Contract |
|:--|:--|:--|
| `_engine_backend(rng)` | a `PureRNGs._BackendToken` | Selects the normal and exponential transforms, allocation, residence checks, and serviceability checks. |
| `_take_bits(rng, cursor, Val(W))` | `(raw::UInt64, cursor)` | The engine's next W-bit field, in the low bits of `raw`. |
| `_skip_takes(rng, cursor, count, Val(W))` | `cursor` | The cursor `count` takes later, without reading them. |
| `_cursor_ordinal(rng, cursor)` | `UInt64` | The stream index of the cursor's next take. It keys the Gamma fallback's child stream. |
| `_draw_cursor(rng, count, Val(W))` | `(cursor, next_rng)` | Reserves `count` takes at the held position: the cursor of the first take and the generator past the last. The engine applies its alignment and exhaustion rules here. |
| `_addressed_state(rng, count, Val(W), i)` | generator | The generator at the start of draw `i`, counting from one, of `count`-take draws. `rng` does not change. Throws `ArgumentError` for `i < 1` and the engine's exhaustion error past the stream end. |
| `_fill_cursor(rng, count, Val(W), ordinal::UInt64)` | `cursor` | The cursor of zero-based draw `ordinal` of a fill that starts at the held position. The fill has reserved its span, so this hook does not check. |

Optional overrides, with defaults built from the hooks above:

| Hook | Default |
|:--|:--|
| `_engine_draw_next(rng, codec, T)` | `_draw_cursor`, then one `_codec_take`. |
| `_engine_draw_at(rng, codec, T, i)` | `_addressed_state`, `_draw_cursor`, then one `_codec_take`. |
| `_engine_fill!(rng, destination, threaded, codec)` | A host loop over one cursor, or over one `_fill_cursor` per chunk when `threaded`. It throws `ArgumentError` for a device-bound generator. Returns `(destination, next_rng)`. |

Override `_engine_fill!` to keep bulk generation fast. The default walks a cursor
take by take, so an engine whose stream is cheapest in whole blocks or rows
should generate those blocks and run `_codec_take` over a cursor into them. A
device fill needs this override: a kernel can call `_codec_take` on the
engine's own cursor type.

## Public methods an engine defines

The Gamma fallback draws a child stream through public methods on the engine's
type:

- `subrng(rng, purpose)`
- `randn_next(rng, F)` for the draw's float type `F`
- `rand_next(rng, UInt64)`

The engine's public entry points forward to the bodies, which take the public
arguments unchanged:

| Public function | Body |
|:--|:--|
| `randn_next`, `randn_next!`, `randn_at` | `_engine_randn_next`, `_engine_randn_next!`, `_engine_randn_at` |
| `randexp_next`, `randexp_next!`, `randexp_at` | `_engine_randexp_next`, `_engine_randexp_next!`, `_engine_randexp_at` |
| `rand_next`, `rand_next!`, `rand_at` with a distribution | `_engine_rand_next`, `_engine_rand_next!`, `_engine_rand_at` |

For example:

```julia
PureRNGs.randn_next(rng::MyEngine, args...; kwargs...) =
    PureRNGs._engine_randn_next(rng, args...; kwargs...)
PureRNGs.rand_next(rng::MyEngine, d; kwargs...) = PureRNGs._engine_rand_next(rng, d; kwargs...)
```

Keep the engine's own `rand_next(rng, ::Type{T})` methods. They are more specific
than an untyped forwarder.

## Stream obligations

- A draw's span is fixed by its take signature. Rejection never changes how far
  the generator advances.
- A fill equals a loop of scalar draws, and addressed draw `i` equals element
  `i` of the fill from the same generator.
- `_cursor_ordinal` must not depend on how the engine represents positions.
  Two cursors at the same stream position give the same ordinal.
- Draws whose destination sits on another device throw before any work. Checks
  run on the backend token, so an engine bound to Metal meets Metal's limits.

## Compatibility policy

PureRNGs is at version 0.0.x, and any release may change these hooks. An engine
package should:

- pin an exact PureRNGs version in its compat,
- run a conformance test against that version in its PureRNGs test environment,
- change the pin only after the conformance test passes.

`test/engine_fixture.jl` is a worked example. It wraps a built-in generator,
serves every hook from public uniform draws, and runs its own staged fill. The
core, distributions, and autodiff suites check that it reproduces the wrapped
generator exactly.

## Known limits

- Dirichlet starts each column with `_fill_cursor`. An engine that repositions
  slowly pays that cost once per Dirichlet draw.
- The Enzyme rules for device fills attach to the built-in launcher, so they do
  not reach an engine's own device fill.
- Categorical and the uniform, range, collection, and sampling draws stay on the
  built-in generators.
