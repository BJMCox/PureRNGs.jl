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
| `_child_cursor(rng, purpose::UInt64)` | `(child_rng, cursor)` | The child stream for `purpose`, as the engine's `subrng` derives it, and a cursor at its start. The Gamma fallback reads it through `_take_bits` alone, without a length bound. |
| `_draw_cursor(rng, count, Val(W))` | `(cursor, next_rng)` | Reserves `count` takes at the held position: the cursor of the first take and the generator past the last. The engine applies its alignment and exhaustion rules here. |
| `_addressed_state(rng, count, Val(W), i)` | generator | The generator at the start of draw `i`, counting from one, of `count`-take draws. `rng` does not change. Throws `ArgumentError` for `i < 1` and the engine's exhaustion error when draw `i` ends past the stream end. `i` can be any `Integer`, including `BigInt`. |
| `_fill_cursor(rng, count, Val(W), ordinal::UInt64)` | `cursor` | The cursor of zero-based draw `ordinal` of a fill that starts at the held position. The fill has reserved its span, so this hook does not check. |

Optional overrides, with defaults built from the hooks above:

| Hook | Default |
|:--|:--|
| `_reserve_draws(rng, draws, count, Val(W))` | `_draw_cursor` with `draws * count` takes. It throws `OverflowError` when that product passes `typemax(UInt64)`. Returns `next_rng`. |
| `_engine_draw_next(rng, codec, T)` | `_draw_cursor`, then one `_codec_take`. |
| `_engine_draw_at(rng, codec, T, i)` | `_addressed_state`, `_draw_cursor`, then one `_codec_take`. |
| `_engine_fill!(rng, destination, threaded, codec)` | `_reserve_draws`, then a host loop over one cursor, or over one `_fill_cursor` per chunk when `threaded`. It throws `ArgumentError` for a device-bound generator. Returns `(destination, next_rng)`. |
| `_engine_fill_columns!(rng, destination::AbstractMatrix, threaded, codec)` | `_reserve_draws` for `size(destination, 2)` draws, then on the host one `_fill_cursor` and `_column_take!` per column, and on a device the same per column when there is a column for every resident workitem (`_device_workitems(backend)`), otherwise one `_fill_cursor` and `_component_log` per element followed by `_normalize_column!` per column. Returns `(destination, next_rng)`. |

Override `_engine_fill!` to keep bulk generation fast. The default walks a cursor
take by take, so an engine whose stream is cheapest in whole blocks or rows
should generate those blocks and run `_codec_take` over a cursor into them. A
device fill needs this override: a kernel can call `_codec_take` on the
engine's own cursor type.

## Spans

The core never multiplies a count or an index before a hook sees it:

- `_reserve_draws` gets the draw count and the per-draw take count as separate
  factors.
- `_addressed_state` gets the per-draw take count and the index as they are.
- `count` in `_draw_cursor` is one draw's takes, or a product the core has
  checked.

The engine checks the whole span, `draws * count * W` bits or the end of draw
`i`, before it returns. Check with division or wide arithmetic, because the
product can pass `typemax(UInt64)`. A fill calls `_reserve_draws` before it
writes anything, so a rejected span leaves the destination untouched. Override
`_reserve_draws` to throw the engine's own exhaustion error instead of the
default's `OverflowError`.

## Column draws

A multivariate draw that fills one matrix column per draw uses a *column codec*.
Dirichlet is the column codec `PureRNGs._DirichletCodec(alpha)`, with `alpha` on
the engine's backend:

```julia
count, width = PureRNGs._codec_takes(codec, T)   # one column's takes
cursor = PureRNGs._column_take!(codec, rng, cursor, destination, column)
```

`_column_take!` writes `destination[:, column]` from the column's takes and
returns the cursor past them. It reads one Gamma span per component through
`_take_bits` and `_skip_takes`, then normalizes the column by log-sum-exp. The
Gamma and log-Gamma mathematics and the normalization stay in PureRNGs, in the
two halves of the column:

```julia
count, width = PureRNGs._component_takes(codec, T)   # one component's takes
value = PureRNGs._component_log(codec, rng, cursor, component, T)
PureRNGs._normalize_column!(destination, column)     # log values to the draw
```

The components of a column sit in consecutive spans, so component `c` of column
`j` is draw `(j - 1) * length(alpha) + c` of a fill of `count`-take draws. An
engine whose bulk fill runs one workitem per draw can fill the log-gamma matrix
as that element fill, `_component_log` per element, and then run
`_normalize_column!` per column. The draws equal the column fill's.

Override `_engine_fill_columns!` to feed the columns from the engine's bulk
stream:

1. Call `_reserve_draws(rng, size(destination, 2), count, width)` first.
2. Start a cursor at the held position.
3. Run `_column_take!` column by column. Each call starts where the previous
   one returned.

A threaded or device override starts each chunk or workitem at its own column
start, as `_fill_cursor` with ordinal `column - 1` does. Addressed draw `i`
fills one column from `_addressed_state(rng, count, width, i)`, so it passes the
engine's `_engine_fill_columns!` too.

A device override passes the codec to a kernel. The KernelAbstractions
extension defines `Adapt.adapt_structure` for the codec, so its `alpha` becomes
a device argument.

### Differentiation

`_component_log` gets the component's log-Gamma draw from
`_gamma_log_value(shape, gamma_codec, rng, ordinal, cursor)`, with `ordinal`
from `_cursor_ordinal`. The AD rules attach there:

- ForwardDiff: a dual shape dispatches to the rule's method. The column and the
  destination hold duals, and the normalization differentiates as arithmetic.
- Mooncake and Enzyme: rules on `_gamma_log_value` supply the implicit shape
  derivative for float shapes.

An override keeps these rules only when it calls `_column_take!` or
`_component_log` and no Gamma code of its own. The Enzyme rules for device
fills still attach to the built-in launcher only.

## Public methods an engine defines

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
serves every hook from public uniform draws, and runs its own staged fill. A
second engine in the same file defines only the required hooks and takes every
default. The core, distributions, and autodiff suites check that the first
reproduces the wrapped generator exactly, and the core suite checks the second.

## Known limits

- The default `_engine_fill_columns!` starts each host column, and each device
  element, with `_fill_cursor`. An engine that repositions slowly should
  override it.
- The Enzyme rules for device fills attach to the built-in launcher, so they do
  not reach an engine's own device fill.
- Categorical and the uniform, range, collection, and sampling draws stay on the
  built-in generators.
