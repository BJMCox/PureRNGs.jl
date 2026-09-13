# Temporary diagnostic branch. Do not merge this script.
using PureRNGs, Test, InteractiveUtils

const IR = PureRNGs
const GENERATOR_TYPES = (
    Philox2x32,
    Philox4x32,
    Philox2x64,
    Philox4x64,
    Threefry2x32,
    Threefry4x32,
    Threefry2x64,
    Threefry4x64,
    ChaCha,
)
const PURE_UNIFORM_TYPES = (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)

# The existing allocating suite uses this helper from test/uniform.jl.
function _reference_position(rng, additional_bits::Integer)
    block_bits = BigInt(IR._block_bits(rng))
    position = rng.position
    block =
        position isa IR._Position64 ? BigInt(position.block) :
        (BigInt(position.hi) << 64) + BigInt(position.lo)
    total = BigInt(position.bit) + BigInt(additional_bits)
    block_delta, bit = divrem(total, block_bits)
    block += block_delta
    if position isa IR._Position64
        return IR._Position64(UInt64(block), UInt16(bit))
    end
    return IR._Position128(
        UInt64(block & typemax(UInt64)),
        UInt64(block >> 64),
        UInt16(bit),
    )
end

versioninfo()
const DIAGNOSTIC_DIR = joinpath("diagnostics", "O$(Base.JLOptions().opt_level)")
mkpath(DIAGNOSTIC_DIR)
failed = false
try
    include("uniform_allocating.jl")
catch err
    cause = err isa LoadError ? err.error : err
    cause isa Test.TestSetException || rethrow()
    global failed = true
    println("[DEBUG-continuation] Original allocating suite failed")
end

function probe(seed, ::Type{T}) where {T}
    rng = Philox4x64(seed)
    values, successor = rand_next(rng, T, 12)
    scalar_values = Vector{T}(undef, 12)
    cursor = rng
    for index in eachindex(scalar_values)
        scalar_values[index], cursor = rand_next(cursor, T)
    end
    reserved = IR._reserve(rng, UInt64(12) * UInt64(IR._draw_bits(T)), UInt64(0))
    position = successor.position
    counter = (position.lo, position.hi, UInt64(0), UInt64(0))
    portable = IR._philox4x64(counter, rng.key)
    native = IR._core_block(typeof(rng), rng.key, (position.lo, position.hi))
    next_batch_value = first(rand_next(successor, T))
    next_scalar_value = first(rand_next(cursor, T))
    return (;
        type = T,
        samples_equal = values == scalar_values,
        state_equal = successor === cursor,
        next_value_equal = next_batch_value == next_scalar_value,
        next_batch_value,
        next_scalar_value,
        position,
        batch_words = successor.block_words,
        scalar_words = cursor.block_words,
        reserved_words = reserved.block_words,
        portable_words = portable,
        native_words = native,
    )
end

open(joinpath(DIAGNOSTIC_DIR, "probes.txt"), "w") do io
    for T in (UInt64, Int64, Float64)
        result = probe(0x62a, T)
        println(io, result)
        println("[DEBUG-continuation] ", result)
        global failed |=
            !(result.samples_equal && result.state_equal && result.next_value_equal)
    end
end

function dispatched_probe(seed, ::Type{T}) where {T}
    rng = Philox4x64(seed)
    _, successor = Base.invokelatest(rand_next, rng, T, 12)
    expected = foldl(
        (state, _) -> last(rand_next(state, T)), 1:12; init = rng,
    )
    bits = UInt64(12) * UInt64(IR._draw_bits(T))
    reserved = Base.invokelatest(IR._reserve, rng, bits, UInt64(0))
    position = successor.position
    native = Base.invokelatest(
        IR._core_block, typeof(rng), rng.key, (position.lo, position.hi),
    )
    result = (;
        type = T,
        state_equal = successor === expected,
        batch_words = successor.block_words,
        expected_words = expected.block_words,
        reserved_words = reserved.block_words,
        native_words = native,
        next_batch = first(Base.invokelatest(rand_next, successor, T)),
        next_expected = first(Base.invokelatest(rand_next, expected, T)),
    )
    println("[DEBUG-continuation] dispatched ", result)
    return result
end

for T in (UInt64, Int64, Float64)
    result = dispatched_probe(0x62a, T)
    global failed |= !result.state_equal
end

function batch12(rng)
    return rand_next(rng, UInt64, 12)
end

rng = Philox4x64(0x62a)
open(joinpath(DIAGNOSTIC_DIR, "batch12.ll"), "w") do io
    code_llvm(io, batch12, Tuple{typeof(rng)}; debuginfo = :none)
end
open(joinpath(DIAGNOSTIC_DIR, "batch12.asm"), "w") do io
    code_native(io, batch12, Tuple{typeof(rng)}; debuginfo = :none)
end

for (name, fn, types) in (
    ("rand_next", rand_next, Tuple{typeof(rng),Type{UInt64},Int}),
    ("reserve", IR._reserve, Tuple{typeof(rng),UInt64,UInt64}),
)
    open(joinpath(DIAGNOSTIC_DIR, "$name.ll"), "w") do io
        code_llvm(io, fn, types; debuginfo = :none)
    end
    open(joinpath(DIAGNOSTIC_DIR, "$name.asm"), "w") do io
        code_native(io, fn, types; debuginfo = :none)
    end
end

failed && error("Continuation diagnostic reproduced a mismatch")
