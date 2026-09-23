struct _CoreWord{W,O,T}
    value::T
end

function _word_constant end
function _word_from_value end
function _word_add end
function _word_xor end
function _word_rotate end
function _word_mulhilo end

@inline function _rounded_product(a, b, sign)
    product = a * b
    # Callers guarantee a nonnegative product. The dynamic sign and abs preserve
    # that value while preventing XLA from contracting its rounding into the
    # next operation. Without it, a compiled central normal draw differs from the
    # CPU result in its last bit.
    return abs(copysign(product, sign))
end

@inline _core_word(::Val{W}, ops::O, value::T) where {W,O,T} = _CoreWord{W,O,T}(value)
@inline _core_word(::Type{_CoreWord{W,O,T}}, value) where {W,O,T} =
    _CoreWord{W,O,typeof(value)}(value)

@inline _core_constant(word::Unsigned, value::Integer) = typeof(word)(value)
@inline _core_constant(word::_CoreWord{W,O}, value::Integer) where {W,O} =
    _core_word(Val(W), O(), _word_constant(O(), Val(W), word.value, value))
@inline _core_from_value(word::Unsigned, value::Integer) = value % typeof(word)
@inline _core_from_value(word::_CoreWord{W,O}, value) where {W,O} =
    _core_word(Val(W), O(), _word_from_value(O(), Val(W), word.value, value))
@inline _core_add(a::Unsigned, b::Unsigned) = a + b
@inline _core_xor(a::Unsigned, b::Unsigned) = a ⊻ b
@inline _core_rotate(value::Unsigned, count::Int) = bitrotate(value, count)

@inline function _core_add(a::_CoreWord{W,O}, b::_CoreWord{W,O}) where {W,O}
    return _core_word(typeof(a), _word_add(O(), Val(W), a.value, b.value))
end

@inline function _core_xor(a::_CoreWord{W,O}, b::_CoreWord{W,O}) where {W,O}
    return _core_word(typeof(a), _word_xor(O(), Val(W), a.value, b.value))
end

@inline function _core_rotate(value::_CoreWord{W,O}, count::Int) where {W,O}
    return _core_word(typeof(value), _word_rotate(O(), Val(W), value.value, count))
end

# A point in a core where a backend may materialize the round state. Cores
# mark the state after each round, and ChaCha marks each rotated word. A
# backend whose compiler would otherwise fuse the whole round chain into one
# kernel overrides `_word_checkpoint`.
@inline _core_checkpoint(words) = words
@inline _core_checkpoint(word::_CoreWord{W,O}) where {W,O} =
    only(_word_checkpoint(O(), (word,)))
@inline _core_checkpoint(words::Tuple{_CoreWord{W,O},Vararg{_CoreWord{W,O}}}) where {W,O} =
    _word_checkpoint(O(), words)
@inline _word_checkpoint(ops, words) = words

@inline function _mulhilo32(a::_CoreWord{32,O}, b::_CoreWord{32,O}) where {O}
    hi, lo = _word_mulhilo(O(), Val(32), a.value, b.value)
    return _core_word(typeof(a), hi), _core_word(typeof(a), lo)
end

@inline function _mulhilo64(a::_CoreWord{64,O}, b::_CoreWord{64,O}) where {O}
    hi, lo = _word_mulhilo(O(), Val(64), a.value, b.value)
    return _core_word(typeof(a), hi), _core_word(typeof(a), lo)
end

# Four-word Philox needs native x86 multiplication: LLVM 20 can miscompile its
# paired i128 products. The plain-word core stays portable for accelerators.
struct _HostWordOps{N} end

@inline _word_constant(::_HostWordOps, ::Val{W}, anchor::Unsigned, value) where {W} =
    typeof(anchor)(value)
@inline _word_from_value(::_HostWordOps, ::Val{W}, anchor::Unsigned, value) where {W} =
    value % typeof(anchor)
@inline _word_add(::_HostWordOps, ::Val{W}, a, b) where {W} = a + b
@inline _word_xor(::_HostWordOps, ::Val{W}, a, b) where {W} = a ⊻ b
@inline _word_rotate(::_HostWordOps, ::Val{W}, value, count) where {W} =
    bitrotate(value, count)
@inline function _word_mulhilo(::_HostWordOps{N}, ::Val{64}, a::UInt64, b::UInt64) where {N}
    @static if Sys.ARCH === :x86_64
        if N == 4
            return Core.Intrinsics.llvmcall(
                raw"""
                %product = call {i64, i64} asm "mulq $3", "={dx},={ax},1,r,~{flags}"(i64 %0, i64 %1)
                %hi = extractvalue {i64, i64} %product, 0
                %lo = extractvalue {i64, i64} %product, 1
                %first = insertvalue [2 x i64] undef, i64 %hi, 0
                %both = insertvalue [2 x i64] %first, i64 %lo, 1
                ret [2 x i64] %both
                """,
                NTuple{2,UInt64},
                Tuple{UInt64,UInt64},
                a,
                b,
            )
        end
    end
    product = widemul(a, b)
    return (product >> 64) % UInt64, product % UInt64
end

@inline _host_word(value::UInt64, ::Val{N}) where {N} =
    _core_word(Val(64), _HostWordOps{N}(), value)
@inline _host_words(values::NTuple{N,UInt64}, lanes::Val) where {N} =
    map(value -> _host_word(value, lanes), values)
@inline _unwrap_words(words::NTuple{N,_CoreWord}) where {N} = map(word -> word.value, words)
