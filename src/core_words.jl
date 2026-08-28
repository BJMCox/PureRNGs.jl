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
    # next operation. Without it, the central R43 probe changes from
    # 0xbff55e55782ee12e to 0xbff55e55782ee12d.
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

@inline function _mulhilo32(a::_CoreWord{32,O}, b::_CoreWord{32,O}) where {O}
    hi, lo = _word_mulhilo(O(), Val(32), a.value, b.value)
    return _core_word(typeof(a), hi), _core_word(typeof(a), lo)
end

@inline function _mulhilo64(a::_CoreWord{64,O}, b::_CoreWord{64,O}) where {O}
    hi, lo = _word_mulhilo(O(), Val(64), a.value, b.value)
    return _core_word(typeof(a), hi), _core_word(typeof(a), lo)
end
