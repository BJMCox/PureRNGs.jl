# The laws that hold for every draw family, each stated once over
# `DRAW_FAMILIES`. Every law body is a function so the closures a family record
# carries stay concretely typed, which the allocation law depends on.

function _scalar_draw_law(family, spec, F)
    start = _positioned(F, 0x742, UInt64(9), UInt16(47))
    width = family.width(spec)
    _, expected = _reference_chain(start, cursor -> family.draw(cursor, spec), width, 9)

    value, next_rng = family.draw_next(start, spec)
    @test value === expected[1]
    @test next_rng.position == _reference_position(start, width)

    # The addressed draw lands where a chain of draws of the same width reaches.
    for index in eachindex(expected)
        @test family.draw_at(start, spec, index) === expected[index]
    end
    # Every form is pure: none of them moved the generator they read.
    @test family.draw(start, spec) === expected[1]
    @test start.position == _positioned(F, 0x742, UInt64(9), UInt16(47)).position

    @test_throws ArgumentError family.draw_at(start, spec, 0)
    @test_throws ArgumentError family.draw_at(start, spec, -1)
    return nothing
end

function _scalar_capacity_law(family, spec, F)
    last = _terminal_rng(F, family.width(spec))
    position = last.position
    @test family.draw_at(last, spec, 1) === family.draw(last, spec)
    @test last.position == position
    @test_throws StreamExhausted family.draw_at(last, spec, 2)

    exhausted = IR._rebuild(last, _terminal_position(last), last.device)
    @test_throws StreamExhausted family.draw(exhausted, spec)
    @test_throws StreamExhausted family.draw_next(exhausted, spec)
    @test_throws StreamExhausted family.draw_at(exhausted, spec, 1)
    @test exhausted.position == _terminal_position(last)
    return nothing
end

function _bulk_fill_law(family, spec, F)
    T = family.element(spec)
    rng = _positioned(F, 0x747, UInt64(6), UInt16(0))
    chain_rng, expected =
        _reference_chain(rng, cursor -> family.draw(cursor, spec), family.width(spec), 17)

    serial = Vector{T}(undef, 17)
    threaded = similar(serial)
    @test family.fill!(rng, serial, spec; threaded = false) === serial
    @test family.fill!(rng, threaded, spec; threaded = true) === threaded
    @test serial == threaded == expected
    @test rng.position.bit === UInt16(0)

    continued, continued_rng =
        family.fill_next!(rng, similar(serial), spec; threaded = false)
    @test continued == expected
    @test continued_rng.position == chain_rng.position

    matrix, matrix_rng = family.allocate(rng, spec, 1, 17)
    @test vec(matrix) == expected
    @test size(matrix) == (1, 17)
    @test matrix_rng.position == chain_rng.position
    @test vec(family.allocate_pure(rng, spec, 1, 17)) == expected
    return nothing
end

function _strided_view_law(family, spec)
    T = family.element(spec)
    rng = _positioned(Philox4x32, 0x748, UInt64(3), UInt16(61))
    chain_rng, expected =
        _reference_chain(rng, cursor -> family.draw(cursor, spec), family.width(spec), 12)

    storage = zeros(T, 24)
    destination = @view storage[2:2:24]
    returned, view_rng = family.fill_next!(rng, destination, spec; threaded = false)
    @test returned === destination
    @test collect(destination) == expected
    @test all(iszero, @view storage[1:2:23])
    @test view_rng.position == chain_rng.position

    threaded_storage = zeros(T, 24)
    threaded_view = @view threaded_storage[2:2:24]
    family.fill!(rng, threaded_view, spec; threaded = true)
    @test collect(threaded_view) == expected
    @test all(iszero, @view threaded_storage[1:2:23])
    return nothing
end

function _grouped_fill_law(family, spec, count, bit)
    T = family.element(spec)
    rng = _positioned(Philox4x32, 0x74b, UInt64(9), bit)
    chain_rng, expected = _reference_chain(
        rng,
        cursor -> family.draw(cursor, spec),
        family.width(spec),
        count,
    )

    serial = Vector{T}(undef, count)
    _, serial_rng = family.fill_next!(rng, serial, spec; threaded = false)
    threaded = similar(serial)
    family.fill!(rng, threaded, spec; threaded = true)
    @test serial == expected
    @test threaded == expected
    @test serial_rng.position == chain_rng.position
    return nothing
end

function _small_allocating_law(family, spec, count)
    rng = _positioned(Philox4x32, 0x74b1, UInt64(5), UInt16(61))
    chain_rng, expected = _reference_chain(
        rng,
        cursor -> family.draw(cursor, spec),
        family.width(spec),
        count,
    )
    values, next_rng = family.allocate(rng, spec, count)
    @test values == expected
    @test next_rng.position == chain_rng.position
    return nothing
end

function _empty_destination_law(family, spec)
    rng = Philox4x32(0x74c)
    exhausted = IR._rebuild(rng, _terminal_position(rng), rng.device)

    empty = family.element(spec)[]
    @test family.fill!(exhausted, empty, spec; threaded = false) === empty
    returned, empty_rng = family.fill_next!(exhausted, empty, spec; threaded = false)
    @test returned === empty
    @test empty_rng === exhausted

    allocated, allocated_rng = family.allocate(exhausted, spec, 0)
    @test isempty(allocated)
    @test allocated_rng === exhausted
    @test isempty(family.allocate_pure(exhausted, spec, 0))
    return nothing
end

function _exhaustion_law(family, spec, F)
    T = family.element(spec)
    width = family.width(spec)
    last = _terminal_rng(F, width)
    position = last.position

    destination = fill(one(T), 2)
    before = copy(destination)
    @test_throws StreamExhausted family.fill!(last, destination, spec; threaded = false)
    @test destination == before
    @test_throws StreamExhausted family.fill_next!(
        last,
        destination,
        spec;
        threaded = false,
    )
    @test destination == before
    @test last.position == position

    final = Vector{T}(undef, 1)
    returned, final_rng = family.fill_next!(last, final, spec; threaded = false)
    @test returned === final
    @test final[1] === family.draw(last, spec)
    @test final_rng.position == _terminal_position(last)

    allocated, allocated_rng = family.allocate(last, spec, 1)
    @test allocated == final
    @test allocated_rng.position == _terminal_position(last)
    @test family.allocate_pure(last, spec, 1) == final

    # A one-bit draw ends on the last bit of the stream, so no position is short
    # of it by one.
    if width > 1
        insufficient = IR._rebuild(
            last,
            _position_from_absolute(last, _stream_capacity(last) - width + 1),
            last.device,
        )
        @test_throws StreamExhausted family.allocate(insufficient, spec, 1)
        @test_throws StreamExhausted family.allocate_pure(insufficient, spec, 1)
    end
    return nothing
end

function _serial_fill_allocation_law(family, spec, F)
    rng = F(0x74d)
    destination = Vector{family.element(spec)}(undef, 7)
    allocations = _serial_fill_allocations(rng, destination) do generator, buffer
        family.fill_next!(generator, buffer, spec; threaded = false)
    end
    @test allocations == 0
    return nothing
end

@testset "R23, R25, and R29 scalar and addressed draws" begin
    for family in DRAW_FAMILIES, spec in family.specs, F in GENERATOR_TYPES
        _scalar_draw_law(family, spec, F)
        _scalar_capacity_law(family, spec, F)
    end
end

@testset "R23, R24, and R26 bulk fills equal the scalar chain" begin
    for family in DRAW_FAMILIES, spec in family.specs
        for F in GENERATOR_TYPES
            _bulk_fill_law(family, spec, F)
        end
        _strided_view_law(family, spec)
    end
end

@testset "R23 and R26 grouped fills equal the scalar chain" begin
    # The Philox4x32 fill decodes whole groups of aligned draws, so the stream
    # has to match the scalar chain on either side of a group boundary, at a
    # start bit no group can align to, and at the last bit of a block.
    for family in DRAW_FAMILIES
        for spec in family.specs,
            count in (1, 31, 32, 33, 1000),
            bit in (UInt16(0), UInt16(3), UInt16(61), UInt16(127))

            _grouped_fill_law(family, spec, count, bit)
        end
        # One long fill per family crosses many groups and every chunk seam.
        for bit in (UInt16(0), UInt16(3))
            _grouped_fill_law(family, first(family.specs), 100_003, bit)
        end
    end
end

@testset "R26 small allocating fill boundary" begin
    # 128 elements take the small allocating path, 129 the dense one.
    for family in DRAW_FAMILIES, spec in family.specs, count in (128, 129)
        _small_allocating_law(family, spec, count)
    end
end

@testset "R30, R39, R40, and R54 fill exhaustion and preflight" begin
    for family in DRAW_FAMILIES, spec in family.specs
        _empty_destination_law(family, spec)
        for F in GENERATOR_TYPES
            _exhaustion_law(family, spec, F)
        end
    end
end

@testset "R23 and R30 serial fills allocate nothing" begin
    for family in DRAW_FAMILIES, spec in family.specs, F in GENERATOR_TYPES
        _serial_fill_allocation_law(family, spec, F)
    end
end
