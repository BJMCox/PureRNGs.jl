# Laws for the result types beyond the 32- and 64-bit words: every draw reads its
# width of MSB-first stream bits at the generator's position, so each type is
# fixed by the draws of the original types at the same position.

_u64s(rng, count) = first(rand_next(rng, UInt64, count))

@testset "narrow, wide, Float16, and complex draws compose from the word draws" begin
    for F in (Philox2x32, Philox4x32, Threefry4x64, ChaCha), bit in (0, 3)
        rng = F(0x7c1, bit)
        word = first(rand_next(rng, UInt32))
        @test rand(rng, UInt8) === (word >> 24) % UInt8
        @test rand(rng, Int8) === reinterpret(Int8, (word >> 24) % UInt8)
        @test rand(rng, UInt16) === (word >> 16) % UInt16
        @test rand(rng, Int16) === reinterpret(Int16, (word >> 16) % UInt16)
        @test rand(rng, Float16) === Float16((word >> 21) / 2048)
        hi, lo = _u64s(rng, 2)
        @test rand(rng, UInt128) === (UInt128(hi) << 64) | lo
        @test rand(rng, Int128) === reinterpret(Int128, (UInt128(hi) << 64) | lo)
        real, after_real = rand_next(rng, Float32)
        @test rand(rng, ComplexF32) === ComplexF32(real, rand(after_real, Float32))
        @test randn(rng, Float16) === Float16(randn(rng, Float32))
        @test randexp(rng, Float16) === Float16(randexp(rng, Float32))
        normal_real, after_normal = randn_next(rng, Float64)
        @test randn(rng, ComplexF64) ===
              sqrt(0.5) * ComplexF64(normal_real, randn(after_normal, Float64))
    end
end

@testset "Float16 draws lie on the 2^-11 lattice in [0, 1)" begin
    values = first(rand_next(Philox4x32(0x7c2), Float16, 50_000))
    @test all(value -> 0 <= value < 1 && isinteger(2048 * Float32(value)), values)
    @test length(unique(values)) == 2048
end

@testset "fills of the new types equal the chained scalar draws" begin
    types = (UInt8, Int16, Float16, UInt128, Int128, ComplexF16, ComplexF64)
    for F in (Philox2x32, Philox4x32, Threefry4x64, ChaCha), T in types, bit in (0, 5)
        rng = F(0x7c3, bit)
        count = 1001
        chained = Vector{T}(undef, count)
        state = rng
        for index = 1:count
            chained[index], state = rand_next(state, T)
        end
        filled, next_rng = rand_next(rng, T, count)
        @test filled == chained
        @test next_rng === state
        @test first(rand_next(rng, T, count; threaded = true)) == chained
        @test rand_at(rng, T, 400:600) == chained[400:600]
        offset = ZeroBasedVector(Vector{T}(undef, count))
        rand_next!(rng, offset)
        @test offset.data == chained
    end
    for T in (Float16, ComplexF32), F in (Philox4x32, ChaCha)
        rng = F(0x7c4, 7)
        chained = Vector{T}(undef, 300)
        state = rng
        for index = 1:300
            chained[index], state = randn_next(state, T)
        end
        filled, next_rng = randn_next(rng, T, 300)
        @test filled == chained
        @test next_rng === state
        @test randn_at(rng, T, 123) === chained[123]
    end
end

# The offset is floor(x * span / 2^K) for the K-bit candidate x read MSB-first,
# with K = 64 for spans through 2^32, 128 through 2^64, and 192 above. A span
# that fills its type makes this the raw top bits of the candidate.
function _range_oracle(rng, range)
    span = big(length(range))
    iszero(span) && (span = big(2)^(8 * sizeof(eltype(range))))
    bits = span <= big(2)^32 ? 64 : span <= big(2)^64 ? 128 : 192
    words = _u64s(rng, bits ÷ 64)
    candidate = foldl((acc, word) -> (acc << 64) | big(word), words; init = big(0))
    offset = (candidate * span) >> bits
    return eltype(range)(big(first(range)) + offset * big(step(range)))
end

@testset "128-bit integer ranges reduce exactly" begin
    ranges = (
        Int128(-3):Int128(9),
        Int128(0):(Int128(2)^40),
        UInt128(0):(UInt128(1)<<64-1),
        Int128(-7):Int128(5):(Int128(2)^100),
        UInt128(3):typemax(UInt128),
    )
    for F in (Philox4x32, Threefry2x64), range in ranges, bit in (0, 11)
        rng = F(0x7c5, bit)
        @test rand(rng, range) == _range_oracle(rng, range)
        chained = Vector{eltype(range)}(undef, 200)
        state = rng
        for index = 1:200
            chained[index], state = rand_next(state, range)
        end
        filled, next_rng = rand_next(rng, range, 200)
        @test filled == chained
        @test next_rng === state
        @test all(in(range), filled)
    end
    # The full range adds the raw 128 bits to its first element, as the 64-bit
    # full ranges do.
    for T in (Int128, UInt128)
        rng = Philox4x32(0x7c6)
        raw = rand(rng, UInt128)
        @test rand(rng, typemin(T):typemax(T)) ===
              reinterpret(T, typemin(T) % UInt128 + raw)
    end
end

@testset "128-bit fills stay on the CPU" begin
    device_rng = MLD.CUDADevice()(Philox4x32(0x7c7))
    for spec in (UInt128, Int128)
        @test_throws ArgumentError rand_next(device_rng, spec, 4)
    end
    @test_throws ArgumentError rand_next(device_rng, Int128(1):Int128(6), 4)
    @test rand(device_rng, UInt128) === rand(Philox4x32(0x7c7), UInt128)
end

# Chunks hold 1024 (128-bit) to 16384 (8-bit) elements and a fill splits from
# three chunks, so these counts cross chunk seams in the threaded fill.
@testset "threaded fills of the new types cross chunk seams" begin
    for T in (UInt8, Float16, UInt128, ComplexF64), F in (Philox4x32, ChaCha)
        rng = F(0x7c8, 9)
        count = 3 * PureRNGs._fill_chunk_elements(Val(:uniform), T) + 5
        chained = Vector{T}(undef, count)
        state = rng
        for index = 1:count
            chained[index], state = rand_next(state, T)
        end
        threaded, next_rng = rand_next(rng, T, count; threaded = true)
        @test threaded == chained
        @test next_rng === state
    end
end

@testset "128-bit populations sample by index" begin
    rng = Philox4x32(0x7c9)
    population = (UInt128(2)^127):(UInt128(2)^127+5)
    samples, next_rng = randsample_next(rng, population, 40)
    @test samples == population[first(rand_next(rng, 1:6, 40))]
    @test next_rng === last(rand_next(rng, 1:6, 40))
    @test_throws ArgumentError randsample(rng, typemin(Int128):typemax(Int128), 4)
end
