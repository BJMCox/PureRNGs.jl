# Vectors from draft-strombergson-chacha-test-vectors: 256-bit key, 64-bit
# nonce in the two words after the 64-bit block counter, keystream bytes given
# little-endian. Block 1 checks the counter placement.
_chacha_words(hex::AbstractString) = Tuple(reinterpret(UInt32, hex2bytes(hex)))
_chacha_counter(block::Integer, nonce::AbstractString) =
    (UInt32(block), UInt32(0), _chacha_words(nonce)...)

const CHACHA_ZERO_KEY = ntuple(_ -> UInt32(0), 8)
const CHACHA_TC7_KEY =
    _chacha_words("00112233445566778899aabbccddeeffffeeddccbbaa99887766554433221100")
const CHACHA_TC7_NONCE = "0f1e2d3c4b5a6978"

@testset "ChaCha20 known-answer vectors" begin
    core = PureRNGs._chacha
    @test core(_chacha_counter(0, "0000000000000000"), CHACHA_ZERO_KEY, Val(20)) ==
          _chacha_words(
        "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7" *
        "da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586",
    )
    @test core(_chacha_counter(1, "0000000000000000"), CHACHA_ZERO_KEY, Val(20)) ==
          _chacha_words(
        "9f07e7be5551387a98ba977c732d080dcb0f29a048e3656912c6533e32ee7aed" *
        "29b721769ce64e43d57133b074d839d531ed1f28510afb45ace10a1f4b794d6f",
    )
    @test core(_chacha_counter(0, CHACHA_TC7_NONCE), CHACHA_TC7_KEY, Val(20)) ==
          _chacha_words(
        "9fadf409c00811d00431d67efbd88fba59218d5d6708b1d685863fabbb0e961e" *
        "ea480fd6fb532bfd494b2151015057423ab60a63fe4f55f7a212e2167ccab931",
    )
end

@testset "ChaCha12 known-answer vectors" begin
    core = PureRNGs._chacha
    @test core(_chacha_counter(0, "0000000000000000"), CHACHA_ZERO_KEY) == _chacha_words(
        "9bf49a6a0755f953811fce125f2683d50429c3bb49e074147e0089a52eae155f" *
        "0564f879d27ae3c02ce82834acfa8c793a629f2ca0de6919610be82f411326be",
    )
    @test core(_chacha_counter(0, CHACHA_TC7_NONCE), CHACHA_TC7_KEY, Val(12)) ==
          _chacha_words(
        "7ed12a3a63912ae941ba6d4c0d5e862e568b0e5589346935505f064b8c2698db" *
        "f7d850667d8e67be639f3b4f6a16f92e65ea80f6c7429445da1fc2c1b9365040",
    )
end

@testset "ChaCha8 known-answer vectors" begin
    core = PureRNGs._chacha
    @test core(_chacha_counter(0, "0000000000000000"), CHACHA_ZERO_KEY, Val(8)) ==
          _chacha_words(
        "3e00ef2f895f40d67f5bb8e81f09a5a12c840ec3ce9a7f3b181be188ef711a1e" *
        "984ce172b9216f419f445367456d5619314a42a3da86b001387bfdb80e0cfe42",
    )
    @test core(_chacha_counter(0, CHACHA_TC7_NONCE), CHACHA_TC7_KEY, Val(8)) ==
          _chacha_words(
        "db43ad9d1e842d1272e4530e276b3f568f8859b3f7cf6d9d2c74fa53808cb515" *
        "7a8ebf46ad3dcc4b6c7dadde131784b0120e0e22f6d5f9ffa7407d4a21b695d9",
    )
end

@testset "ChaCha stream layout" begin
    rng = ChaCha(0)
    # The generator's first block is the zero-key, zero-nonce keystream block.
    @test rng.block_words == PureRNGs._block_words(
        PureRNGs._chacha(_chacha_counter(0, "0000000000000000"), CHACHA_ZERO_KEY),
    )
    @test PureRNGs._block_bits(rng) == 512
    # Draws read the block words from the most significant bit, so the first
    # `UInt64` is the first two little-endian keystream words.
    @test rand(rng, UInt32) == 0x6a9af49b
    @test rand(rng, UInt64) == 0x6a9af49b53f95507
    @test ChaCha12(0) === rng
    @test rand(ChaCha20(0), UInt32) == 0xade0b876
    @test rand(ChaCha8(0), UInt32) == 0x2fef003e
end

# RFC 8439 section 2.3.2, with the four counter words given as raw state words 12
# to 15: the 32-bit block counter 1 followed by the three nonce words.
@testset "ChaCha20 RFC 8439 block function" begin
    key = ntuple(i -> reinterpret(UInt32, UInt8.(4(i-1) .+ (0:3)))[1], 8)
    counter = (0x00000001, 0x09000000, 0x4a000000, 0x00000000)
    @test PureRNGs._chacha(counter, key, Val(20)) == (
        0xe4e7f110,
        0x15593bd1,
        0x1fdd0f50,
        0xc47120a3,
        0xc7f4d1c7,
        0x0368c033,
        0x9aaa2204,
        0x4e6cd4c3,
        0x466482d2,
        0x09aa9f07,
        0x05d7c214,
        0xa2028bd9,
        0xd19c12b5,
        0xb94e16de,
        0xe883d0cb,
        0x4e3c50a2,
    )
end
