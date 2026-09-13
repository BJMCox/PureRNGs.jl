# Temporary diagnostic branch. Do not merge this script.
using PureRNGs, Test, InteractiveUtils

versioninfo()
const EXPECTED_BLOCK = (
    0x88d25590be71246e,
    0x480343a0f4db7f10,
    0xd1541e732061f086,
    0xcee887eb567e6b15,
)
const DIAGNOSTIC_DIR = joinpath("diagnostics", get(ENV, "DIAGNOSTIC_VARIANT", "baseline"))
mkpath(DIAGNOSTIC_DIR)

rng = Philox4x64(0x62a)
_, successor = Base.invokelatest(rand_next, rng, UInt64, 12)
value, _ = Base.invokelatest(rand_next, successor, UInt64)
println("[DEBUG-continuation] cache=", successor.block_words)
println("[DEBUG-continuation] next=", value, " expected=", first(EXPECTED_BLOCK))

types = Tuple{typeof(rng),Type{UInt64},Int}
open(joinpath(DIAGNOSTIC_DIR, "rand_next.ll"), "w") do io
    code_llvm(io, rand_next, types; debuginfo = :none)
end
open(joinpath(DIAGNOSTIC_DIR, "rand_next-unoptimized.ll"), "w") do io
    code_llvm(io, rand_next, types; debuginfo = :none, optimize = false)
end
open(joinpath(DIAGNOSTIC_DIR, "rand_next.asm"), "w") do io
    code_native(io, rand_next, types; debuginfo = :none)
end

@testset "Public continuation matches the fixed stream" begin
    @test value == first(EXPECTED_BLOCK)
    @test successor.block_words == EXPECTED_BLOCK
end
