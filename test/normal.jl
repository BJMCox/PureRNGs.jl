const REFERENCE_AS241_32 = (
    (5.9109374720f1, 1.5929113202f2, 5.0434271938f1, 3.3871327179f0),
    (6.7187563600f1, 7.8757757664f1, 1.7895169469f1, 1.0f0),
    (1.7023821103f-1, 1.3067284816f0, 2.7568153900f0, 1.4234372777f0),
    (1.2021132975f-1, 7.3700164250f-1, 1.0f0),
    (1.7337203997f-2, 4.2868294337f-1, 3.0812263860f0, 6.6579051150f0),
    (1.2258202635f-2, 2.4197894225f-1, 1.0f0),
)

const REFERENCE_AS241_64 = (
    (
        2.5090809287301226727e3,
        3.3430575583588128105e4,
        6.7265770927008700853e4,
        4.5921953931549871457e4,
        1.3731693765509461125e4,
        1.9715909503065514427e3,
        1.3314166789178437745e2,
        3.3871328727963666080,
    ),
    (
        5.2264952788528545610e3,
        2.8729085735721942674e4,
        3.9307895800092710610e4,
        2.1213794301586595867e4,
        5.3941960214247511077e3,
        6.8718700749205790830e2,
        4.2313330701600911252e1,
        1.0,
    ),
    (
        7.74545014278341407640e-4,
        2.27238449892691845833e-2,
        2.41780725177450611770e-1,
        1.27045825245236838258,
        3.64784832476320460504,
        5.76949722146069140550,
        4.63033784615654529590,
        1.42343711074968357734,
    ),
    (
        1.05075007164441684324e-9,
        5.47593808499534494600e-4,
        1.51986665636164571966e-2,
        1.48103976427480074590e-1,
        6.89767334985100004550e-1,
        1.67638483018380384940,
        2.05319162663775882187,
        1.0,
    ),
    (
        2.01033439929228813265e-7,
        2.71155556874348757815e-5,
        1.24266094738807843860e-3,
        2.65321895265761230930e-2,
        2.96560571828504891230e-1,
        1.78482653991729133580,
        5.46378491116411436990,
        6.65790464350110377720,
    ),
    (
        2.04426310338993978564e-15,
        1.42151175831644588870e-7,
        1.84631831751005468180e-5,
        7.86869131145613259100e-4,
        1.48753612908506148525e-2,
        1.36929880922735805310e-1,
        5.99832206555887937690e-1,
        1.0,
    ),
)

function _reference_horner(x::T, coefficients::NTuple{N,T}) where {T,N}
    value = coefficients[1]
    for index = 2:N
        value = fma(value, x, coefficients[index])
    end
    return value
end

function _reference_as241(u::T, coefficients) where {T}
    A, B, C, D, E, F = coefficients
    q = u - T(0.5)
    if abs(q) <= T(0.425)
        r = T(0.180625) - q * q
        return q * (_reference_horner(r, A) / _reference_horner(r, B))
    end
    r = sqrt(-log(q < zero(T) ? u : one(T) - u))
    if r <= T(5)
        r = r - T(1.6)
        z = _reference_horner(r, C) / _reference_horner(r, D)
    else
        r = r - T(5)
        z = _reference_horner(r, E) / _reference_horner(r, F)
    end
    return q < zero(T) ? -z : z
end

_reference_as241(u::Float32) = _reference_as241(u, REFERENCE_AS241_32)
_reference_as241(u::Float64) = _reference_as241(u, REFERENCE_AS241_64)

function _reference_normal(rng, ::Type{T}) where {T}
    width = _transformed_width(T)
    raw = _reference_extract(
        rng,
        _reference_position_block(rng.position),
        rng.position.bit,
        width,
    )
    u = if T === Float32
        Float32(((raw % UInt32) << UInt32(1)) | UInt32(1)) * Float32(0x1p-24)
    else
        Float64((raw << UInt64(1)) | UInt64(1)) * Float64(0x1p-53)
    end
    return _reference_as241(u)
end

normal_allocations(rng, ::Type{T}) where {T} = (
    @allocated(randn(rng, T)),
    @allocated(randn_next(rng, T)),
    @allocated(randn_at(rng, T, 3)),
)

@testset "AS241 definition" begin
    # The Float32 table stops at the near tail: the midpoint sweep below shows
    # no Float32 input reaches r > 5, so the far-tail pair is never evaluated.
    @test IR._as241_coefficients(Float32) === REFERENCE_AS241_32[1:4]
    @test IR._as241_coefficients(Float64) === REFERENCE_AS241_64

    for (T, inputs) in
        ((Float32, (0.5f0, 0.95f0, Float32(0x1p-24))), (Float64, (0.5, 0.95, 1.0e-20)))
        for u in inputs
            @test IR._as241(u) === _reference_as241(u)
        end
    end

    # The Float32 transform omits the far tail because the midpoint lattice
    # never reaches r = 5. `_reference_as241` keeps it, so the complete lattice
    # is the proof that omitting it moves no value and no `r` gets that far.
    mismatches = 0
    widest = 0.0f0
    for k = UInt64(0):UInt64(2^23-1)
        u = IR._open_midpoint(Float32, k)
        IR._as241(u) === _reference_as241(u) || (mismatches += 1)
        q = u - 0.5f0
        abs(q) <= 0.425f0 && continue
        widest = max(widest, sqrt(-log(q < 0.0f0 ? u : 1.0f0 - u)))
    end
    @test mismatches == 0
    @test widest < 5.0f0

    @test IR._open_midpoint(Float32, UInt64(0)) === Float32(0x1p-24)
    @test IR._open_midpoint(Float32, UInt64((UInt64(1) << 23) - 1)) ===
          one(Float32) - Float32(0x1p-24)
    @test IR._open_midpoint(Float64, UInt64(0)) === Float64(0x1p-53)
    @test IR._open_midpoint(Float64, (UInt64(1) << 52) - 1) ===
          one(Float64) - Float64(0x1p-53)
end

@testset "backend token selects the normal transform" begin
    # A device-placed generator drawn on the host keeps its token, so the same
    # midpoint reaches a different transform. Only the final value moves: the
    # raw bits, the midpoint, and the returned position are the CPU ones.
    for T in NORMAL_TYPES, bit in (UInt16(0), UInt16(29))
        cpu_rng = _positioned(Philox4x32, 0x74f, UInt64(2), bit)
        width = _transformed_width(T)
        raw = _reference_extract(
            cpu_rng,
            _reference_position_block(cpu_rng.position),
            cpu_rng.position.bit,
            width,
        )
        u = IR._open_midpoint(T, raw)
        @test IR._normal_transform(IR._CPU_BACKEND, u) === _reference_as241(u)
        for backend in (IR._CUDA_BACKEND, IR._AMDGPU_BACKEND, IR._METAL_BACKEND)
            rng = IR._rebuild(cpu_rng, cpu_rng.position, backend)
            expected = IR._normal_transform(backend, u)
            value, next_rng = randn_next(rng, T)
            @test value === expected
            @test randn(rng, T) === expected
            @test randn_at(rng, T, 1) === expected
            @test next_rng.position == _reference_position(cpu_rng, width)
            @test expected !== _reference_as241(u)
        end
    end
end

@testset "packed scalar normals" begin
    # The public draw is the AS241 reference applied to the midpoint of the raw
    # the generator stands on, at every alignment of the significand.
    for F in GENERATOR_TYPES, T in NORMAL_TYPES
        for bit in (UInt16(0), UInt16(23), UInt16(51), UInt16(63))
            rng = _positioned(F, 0x742, UInt64(9), bit)
            @test randn(rng, T) === _reference_normal(rng, T)
        end
    end
end

@testset "normal generator and mixed positions" begin
    # A normal draw leaves the stream where a uniform draw can carry on.
    for F in GENERATOR_TYPES, T in NORMAL_TYPES
        rng = _positioned(F, 0x743, UInt64(4), UInt16(61))
        _, next_rng = randn_next(rng, T)
        final_rng_value, final_rng = rand_next(next_rng, UInt32)
        @test final_rng_value === rand(next_rng, UInt32)
        @test final_rng.position ==
              _reference_position(rng, _transformed_width(T) + _uniform_width(UInt32))

        terminal_rng = _terminal_rng(F, _transformed_width(T))
        _, exhausted = randn_next(terminal_rng, T)
        @test exhausted.position.bit === IR._EXHAUSTED_BIT
    end
end

@testset "Position128 normal low-word carry" begin
    for F in (Philox4x64, Threefry4x64), T in NORMAL_TYPES
        rng = F(0x746)
        bit = UInt16(IR._block_bits(rng) - _transformed_width(T))
        position = IR._Position128(typemax(UInt64), UInt64(7), bit)
        rng = IR._rebuild(rng, position, rng.device)
        _, next_rng = randn_next(rng, T)
        @test next_rng.position == IR._Position128(UInt64(0), UInt64(8), UInt16(0))
    end
end

@testset "normal fixed-work and codegen" begin
    for F in GENERATOR_TYPES
        rng = F(0x744)
        default_value, default_next = randn_next(rng)
        typed_value, typed_next = randn_next(rng, Float64)
        @test default_next === typed_next
        @test default_value === typed_value

        @test_throws ArgumentError randn(rng)

        for T in NORMAL_TYPES
            normal_allocations(rng, T)
            @test normal_allocations(rng, T) == (0, 0, 0)
        end
    end

    rng = MLD.CUDADevice()(Philox4x64(0x745))
    for (function_, signature) in (
        (randn, Tuple{typeof(rng),Type{Float64}}),
        (randn_next, Tuple{typeof(rng),Type{Float32}}),
        (randn_at, Tuple{typeof(rng),Type{Float64},Int}),
        (randn_next!, Tuple{typeof(rng),Vector{Float64}}),
    )
        typed_ir, llvm_ir = _codegen_ir(function_, signature)
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        @test !occursin(r"\bi128\b", llvm_ir)
    end

    @test_throws ArgumentError randn_at(rng, Float64, 0)
    @test_throws ArgumentError randn_at(rng, Float32, -1)
end
