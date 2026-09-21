using TOML

@testset "R36-R38 extension metadata and exports" begin
    project = TOML.parsefile(joinpath(pkgdir(PureRNGs), "Project.toml"))

    # [R36] a CPU-only user loads no KernelAbstractions, so the core suite runs
    # with its extension absent.
    @test Base.get_extension(PureRNGs, :PureRNGsKernelAbstractionsExt) === nothing
    @test !haskey(project["deps"], "KernelAbstractions")

    @test project["extensions"] == Dict(
        "PureRNGsAMDGPUExt" => "AMDGPU",
        "PureRNGsCUDAExt" => ["CUDA", "KernelAbstractions"],
        "PureRNGsDistributionsExt" => "Distributions",
        "PureRNGsEnzymeCoreExt" => "EnzymeCore",
        "PureRNGsKernelAbstractionsExt" => ["KernelAbstractions", "Adapt"],
        "PureRNGsMetalExt" => "Metal",
        "PureRNGsReactantDistributionsExt" => ["Distributions", "Reactant"],
        "PureRNGsReactantExt" => "Reactant",
    )
    declared = union(
        keys(project["deps"]),
        keys(project["weakdeps"]),
        keys(get(project, "extras", Dict())),
        ["julia"],
    )
    for key in keys(project["compat"])
        @test key in declared
    end
    for extra in keys(get(project, "extras", Dict()))
        @test extra in project["targets"]["test"]
    end
    for (_, trigger) in project["extensions"]
        for name in (trigger isa String ? [trigger] : trigger)
            @test haskey(project["weakdeps"], name)
            @test haskey(project["compat"], name)
        end
    end

    @test names(PureRNGs) == [
        :AbstractPureRNG,
        :ChaCha,
        :ChaCha12,
        :ChaCha20,
        :ChaCha8,
        :Philox2x32,
        :Philox2x64,
        :Philox2x64R6,
        :Philox4x32,
        :Philox4x32R7,
        :Philox4x64,
        :Philox4x64R7,
        :PureRNGs,
        :StatefulRNG,
        :StreamExhausted,
        :Threefry2x32,
        :Threefry2x64,
        :Threefry4x32,
        :Threefry4x32R12,
        :Threefry4x64,
        :Threefry4x64R13,
        :WeightTable,
        :rand_at,
        :rand_next,
        :rand_next!,
        :randexp_at,
        :randexp_next,
        :randexp_next!,
        :randn_at,
        :randn_next,
        :randn_next!,
        :randsample,
        :randsample!,
        :randsample_next,
        :randsample_next!,
        :rngkey,
        :rngposition,
        :splitrng,
        :subrng,
    ]
end
