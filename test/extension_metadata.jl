using TOML

@testset "R36-R38 extension metadata and base surface" begin
    project = TOML.parsefile(joinpath(pkgdir(PureRNGs), "Project.toml"))

    @test project["deps"] == Dict(
        "KernelAbstractions" => "63c18a36-062a-441e-b654-da1e3ab1ce7c",
        "MLDataDevices" => "7e8f7934-dd98-4c1a-8fe8-92b47a384d40",
        "Random" => "9a3f8284-a2c9-5f02-9a11-845980a1fd5c",
    )
    @test project["weakdeps"] == Dict(
        "AMDGPU" => "21141c5a-9bdb-4563-92ae-f87d6854732e",
        "CUDA" => "052768ef-5323-5732-b1bb-66c8b64840ba",
        "Distributions" => "31c24e10-a181-5473-b8eb-7969acd0382f",
        "EnzymeCore" => "f151be2c-9106-41f4-ab19-57ee4f262869",
        "Metal" => "dde4c033-4e86-420c-a63e-0dd931031962",
        "Reactant" => "3c362404-f566-11ee-1572-e11a4b42c853",
    )
    @test project["extensions"] == Dict(
        "PureRNGsAMDGPUExt" => "AMDGPU",
        "PureRNGsCUDAExt" => "CUDA",
        "PureRNGsDistributionsExt" => "Distributions",
        "PureRNGsEnzymeCoreExt" => "EnzymeCore",
        "PureRNGsMetalExt" => "Metal",
        "PureRNGsReactantDistributionsExt" => ["Distributions", "Reactant"],
        "PureRNGsReactantExt" => "Reactant",
    )
    @test project["compat"]["AMDGPU"] == "2"
    @test project["compat"]["CUDA"] == "5.8, 6"
    @test project["compat"]["Distributions"] == "0.25"
    @test project["compat"]["EnzymeCore"] == "0.8"
    @test project["compat"]["Metal"] == "1.7"
    @test project["compat"]["Reactant"] == "=0.2.280"

    @test Base.get_extension(PureRNGs, :PureRNGsAMDGPUExt) === nothing
    @test Base.get_extension(PureRNGs, :PureRNGsCUDAExt) === nothing
    @test Base.get_extension(PureRNGs, :PureRNGsDistributionsExt) === nothing
    @test Base.get_extension(PureRNGs, :PureRNGsEnzymeCoreExt) === nothing
    @test Base.get_extension(PureRNGs, :PureRNGsMetalExt) === nothing
    @test Base.get_extension(PureRNGs, :PureRNGsReactantExt) === nothing
    @test Base.get_extension(PureRNGs, :PureRNGsReactantDistributionsExt) ===
          nothing

    @test names(PureRNGs) == [
        :AbstractPureRNG,
        :Philox2x32,
        :Philox2x64,
        :Philox4x32,
        :Philox4x64,
        :PureRNGs,
        :StatefulRNG,
        :Threefry2x32,
        :Threefry2x64,
        :Threefry4x32,
        :Threefry4x64,
        :rand_next,
        :rand_next!,
        :randat,
        :randexp_next,
        :randexp_next!,
        :randexpat,
        :randn_next,
        :randn_next!,
        :randnat,
        :randsample,
        :randsample_next,
        :splitrng,
        :subrng,
    ]

    @test all(F(0) isa AbstractPureRNG for F in FAMILY_TYPES)
end
