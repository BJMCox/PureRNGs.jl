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
        "Metal" => "dde4c033-4e86-420c-a63e-0dd931031962",
    )
    @test project["extensions"] == Dict(
        "PureRNGsAMDGPUExt" => "AMDGPU",
        "PureRNGsCUDAExt" => "CUDA",
        "PureRNGsMetalExt" => "Metal",
    )
    @test project["compat"]["AMDGPU"] == "2"
    @test project["compat"]["CUDA"] == "5.8, 6"
    @test project["compat"]["Metal"] == "1.7"

    @test Base.get_extension(PureRNGs, :PureRNGsAMDGPUExt) === nothing
    @test Base.get_extension(PureRNGs, :PureRNGsCUDAExt) === nothing
    @test Base.get_extension(PureRNGs, :PureRNGsMetalExt) === nothing

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
