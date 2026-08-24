using TOML

@testset "R36-R38 extension metadata and base surface" begin
    project = TOML.parsefile(joinpath(pkgdir(PureRNGs), "Project.toml"))

    @test project["weakdeps"] == Dict(
        "AMDGPU" => "21141c5a-9bdb-4563-92ae-f87d6854732e",
        "CUDA" => "052768ef-5323-5732-b1bb-66c8b64840ba",
    )
    @test project["extensions"] ==
          Dict("PureRNGsAMDGPUExt" => "AMDGPU", "PureRNGsCUDAExt" => "CUDA")
    @test project["compat"]["AMDGPU"] == "2"
    @test project["compat"]["CUDA"] == "5.8, 6"

    @test Base.get_extension(PureRNGs, :PureRNGsAMDGPUExt) === nothing
    @test Base.get_extension(PureRNGs, :PureRNGsCUDAExt) === nothing

    @test names(PureRNGs) == [
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
        :randn_next,
        :randn_next!,
        :randnat,
        :splitrng,
        :subrng,
    ]
end
