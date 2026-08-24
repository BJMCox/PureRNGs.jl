using TOML

@testset "CUDA extension metadata and base surface" begin
    project = TOML.parsefile(joinpath(pkgdir(PureRNGs), "Project.toml"))

    @test project["weakdeps"]["CUDA"] == "052768ef-5323-5732-b1bb-66c8b64840ba"
    @test project["extensions"]["PureRNGsCUDAExt"] == "CUDA"
    @test project["compat"]["CUDA"] == "5.8, 6"

    @test Base.get_extension(PureRNGs, :PureRNGsCUDAExt) === nothing

    @test names(PureRNGs) == [
        :Philox2x32,
        :Philox2x64,
        :Philox4x32,
        :Philox4x64,
        :PureRNGs,
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
