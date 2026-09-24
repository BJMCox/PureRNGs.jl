using Test
using PureRNGs

include("fixtures.jl")

# The files are included inside one testset so a failure is recorded in the outer set
# instead of thrown. A thrown failure would abort every later file.
@testset "PureRNGs" begin
    include("philox.jl")
    include("threefry.jl")
    include("chacha.jl")
    include("generators.jl")
    include("oracle_conformance.jl")
    include("derive.jl")
    include("bits.jl")
    include("uniform.jl")
    include("uniform_forms.jl")
    include("type_coverage.jl")
    include("integers.jl")
    include("range_fill.jl")
    include("device_base_seams.jl")
    include("extension_metadata.jl")
    include("normal.jl")
    include("exponential.jl")
    include("fill_laws.jl")
    include("custom_axes.jl")
    include("sampling.jl")
    include("collections.jl")
    include("permutations.jl")
    include("stateful.jl")
    include("closed_audits.jl")
    include("weighted_sampling.jl")
    include("accuracy.jl")
    include("statistical_smoke.jl")
    include("errors.jl")
end
