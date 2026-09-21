using Test

# The suite is included inside one testset so a failure is recorded in the outer set
# instead of thrown. A thrown failure would abort every later testset in the file.
@testset "distributions" begin
    include("suite.jl")
end
