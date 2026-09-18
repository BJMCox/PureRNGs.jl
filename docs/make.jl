pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using Documenter
using PureRNGs
using Random

DocMeta.setdocmeta!(PureRNGs, :DocTestSetup, :(using PureRNGs); recursive = true)

makedocs(
    sitename = "PureRNGs.jl",
    modules = [PureRNGs],
    remotes = nothing,
    doctest = true,
    linkcheck = true,
    checkdocs = :exports,
    warnonly = [:linkcheck],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/BJMCox/PureRNGs.jl",
        canonical = "https://bjmcox.github.io/PureRNGs.jl/",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Manual" => [
            "Generators and streams" => "manual/generators.md",
            "Arrays and performance" => "manual/arrays.md",
            "Sampling" => "manual/sampling.md",
            "Devices" => "manual/devices.md",
            "Reproducibility" => "manual/reproducibility.md",
        ],
        "Tutorials" => [
            "Parallel jobs" => "tutorials/parallel.md",
            "GPU kernels" => "tutorials/cuda.md",
        ],
        "Integrations" => [
            "Random" => "integrations/random.md",
            "Distributions" => "integrations/distributions.md",
            "Enzyme and Reactant" => "integrations/compilation.md",
        ],
        "API reference" => "api.md",
    ],
)
