pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using Documenter
using PureRNGs

DocMeta.setdocmeta!(PureRNGs, :DocTestSetup, :(using PureRNGs); recursive = true)

makedocs(
    sitename = "PureRNGs.jl",
    modules = [PureRNGs],
    remotes = nothing,
    doctest = true,
    linkcheck = true,
    checkdocs = :exports,
    warnonly = false,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Immutable workflows" => "tutorials/immutable-workflows.md",
            "Splitting and devices" => "tutorials/splitting-and-devices.md",
            "Sampling" => "tutorials/sampling.md",
            "Stateful interoperability" => "tutorials/stateful-interop.md",
        ],
        "Guides" => [
            "Device binding" => "guides/devices.md",
            "Splitting keys" => "guides/splitting.md",
            "Reproducibility" => "guides/reproducibility.md",
        ],
        "API reference" => "api.md",
    ],
)
