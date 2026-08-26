using Documenter
using PureRNGs

DocMeta.setdocmeta!(
    PureRNGs,
    :DocTestSetup,
    :(using PureRNGs);
    recursive = true,
)

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
        "API reference" => "api.md",
    ],
)
