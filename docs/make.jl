using Documenter
using SlicedWasserstein

DocMeta.setdocmeta!(
    SlicedWasserstein,
    :DocTestSetup,
    :(using SlicedWasserstein);
    recursive = true,
)

makedocs(
    sitename = "SlicedWasserstein.jl",
    modules = [SlicedWasserstein],
    pages = [
        "Home" => "index.md",
        "Background and methodology" => "background.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/gianlucacovini/SlicedWasserstein.jl",
    devbranch = "main",
)
