using Documenter, RiskMeasures

makedocs(sitename="RiskMeasures.jl",
         modules = [RiskMeasures],
         format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
         pages = ["index.md"]
         )

deploydocs(;
    repo = "github.com/RiskAverseRL/RiskMeasures.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"],
    push_preview=true,
)
