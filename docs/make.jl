using BranchChain
using Documenter

DocMeta.setdocmeta!(BranchChain, :DocTestSetup, :(using BranchChain); recursive=true)

makedocs(;
    modules=[BranchChain],
    authors="Ben Murrell",
    sitename="BranchChain.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/BranchChain.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/BranchChain.jl",
    devbranch="main",
)
