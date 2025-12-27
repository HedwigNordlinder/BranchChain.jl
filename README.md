# BranchChain

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/BranchChain.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/BranchChain.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/BranchChain.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/BranchChain.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/BranchChain.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/BranchChain.jl)

## Demo

Set up BranchChain with CUDA in a temporary environment and load the model:

```julia
using Pkg
Registry.add(url="https://github.com/MurrellGroup/MurrellGroupRegistry")
Pkg.activate(temp=true)
Pkg.add(url="https://github.com/MurrellGroup/BranchChain.jl")
Pkg.add(["DLProteinFormats", "CUDA", "cuDNN"])

ENV["CUDA_VISIBLE_DEVICES"] = 0
using CUDA, cuDNN

using BranchChain
dev = BranchChain.gpu

model = load_model("branchchain_tune1.jld") |> dev;
```

### Sampling

Redesign the whole second chain:

```julia
template = (BranchChain.pdb"7F5H"1)[[1,2]]
to_redesign = [template[2].sequence]
samp = design(model, X1_from_pdb(template, to_redesign), steps = 200, device = dev, path = "samp-1.pdb", vidpath = "samp-1-anim");
```

Redesign a portion of the second chain:

```julia
template = (BranchChain.pdb"7F5H"1)[[1,2]]
to_redesign = ["QVQLQESGGGLVQAGGSLRLSCAASGSDFSSSTMGWYRQAPGKQREFVAISSEGSTSYAGSVKGRFTISRDNAKNTVYLQMNSLEPED","QVTVSA"]
samp = design(model, X1_from_pdb(template, to_redesign), steps = 200, device = dev, path = "samp-2.pdb", vidpath = "samp-2-anim");
```

### Conditional sampling

```julia
using DLProteinFormats

feature_table = DLProteinFormats.load(PDBTable);
sampling_ff = featurizer(feature_table, DLProteinFormats.CHAIN_FEATS_V1)

# Design a single chain RBD binder with nanobody-like features:
feature_table[feature_table.pdb_id .== "7F5H", :]
template = (BranchChain.pdb"7F5H"1)[[1,2]]
to_redesign = [template[2].sequence]
samp = design(model, X1_from_pdb(template, to_redesign), template.name, [t.id for t in template], sampling_ff;
        device = dev, recycles = 1, steps = 200, path = "samp-feat.pdb", vidpath = "samp-feat-anim")
```
