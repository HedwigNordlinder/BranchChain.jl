module BranchChain

#add DLProteinFormats, Flux, CannotWaitForTheseOptimisers, LearningSchedules, JLD2, Dates, BatchedTransformations, ProteinChains, Flowfusion, Distributions, ForwardBackward, RandomFeatureMaps, InvariantPointAttention, Onion, StatsBase, Random
#add https://github.com/MurrellGroup/BranchingFlows.jl

using DLProteinFormats, Flux, CannotWaitForTheseOptimisers, LearningSchedules, JLD2, Dates, BatchedTransformations, ProteinChains, HuggingFaceApi
using BranchingFlows, Flowfusion, Distributions, ForwardBackward, RandomFeatureMaps, InvariantPointAttention, Onion, StatsBase, Random
using DLProteinFormats: load, PDBSimpleFlat, batch_flatrecs, sample_batched_inds, length2batch

include("models.jl")
include("design_mask.jl")
include("utils.jl")
include("sym.jl")

end