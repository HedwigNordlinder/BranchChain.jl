using Pkg
Registry.add(url="https://github.com/MurrellGroup/MurrellGroupRegistry")
Pkg.activate(temp=true)
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.add(["DLProteinFormats", "CUDA", "cuDNN"])

ENV["CUDA_VISIBLE_DEVICES"] = 0
using CUDA, cuDNN
device!(0) # <- pick your GPU (zero indexed)

using BranchChain
dev = BranchChain.gpu

model = load_model("branchchain_feat64.jld") |> devi;

outdir = "conditional_sampling/"
mkpath(outdir)

feature_table = DLProteinFormats.load(PDBTable);
sampling_ff = featurizer(feature_table, DLProteinFormats.CHAIN_FEATS_64)

#Design a single chain RBD binder with nanobody-like features (inherited from the same PDB):
template = (BranchChain.pdb"7F5H"1)[[1,2]]
samp = design(model, X1_from_pdb(template, [template[2].sequence]), template.name, [t.id for t in template], sampling_ff;
        device = devi, recycles = 1, steps = 200, vidpath = "$(outdir)/rbdbinder_vid", path = "$(outdir)/rbdbindervid.pdb")

#RBD binder with SH3 features:
feature_table[feature_table.pdb_id .== "7F5H",:] #RBD
feature_table[feature_table.pdb_id .== "1SHG",:] #SH3
template = (BranchChain.pdb"7F5H"1)[[1,2]]
design(model, X1_from_pdb(template, [template[2].sequence]), ["7F5H","1SHG"], ["A","A"], #<-Note: These must match the order of the chains in the template!
        sampling_ff; recycles = 1, steps = 250, vidpath = "$(outdir)/RBDSH3cond_vid", device = devi, path = "$(outdir)/RBDSH3cond.pdb")

#RBD binder with Darpin features:
feature_table[feature_table.pdb_id .== "7F5H",:] #RBD
feature_table[feature_table.pdb_id .== "5OOU",:] #Darpin
template = (BranchChain.pdb"7F5H"1)[[1,2]]
design(model, X1_from_pdb(template, [template[2].sequence]), ["7F5H","5OOU"], ["A","A"],
        sampling_ff; recycles = 1, steps = 250, vidpath = "$(outdir)/RBDdarpin_vid", device = devi, path = "$(outdir)/RBDdarpin.pdb")

#Scaffold a 3GBP binder on the FGDF motif, with nanobody-like features
feature_table[feature_table.pdb_id .== "5FW5",:] #G3BP
feature_table[feature_table.pdb_id .== "7OAO",:] #Nanobody
template = (BranchChain.pdb"5FW5"1)[[3,1]]
design_template = X1_from_pdb(template)
design_template.flowmask[1:1] .= true
design_template.flowmask[7:length(template[1].sequence)] .= true
design(model, design_template, ["7OAO","5FW5"], ["FFF","A"], 
        sampling_ff; recycles = 1, steps = 250, vidpath = "$(outdir)/G3BPnanobody_vid", device = devi, path = "$(outdir)/G3BPnanobody.pdb")

#Scaffold a 3GBP binder on the FGDF motif, with minibinder-like features:
feature_table[feature_table.pdb_id .== "5FW5",:] #G3BP
feature_table[feature_table.pdb_id .== "8Y9B",:] #Minibinder
template = (BranchChain.pdb"5FW5"1)[[3,1]]
design_template = X1_from_pdb(template)
design_template.flowmask[1:2] .= true
design_template.flowmask[7:length(template[1].sequence)] .= true
design(model, design_template, ["8Y9B","5FW5"], ["A","A"],
        sampling_ff; recycles = 1, steps = 250, vidpath = "$(outdir)/G3BPminibinder_vid", device = devi, path = "$(outdir)/G3BPminibinder.pdb")

#Extend the small nanoluc fragment:
feature_table[feature_table.pdb_id .== "7SNX",:]
template = (BranchChain.pdb"7SNX"1)[[1,2]]
design_template = X1_from_pdb(template)
design_template.flowmask[findfirst(design_template.groupings .== 2)] = true
design_template.flowmask[findlast(design_template.groupings .== 2)] = true
fo = Dict(["B" => Dict(["length" => 150,"sheet_proportion" => 0.7])]) #Make it longer.
design(model, design_template, template.name, [t.id for t in template], recycles = 1,
        sampling_ff; feature_override = fo, vidpath = "$(outdir)/nanolucextend_vid", device = devi, path = "$(outdir)/nanolucextend.pdb")

#Extend the small nanoluc fragment, but give it nanobody-like features:
feature_table[feature_table.pdb_id .== "7SNX",:]
template = (BranchChain.pdb"7SNX"1)[[1,2]]
design_template = X1_from_pdb(template)
design_template.flowmask[findfirst(design_template.groupings .== 2)] = true
design_template.flowmask[findlast(design_template.groupings .== 2)] = true
fo = Dict(["FFF" => Dict(["length" => 150])]) #Make it a little longer to accomodate the sm-bit
design(model, design_template, ["7SNX","7OAO"], ["A","FFF"], recycles = 1, steps = 250,
        sampling_ff; feature_override = fo, vidpath = "$(outdir)/nanolucnanobody_vid", device = devi, path = "$(outdir)/nanolucnanobody.pdb")
