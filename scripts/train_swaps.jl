using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")

#]add Flux, Distributions, Dates, DLProteinFormats, LearningSchedules, CannotWaitForTheseOptimisers, JLD2, CUDA, cuDNN
#]add Revise

using BranchChain
using Flux, Distributions, Dates
using DLProteinFormats: load, CHAIN_FEATS_64, PDBSimpleFlatV2, PDBAtom14, PDBClusters, PDBTable, sample_batched_inds, length2batch, featurizer, CHAIN_FEATS_V1, broadcast_features, pdbid_clean
using LearningSchedules
using CannotWaitForTheseOptimisers: Muon
using JLD2: jldsave
import JLD2

ENV["CUDA_VISIBLE_DEVICES"] = 1
#ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "80GiB"
using CUDA, cuDNN

device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum

device = gpu

X0_mean_length = 0
deletion_pad = 1.1
per_chain_upper_X0_len = 1 + quantile(Poisson(X0_mean_length), 0.95)

rundir = "runs/branchchain_swaps_$(Date(now()))_$(rand(100000:999999))"
mkpath("$(rundir)/samples")
mkpath("$(rundir)/vids")

#olddat = load(PDBSimpleFlatV2_500);
#dat = load(PDBAtom14);
dat = JLD2.load("/home/murrellb/sidechains/pdb-atom14.jld2")["data"];

sidechain_model = JLD2.load("/home/murrellb/BranchChain.jl/sandbox/sidechain_SWAE_dim8_model256_ns0.1.jld")["model_state"] |> device
cpu_decoder = cpu(sidechain_model.decoder)

#@time latent_side_chains = [cpu(sidechain_model.encoder(device(reshape(local_atom14(dat[i].locs, dat[i].rots, dat[i].atom14_coords[:,4:end,:]), 33, :)))) for i in 1:length(dat)];
#jldsave("latent_side_chains.jld", latent_side_chains = latent_side_chains);
latent_side_chains = JLD2.load("latent_side_chains.jld")["latent_side_chains"];


feature_table = load(PDBTable);
pdb_clusters = load(PDBClusters);

train_ff = featurizer(feature_table, CHAIN_FEATS_64, all_mask_prob = 0.05, feat_mask_prob = Beta(1,5))
sampling_ff = featurizer(feature_table, CHAIN_FEATS_64)
clusters = [get(pdb_clusters,c,0) for c in pdbid_clean.(dat.name)] #Zero fallback BEACAUSE SOMEONE NEEDS TO UPDATE THE CLUSTER TABLE

#To prevent OOM, we now need to factor in that some low-t samples might have more elements than their X1 lengths:
len_lbs = max.(length.(dat.AAs), length.(union.(dat.chainids)) .* per_chain_upper_X0_len) .* deletion_pad

oldmodel = load_model("branchchain_feat64_30tune.jld");
model = BranchChainLS2(merge(BranchChain.BranchChainLS2(384, 6, 6, 256).layers, oldmodel.layers)) |> device;

#Minimal effect upon init:
for l in model.layers.sides_to_main
    l.weight ./= 5;
end
for l in model.layers.main_to_sides
    l.weight ./= 5;
end
model.layers.side_chain_cond.weight ./= 5;
for l in model.layers.side_chain_ipa_blocks
    l.ff.w2.weight ./= 6;
    l.ipa.layers.ipa_linear.weight ./= 6;
end
model.layers.side_chain_embedder.weight ./= 3;
model.layers.side_chain_rff_embedder.weight ./= 3;
model.layers.side_chain_decoder.weight ./= 3;
model.layers.sc_side_chain_embedder.weight ./= 3;


#=
template = (BranchChain.pdb"7F5H"1)[[1,2]]
to_redesign = [template[2].sequence]
design(model, X1_from_pdb(template, to_redesign), template.name, [t.id for t in template], sampling_ff; vidpath = "testvid", device = device, path = "test.pdb")
=#

sched = burnin_learning_schedule(0.000001f0, 0.00020f0, 1.05f0, 0.999995f0);
opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> any(size(x) .== 21)), model);

Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing);

struct BatchDataset{T}
    batchinds::T
end
Base.length(x::BatchDataset) = length(x.batchinds)
Base.getindex(x::BatchDataset, i) = training_prep(x.batchinds[i], dat, deletion_pad, X0_mean_length, train_ff, latent_side_chains)
function batchloader(; device=identity, parallel=true)
    uncapped_l2b = length2batch(1500, 1.25)
    batchinds = sample_batched_inds(len_lbs, clusters, l2b = x -> min(uncapped_l2b(x), 100))
    @show length(batchinds)
    x = BatchDataset(batchinds)
    dataloader = Flux.DataLoader(x; batchsize=-1, parallel)
    return device(dataloader)
end

#=
ts = training_prep([1], dat, deletion_pad, X0_mean_length, train_ff, latent_side_chains) |> device;
sc_frames = nothing
ts.t
frames, aa_logits, atom14, count_log, del_logit = model(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, ts.chain_features, sc_frames = sc_frames)
=#

starttime = now()
textlog("$(rundir)/log.csv", ["epoch", "batch", "learning rate", "loss"])
for epoch in 1:6 #Test run
    if epoch == 5
        sched = linear_decay_schedule(sched.lr, 0.000000001f0, 5600) 
        #sched = linear_decay_schedule(sched.lr, 0.000000001f0, 2800)  #Single epoch
    end
    for (i, ts) in enumerate(batchloader(; device = device))
        sc = nothing
        for _ in 1:rand(Poisson(1))
            sc_frames, _, sc_sidechains, _, _ = model(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, ts.chain_features, sc_frames = sc)
            sc = (;frames = sc_frames, sidechains = sc_sidechains)
        end
        @show ts.t
        l, grad = Flux.withgradient(model) do m
            frames, aa_logits, atom14, count_log, del_logit = m(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, ts.chain_features, sc_frames = sc)
            l_loc, l_rot, l_aas, l_atom14, l_splits, l_del = losses(BranchChain.P, (frames, aa_logits, atom14, count_log, del_logit), ts)
            @show l_loc, l_rot, l_aas, l_atom14, l_splits, l_del
            l_loc + l_rot + l_aas + l_atom14 + l_splits + l_del
        end
        Flux.update!(opt_state, model, grad[1])
        (mod(i, 10) == 0) && Flux.adjust!(opt_state, next_rate(sched))
        if mod(i,50) == 0
            println("\n\nMean time per 50 batches: $((now() - starttime))\n\n") #6000 Ada: ~53472 milliseconds
            starttime = now()
        end
        textlog("$(rundir)/log.csv", [epoch, i, sched.lr, l])
        if mod(i, 5000) == 500
            for v in 1:3
                try
                    sampname = "e$(epoch)_b$(i)_samp$(v)"    
                    vidpath = "$(rundir)/vids/$(sampname)"
                    feature_table[feature_table.pdb_id .== "7F5H",:]
                    template = compoundstate(merge(dat[172742], (;latent_sc = latent_side_chains[172742])), mask_override = dat[172742].chainids .== 2)
                    design(model, template[1], template[3], template[4], sampling_ff; vidpath = vidpath, device = device, path = "$(rundir)/samples/$(sampname).pdb", side_chain_decoder = cpu_decoder)
                catch
                    println("Error in design sample for samp $v")
                end
            end
            jldsave("$(rundir)/model_epoch_$(epoch)_current.jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
        end
    end
    jldsave("$(rundir)/model_epoch_$(epoch).jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end

#BranchChain.jldsave("$(rundir)/branchchain_AA.jld", model_state = cpu(model));


#template = compoundstate(dat[172742], mask_override = dat[172742].chainids .== 2)
#design(model, template[1], template[3], template[4], sampling_ff; vidpath = "testvid", device = device, path = "test.pdb")
#design(model, template[1], template[3], template[4], sampling_ff; vidpath = vidpath, device = device, path = "$(rundir)/samples/$(sampname).pdb")



sampname = "test_samp1"    
vidpath = "testvid1"
feature_table[feature_table.pdb_id .== "7F5H",:]
template = compoundstate(merge(dat[172742], (;latent_sc = latent_side_chains[172742])), mask_override = dat[172742].chainids .== 2)
design(model, template[1], template[3], template[4], sampling_ff; 
    vidpath = vidpath, device = device, path = "testviiid.pdb", side_chain_decoder = cpu_decoder)