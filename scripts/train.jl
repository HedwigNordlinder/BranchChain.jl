using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")

using BranchChain
using Flux, Distributions, Dates
using DLProteinFormats: load, CHAIN_FEATS_64, PDBSimpleFlatV2, PDBClusters, PDBTable, sample_batched_inds, length2batch, featurizer, CHAIN_FEATS_V1, broadcast_features, pdbid_clean
using LearningSchedules
using CannotWaitForTheseOptimisers: Muon
using JLD2: jldsave

ENV["CUDA_VISIBLE_DEVICES"] = 0
using CUDA


X0_mean_length = 0
deletion_pad = 1.1
per_chain_upper_X0_len = 1 + quantile(Poisson(X0_mean_length), 0.95)

device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
device = gpu

rundir = "runs/branchchain_featurecond_withsymmetry_$(Date(now()))_$(rand(100000:999999))"
mkpath("$(rundir)/samples")
mkpath("$(rundir)/vids")

dat = load(PDBSimpleFlatV2);
feature_table = load(PDBTable);
pdb_clusters = load(PDBClusters);

train_ff = featurizer(feature_table, CHAIN_FEATS_64)
sampling_ff = featurizer(feature_table, CHAIN_FEATS_64)
clusters = [pdb_clusters[c] for c in pdbid_clean.(dat.name)]

#To prevent OOM, we now need to factor in that some low-t samples might have more elements than their X1 lengths:
len_lbs = max.(dat.len, length.(union.(dat.chainids)) .* per_chain_upper_X0_len) .* deletion_pad

V1layers = load_model("branchchain_tune1.jld").layers;
v3_scaf = BranchChain.BranchChainV3(config = V1layers.config);
model = BranchChain.BranchChainV3(merge(v3_scaf.layers, V1layers)) |> device;
model.layers.feature_embedder.weight ./= 2;

sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.99995f0)
opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> any(size(x) .== 21)), model)
Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

struct BatchDataset{T}
    batchinds::T
end
Base.length(x::BatchDataset) = length(x.batchinds)
Base.getindex(x::BatchDataset, i) = training_prep(x.batchinds[i], dat, deletion_pad, X0_mean_length, train_ff)
function batchloader(; device=identity, parallel=true)
    uncapped_l2b = length2batch(1500, 1.25)
    batchinds = sample_batched_inds(len_lbs, clusters, l2b = x -> min(uncapped_l2b(x), 100))
    @show length(batchinds)
    x = BatchDataset(batchinds)
    dataloader = Flux.DataLoader(x; batchsize=-1, parallel)
    return device(dataloader)
end

Flux.freeze!(opt_state)
Flux.thaw!(opt_state.layers.feature_embedder)

textlog("$(rundir)/log.csv", ["epoch", "batch", "learning rate", "loss"])
for epoch in 1:10
    if epoch == 9
        sched = linear_decay_schedule(sched.lr, 0.000000001f0, 5800) 
    end
    for (i, ts) in enumerate(batchloader(; device = device))
        if epoch == 1 && i == 2000 #<-First 2000 batches only training the feature embedder
            Flux.thaw!(opt_state)
            sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.99995f0)
            break;
        end
        sc_frames = nothing
        for _ in 1:rand(Poisson(1))
            sc_frames, _ = model(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, ts.chain_features, sc_frames = sc_frames)
        end
        l, grad = Flux.withgradient(model) do m
            frames, aa_logits, count_log, del_logit = m(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, ts.chain_features, sc_frames = sc_frames)
            l_loc, l_rot, l_aas, l_splits, l_del = losses(BranchChain.P, (frames, aa_logits, count_log, del_logit), ts)
            @show l_loc, l_rot, l_aas, l_splits, l_del
            l_loc + l_rot + l_aas + l_splits + l_del
        end
        Flux.update!(opt_state, model, grad[1])
        (mod(i, 10) == 0) && Flux.adjust!(opt_state, next_rate(sched))
        textlog("$(rundir)/log.csv", [epoch, i, sched.lr, l])
        if mod(i, 5000) == 1
            for v in 1:3
                try
                    sampname = "e$(epoch)_b$(i)_samp$(v)"    
                    vidpath = "$(rundir)/vids/$(sampname)"
                    feature_table[feature_table.pdb_id .== "7F5H",:]
                    template = (BranchChain.pdb"7F5H"1)[[1,2]]
                    to_redesign = [template[2].sequence]
                    design(model, X1_from_pdb(template, to_redesign), template.name, [t.id for t in template], sampling_ff; vidpath = vidpath, device = device, path = "$(rundir)/samples/$(sampname).pdb")
                catch
                    println("Error in design sample for samp $v")
                end
            end
            jldsave("$(rundir)/model_epoch_$(epoch)_batch_$(i).jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
        end
    end
    jldsave("$(rundir)/model_epoch_$(epoch).jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end

BranchChain.jldsave("$(rundir)/branchchain_feat64.jld", model_state = model);

