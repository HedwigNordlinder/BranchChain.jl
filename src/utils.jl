"""
    textlog(filepath, l; also_print=true)

Append a comma-separated log line to `filepath`, optionally echoing to stdout.

- `filepath`: path to the text log file (created if needed).
- `l`: collection of values to be converted to strings and joined with commas.
- `also_print`: when `true`, prints the same line to `stdout`.
"""
function textlog(filepath::String, l; also_print = true)
    f = open(filepath,"a")
        write(f, join(string.(l),", "))
        write(f, "\n")
    close(f)
    if also_print
        println(join(string.(l),", "))
    end
end

"""
    X1_modifier(X1)

Post-process a target state so that residues marked for deletion use a dummy AA.

This sets the amino-acid state at indices `X1.del` to 21 (the dummy residue) in
`X1.state[3]`. It is passed into `branching_bridge` so that deleted residues
have consistent terminal states at `t=1`.
"""
function X1_modifier(X1)
    X1.state[3].S.state[X1.del] .= 21
    return X1
end


"""
    training_prep(b, dat, deletion_pad, X0_mean_length)

Prepare a mini-batch for training the branching flow / design model.

For batch indices `b`, samples masked terminal states from `dat`, builds a branching bridge, and
returns a named tuple with all inputs and targets needed by `BranchChainV1`
and `losses`.
"""
function training_prep(b, dat, deletion_pad, X0_mean_length, feature_func; P = P)
    sampled = compoundstate.(dat[b])
    X1s = [s[1] for s in sampled]
    hasnobreaks = [s[2] for s in sampled]
    pdb_ids = [s[3] for s in sampled]
    chain_labels = [s[4] for s in sampled]
    t = Uniform(0f0,1f0)
    bat = branching_bridge(P, X0sampler, X1s, t, 
                            coalescence_factor = 1.0, 
                            use_branching_time_prob = 0.5,
                            merger = BranchingFlows.canonical_anchor_merge,
                            length_mins = Poisson(X0_mean_length),
                            deletion_pad = deletion_pad,
                            X1_modifier = X1_modifier,
                        )

    #Feature code here:
    chain_features = broadcast_features(pdb_ids, chain_labels, bat.Xt.groupings, feature_func)

    rotξ = Guide(bat.Xt.state[2], bat.X1anchor[2])
    resinds = similar(bat.Xt.groupings) .= 1:size(bat.Xt.groupings, 1) #We should consider using native res inds for the residues fixed in conditioning.
    return (;t = bat.t, chainids = bat.Xt.groupings, resinds,
                    Xt = bat.Xt, hasnobreaks = hasnobreaks,
                    rotξ_target = rotξ, X1_locs_target = bat.X1anchor[1], X1aas_target = bat.X1anchor[3],
                    splits_target = bat.splits_target, del = bat.del, chain_features)
end

"""
    step_spec(model; recycles=3, vidpath=nothing, printseq=true,
              device=identity, frameid=[1])

Build a step function compatible with `gen` that calls the design model.

- `model`: a trained `BranchChainV1` (or compatible) instance.
- `recycles`: number of additional model refinement passes using self-conditioning
  (`sc_frames`) before producing the final prediction.
- `vidpath`: optional folder where intermediate `Xt` and `X1hat` PDBs are
  written for visualization. If `nothing`, no PDBs are written.
- `printseq`: when `true`, prints the current amino-acid sequence at each step.
- `device`: function to move inputs onto the desired device (e.g. `gpu`).
- `frameid`: single-element vector holding a mutable frame counter used in
  file naming; typically left at the default.
"""
#Self cond bug? Indexing state not being updated properly?
function step_spec(model::Union{BranchChainV2,BranchChainV3}, pdb_id, chain_labels, feature_func; 
    recycles = 0, vidpath = nothing, printseq = true, device = identity, frameid = [1], feature_override = nothing, transform_array = [])
    sc_frames = nothing
    function mod_wrapper(t, Xₜ; frameid = frameid, recycles = recycles)
        !isnothing(vidpath) && export_pdb(vidpath*"/Xt/$(string(frameid[1], pad = 4)).pdb", Xₜ.state, Xₜ.groupings, collect(1:length(Xₜ.groupings)))
        Xtstate = Xₜ.state
        frominds = Xtstate[4].S.state[:]
        chain_features = broadcast_features([pdb_id], [chain_labels], Xₜ.groupings, (a,b) -> feature_func(a,b,override = feature_override))
        if !isnothing(sc_frames)
            sc_frames = Translation(sc_frames.composed.outer.values[:,:,frominds,:]) ∘ Rotation(sc_frames.composed.inner.values[:,:,frominds,:])
        end
        printseq && println(replace(DLProteinFormats.ints_to_aa(tensor(Xtstate[3])[:]), "X"=>"-"), ":", frameid[1])
        if length(tensor(Xtstate[3])[:]) > 2000
            error("Chain too long")
        end
        resinds = similar(Xₜ.groupings) .= 1:size(Xₜ.groupings, 1)
        input_bundle = ([t]', Xₜ, Xₜ.groupings, resinds, [true], chain_features) |> device
        for _ in 1:recycles
            sc_frames, _ = model(input_bundle..., sc_frames = gpu(sc_frames))
        end
        pred = model(input_bundle..., sc_frames = gpu(sc_frames)) |> cpu
        sc_frames = deepcopy(pred[1])
        state_pred = ContinuousState(values(translation(pred[1]))), ManifoldState(rotM, eachslice(cpu(values(linear(pred[1]))), dims=(3,4))), pred[2], nothing
        !isnothing(vidpath) && export_pdb(vidpath*"/X1hat/$(string(frameid[1], pad = 4)).pdb", (state_pred[1], state_pred[2], Xₜ.state[3]), Xₜ.groupings, collect(1:length(Xₜ.groupings)))
        frameid[1] += 1
        Xtstate[4].S.state .= 1:length(Xtstate[4].S.state) #<-Update the indexing state? Not being used.
        return state_pred, pred[3], pred[4]
    end
    return mod_wrapper
end

#Alternative: Define "matched state" which holds on to a transform_array.
#step will take the first part for some elements (eg. the discrete ones) but maintain the full state space for the others.
#The symmetry will be enfoced at the "step" for the discrete components, but the "prediction" level for the continuous ones.

#Need to think about how to handle both conditional and unconditional models more gracefully than a code-duplicated dispatch.
function step_spec(model::BranchChainV1, pdb_id, chain_labels, feature_func; transform_array = [], recycles = 0, vidpath = nothing, printseq = true, device = identity, frameid = [1], feature_override = nothing)
    sc_frames = nothing
    expanded_sc_frames = nothing
    function mod_wrapper(t, Xₜ; frameid = frameid, recycles = recycles)
        Xtstate = Xₜ.state
        frominds = Xtstate[4].S.state[:]
        orig_l = length(frominds)
        if !isnothing(sc_frames) #First expand the split positions.
            sc_frames = Translation(sc_frames.composed.outer.values[:,:,frominds,:]) ∘ Rotation(sc_frames.composed.inner.values[:,:,frominds,:])
        end
        if length(tensor(Xtstate[3])[:]) > 2000
            error("Chain too long")
        end
        Xₜ = expand_state(Xₜ, transform_array)
        !isnothing(vidpath) && export_pdb(vidpath*"/Xt/$(string(frameid[1], pad = 4)).pdb", Xₜ.state, Xₜ.groupings, collect(1:length(Xₜ.groupings)))
        printseq && println(replace(DLProteinFormats.ints_to_aa(tensor(Xₜ.state[3])[:]), "X"=>"-"), ":", frameid[1])
        if !isnothing(sc_frames)
            expanded_sc_frames = expand_frames(sc_frames, transform_array)
        end
        resinds = similar(Xₜ.groupings) .= 1:size(Xₜ.groupings, 1)
        input_bundle = ([t]', Xₜ, Xₜ.groupings, resinds, [true]) |> device
        @show Xₜ.groupings, resinds
        for _ in 1:recycles
            expanded_sc_frames, _ = model(input_bundle..., sc_frames = gpu(expanded_sc_frames))
            expanded_sc_frames = cpu(expanded_sc_frames)
            sc_frames = Translation(expanded_sc_frames.composed.outer.values[:,:,1:orig_l,:]) ∘ Rotation(expanded_sc_frames.composed.inner.values[:,:,1:orig_l,:])
            expanded_sc_frames = expand_frames(sc_frames, transform_array) |> gpu
        end
        pred = model(input_bundle..., sc_frames = gpu(expanded_sc_frames)) |> cpu #frames, aa_logits, count_log, del_logits
        sc_frames = deepcopy(Translation(pred[1].composed.outer.values[:,:,1:orig_l,:]) ∘ Rotation(pred[1].composed.inner.values[:,:,1:orig_l,:]))
        
        state_pred = ContinuousState(values(translation(pred[1]))[:,:,1:orig_l,:]), ManifoldState(rotM, eachslice(cpu(values(linear(pred[1]))[:,:,1:orig_l,:]), dims=(3,4))), pred[2][:,1:orig_l,:], nothing
        vid_pred = ContinuousState(values(translation(pred[1]))), ManifoldState(rotM, eachslice(cpu(values(linear(pred[1]))), dims=(3,4))), pred[2], nothing
        !isnothing(vidpath) && export_pdb(vidpath*"/X1hat/$(string(frameid[1], pad = 4)).pdb", (vid_pred[1], vid_pred[2], Xₜ.state[3]), Xₜ.groupings, collect(1:length(Xₜ.groupings)))
        frameid[1] += 1
        Xtstate[4].S.state .= 1:length(Xtstate[4].S.state) #<-Update the indexing state
        return state_pred, pred[3][1:orig_l,:], pred[4][1:orig_l,:]
    end
    return mod_wrapper
end

"""
    random_template(dat; only_sampled_masked=true)

Sample a random masked target `X1` from the dataset to use as a template.

- `dat`: dataset with fields such as `len` and records consumable by
  `compoundstate`.
- `only_sampled_masked`: when `true`, keeps resampling until at least one
  residue is masked in the sampled example (up to 100 tries).
"""
#NEEDS TO PASS THROUGH PDB AND CHAIN IDS
function random_template(dat; only_sampled_masked = true)
    b = rand(findall(dat.len .< 1000))
    sampled = compoundstate.(dat[[b]])
    X1s = [s[1] for s in sampled]
    if only_sampled_masked
        counter = 0
        while length(X1s[1].flowmask) == sum(X1s[1].flowmask)
            println("Resampling because there are no masked chains")
            counter += 1
            b = rand(findall(dat.len .< 1000))
            sampled = compoundstate.(dat[[b]])
            X1s = [s[1] for s in sampled]
            if counter > 100
                println("Failed to sample a masked chain")
                break
            end
        end
    end
    return X1s[1]
end

"""
    X1_from_pdb(pdb_rec, segments_to_mask)

Construct a masked `BranchingState` from a PDB record, masking specific
sequence segments.

- `pdb_rec`: a `ProteinStructure` / PDB record that can be flattened by
  `DLProteinFormats.flatten`.
- `segments_to_mask`: vector of amino-acid substrings. Every exact match of
  each substring in the flattened sequence is masked (designable). An error is
  thrown if any substring is not found.
"""
#Needs to handle/return features:
function X1_from_pdb(pdb_rec, segments_to_mask::Vector{String}; exclude_flatchain_nums = Int[], recenter = false)
    pdb_rec.cluster = 1
    rec = DLProteinFormats.flatten(pdb_rec)
    L = length(rec.AAs)
    #new bit
    flatAA_chars = collect(join(DLProteinFormats.AAs[rec.AAs]))
    for ex in exclude_flatchain_nums
        flatAA_chars[rec.chainids .== ex] .= '!' #Set bits of sequence to be un-matchable
    end
    #end new bit
    flatAAstring = join(flatAA_chars)
    #flatAAstring = join(DLProteinFormats.AAs[rec.AAs])
    cmask = falses(length(flatAAstring))
    for segment in segments_to_mask
        matches = findall(segment, flatAAstring)
        if isempty(matches)
            error("Segment $segment not found in $flatAAstring")
        end
        for match in matches
            cmask[match] .= true
        end
    end
    X1locs = MaskedState(ContinuousState(rec.locs), cmask, cmask)
    if recenter
        X1locs.S.state .-= mean(X1locs.S.state, dims=3)
    end
    X1rots = MaskedState(ManifoldState(rotM,eachslice(rec.rots, dims=3)), cmask, cmask)
    X1aas = MaskedState((DiscreteState(21, rec.AAs)), cmask, cmask)
    index_state = MaskedState((DiscreteState(0, [1:L;])), cmask, cmask)
    X1 = BranchingState((X1locs, X1rots, X1aas, index_state), rec.chainids, flowmask = cmask, branchmask = cmask) #<- .state, .groupings
    return X1
end

X1_from_pdb(pdb_rec; kwargs...) = X1_from_pdb(pdb_rec, [""]; kwargs...)

"""
    design(model, X1;
           steps = step_sched.(0f0:0.005:1f0),
           path = nothing, vidpath = nothing, printseq = true,
           device = identity,
           P = P,
           X0_mean_length = model.layers.config.X0_mean_length_minus_1,
           deletion_pad = model.layers.config.deletion_pad,
           recycles = 3)

Run the branching-flow generative process to design a sequence/structure for a
given masked target `X1`.

- `model`: trained `BranchChainV1` model with a `layers.config` object
  describing the training hyperparameters.
- `X1`: masked terminal `BranchingState` produced by `compoundstate` or
  `X1_from_pdb`.
- `steps`: schedule of scalar times in `[0,1]` used by `gen`.
- `path`: if not `nothing`, write the final designed structure to this PDB path.
- `vidpath`: if not `nothing`, create folders `Xt/` and `X1hat/` under this
  path and write intermediate and final PDBs there.
- `printseq`: when `true`, print the designed sequence at the final step.
- `device`: function used to place the model inputs on the desired device
  (e.g. `gpu`).
- `P`: branching-flow prior; by default uses the global `P` defined in
  `models.jl`.
- `X0_mean_length`: mean initial length prior used by `branching_bridge`.
- `deletion_pad`: padding amount for deletions in the bridge.
- `recycles`: number of self-conditioning recycles used inside `step_spec`.

Returns the final sampled branching state `samp`.
"""
#Needs to consider features/feature modifications.
#This should be generalized to also be allowed to take in features directly. Or something.
function design(model, X1, pdb_id, chain_labels, feature_func; steps = step_sched.(0f0:0.005:1f0), transform_array = [],
                path = nothing, vidpath = nothing, printseq = true, device = identity, feature_override = nothing,
                P = P, X0_mean_length = model.layers.config.X0_mean_length_minus_1, deletion_pad = model.layers.config.deletion_pad, recycles = 0)
    if steps isa Number
        steps = step_sched.(0f0:Float32(1/steps):1f0)
    end
    hasnobreaks = [true]
    t = [0f0]
    bat = branching_bridge(P, X0sampler, [X1], t, 
                            coalescence_factor = 1.0, 
                            use_branching_time_prob = 0.0,
                            merger = BranchingFlows.canonical_anchor_merge,
                            length_mins = Poisson(X0_mean_length),
                            deletion_pad = deletion_pad,
                            X1_modifier = X1_modifier)
    X0 = bat.Xt
    frameid = [1]
    if !isnothing(vidpath)
        mkpath(vidpath*"/Xt")
        mkpath(vidpath*"/X1hat")
    end
    samp = gen(P, X0, step_spec(model, pdb_id, chain_labels, feature_func; vidpath, printseq, device, frameid, recycles, feature_override, transform_array), steps)
    expanded_samp = expand_state(samp, transform_array)
    printseq && println(replace(DLProteinFormats.ints_to_aa(tensor(expanded_samp.state[3])[:]), "X"=>"-"), ":", frameid[1])
    !isnothing(vidpath) && export_pdb(vidpath*"/Xt/$(string(frameid[1], pad = 4)).pdb", expanded_samp.state, expanded_samp.groupings, collect(1:length(expanded_samp.groupings)))
    !isnothing(vidpath) && export_pdb(vidpath*"/X1hat/$(string(frameid[1], pad = 4)).pdb", expanded_samp.state, expanded_samp.groupings, collect(1:length(expanded_samp.groupings)))
    !isnothing(path) && export_pdb(path, expanded_samp.state, expanded_samp.groupings, collect(1:length(expanded_samp.groupings)))
    return samp
end

design(model, X1; kwargs...) = design(model, X1, "", [""], (x...; kwargs...) -> Dict(); kwargs...)

"""
    gen2prot(samp, chainids, resnums; name="Gen")

Convert a sampled state into a `ProteinStructure`.

- `samp`: a tuple `(locs, rots, aas)` as returned by the sampler.
- `chainids`: integer chain IDs for each residue in the flattened representation.
- `resnums`: residue numbers for each residue.
- `name`: protein name to use in the resulting structure.
"""
function gen2prot(samp, chainids, resnums; name = "Gen", )
    d = Dict(zip(0:25,'A':'Z'))
    chain_letters = get.((d,), chainids, 'Z')
    ProteinStructure(name, Atom{eltype(tensor(samp[1]))}[], DLProteinFormats.unflatten(tensor(samp[1]), tensor(samp[2]), tensor(samp[3]), chain_letters, resnums)[1])
 end

"""
    export_pdb(path, samp, chainids, resnums)

Write a sampled state `samp` to a PDB file.

Convenience wrapper around `ProteinChains.writepdb` that routes through
`gen2prot`.
"""
export_pdb(path, samp, chainids, resnums) = ProteinChains.writepdb(path, gen2prot(samp, chainids, resnums))

step_sched(t) = Float32(1-(cos(t*pi)+1)/2)

"""
    load_model(checkpoint)

Load a pretrained model state from the Hugging Face hub.

- `checkpoint`: filename or revision string within
  `"MurrellLab/BFChainStorm"`. Returns the `"model_state"` object from the
  downloaded JLD2 file. Available checkpoints are:
  - "condsegment_v1.jld"
  - "condchain_v1.jld"
"""
function load_model(checkpoint)
    file = hf_hub_download("MurrellLab/BFChainStorm", checkpoint)
    return JLD2.load(file, "model_state")
end

export textlog, X1_modifier, training_prep, mod_wrapper, X1_from_pdb, design, load_model