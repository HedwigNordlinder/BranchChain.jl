#=Self conditioning strategy:
1) During training, half the samples will have duplicated residues anyway (because we sample t from the coalescence events 50% of the time). Deletions are not handled though.
2) During inference, we'll take split residues and dup them (and remove deleted residues - this is risky).
3) We can always do self-cond and recycling if this misbehaves? Maybe recycling on deletions only? That should be only be ~10% of steps.
=#

#We'll use this to track index changes from splits
struct NullProcess <: Flowfusion.Process end
Flowfusion.endpoint_conditioned_sample(Xa, Xc, p::NullProcess, t_a, t_b, t_c) = Xa
Flowfusion.step(P::NullProcess, Xₜ::Flowfusion.MaskedState, X1targets, s₁, s₂) = Xₜ


oldAdaLN(dim,cond_dim) = AdaLN(Flux.LayerNorm(dim), Dense(cond_dim, dim), Dense(cond_dim, dim))
ipa(l, f, x, pf, c, m) = l(f, x, pair_feats = pf, cond = c, mask = m)
crossipa(l, f1, f2, x, pf, c, m) = l(f1, f2, x, pair_feats = pf, cond = c, mask = m)

"""
    BranchChainV1(; dim=384, depth=6, f_depth=6, config=nothing)

Invariant-point-attention protein design network used inside the branching flow.

The model takes masked backbone+sequence states together with a scalar time
input and predicts updated backbone frames, amino-acid logits, and
insertion/deletion logits. The designable region may change size over the flow
through the branching process.

- `dim`: embedding width for all single-residue representations.
- `depth`: number of main IPA blocks (backbone+sequence transformer layers).
- `f_depth`: number of late layers that are allowed to update the rigid frames
  via `Framemover`.
- `config`: optional configuration object; stored on `model.layers.config` and
  passed through but not interpreted here.

The call overload

```julia
(m::BranchChainV1)(t, BSXt, chainids, resinds, breaks; sc_frames=nothing)
```

is what is actually used during training/sampling. It expects a batched time
tensor `t`, a `BranchingState` `BSXt` containing current locations, rotations
and amino-acid identities, residue and chain indices, and a Boolean break mask
indicating chain discontinuities. If `sc_frames` (self-conditioning frames) are
provided, they are used as an additional template for IPA layers.
"""
struct BranchChainV1{L}
    layers::L
end
Flux.@layer BranchChainV1
function BranchChainV1(dim::Int = 384, depth::Int = 6, f_depth::Int = 6; config = nothing)
    layers = (;
        config = config, #Suggestion: make these so you can extract them with `eval(Meta.parse(model.layers.config.X))`
        depth = depth,
        f_depth = f_depth,
        mask_embedder = Embedding(2 => dim),
        break_embedder = Embedding(2 => dim),
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias=false),
        AApre_t_encoding = Dense(dim => dim, bias=false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias=false),
        AA_embedder = Embedding(21 => dim),
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = oldAdaLN(dim, dim), ln2 = oldAdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
        indelpre_t_encoding = Dense(dim => 3dim),
        count_decoder = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish),
        del_decoder   = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish)
    )
    return BranchChainV1(layers)
end
function (fc::BranchChainV1)(t, BSXt, chainids, resinds, breaks; sc_frames = nothing)
    l = fc.layers
    Xt = BSXt.state
    cmask = BSXt.flowmask
    pmask = Flux.Zygote.@ignore self_att_padding_mask(BSXt.padmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))
    x = l.AA_embedder(tensor(Xt[3])) .+ l.mask_embedder(cmask .+ 1) .+ reshape(l.break_embedder(breaks .+ 1), :, 1, size(t,2))
    x_a,x_b,x_c = nothing, nothing, nothing
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, pmask)
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = 1 .- (1 .- t .* 0.95f0).*cmask)
        end
        if i==4 (x_a = x) end
        if i==5 (x_b = x) end
        if i==6 (x_c = x) end
    end
    aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t,2)))
    catted = vcat(x_a,x_b,x_c)
    indel_pre_t = reshape(l.indelpre_t_encoding(t_rff), :, 1, size(t,2))
    count_log = reshape(l.count_decoder(catted .+ indel_pre_t, false),  :, length(t))
    del_logits = reshape(l.del_decoder(catted .+ indel_pre_t, false),  :, length(t))
    return frames, aa_logits, count_log, del_logits
end


struct BranchChainV2{L}
    layers::L
end
Flux.@layer BranchChainV2
function BranchChainV2(dim::Int = 384, depth::Int = 6, f_depth::Int = 6; config = nothing)
    layers = (;
        config = config, #Suggestion: make these so you can extract them with `eval(Meta.parse(model.layers.config.X))`
        depth = depth,
        f_depth = f_depth,
        mask_embedder = Embedding(2 => dim),
        break_embedder = Embedding(2 => dim),
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias=false),
        AApre_t_encoding = Dense(dim => dim, bias=false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias=false),
        AA_embedder = Embedding(21 => dim),
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = oldAdaLN(dim, dim), ln2 = oldAdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
        indelpre_t_encoding = Dense(dim => 3dim),
        count_decoder = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish),
        del_decoder   = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish),
        feature_embedder = StarGLU(Dense(96 => 2dim, bias=false), Dense(2dim => dim, bias=false), Dense(96 => 2dim, bias=false), Flux.swish) #New thing
    )
    return BranchChainV2(layers)
end
function (fc::BranchChainV2)(t, BSXt, chainids, resinds, breaks, chain_features; sc_frames = nothing)
    l = fc.layers
    Xt = BSXt.state
    cmask = BSXt.flowmask
    pmask = Flux.Zygote.@ignore self_att_padding_mask(BSXt.padmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))    
    x = l.AA_embedder(tensor(Xt[3])) .+ l.mask_embedder(cmask .+ 1) .+ reshape(l.break_embedder(breaks .+ 1), :, 1, size(t,2)) .+ l.feature_embedder(chain_features .+ 0, false)
    x_a,x_b,x_c = nothing, nothing, nothing
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, pmask)
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = 1 .- (1 .- t .* 0.95f0).*cmask)
        end
        if i==4 (x_a = x) end
        if i==5 (x_b = x) end
        if i==6 (x_c = x) end
    end
    aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t,2)))
    catted = vcat(x_a,x_b,x_c)
    indel_pre_t = reshape(l.indelpre_t_encoding(t_rff), :, 1, size(t,2))
    count_log = reshape(l.count_decoder(catted .+ indel_pre_t, false),  :, length(t))
    del_logits = reshape(l.del_decoder(catted .+ indel_pre_t, false),  :, length(t))
    return frames, aa_logits, count_log, del_logits
end


struct BranchChainV3{L}
    layers::L
end
Flux.@layer BranchChainV3
function BranchChainV3(dim::Int = 384, depth::Int = 6, f_depth::Int = 6; config = nothing)
    layers = (;
        config = config, #Suggestion: make these so you can extract them with `eval(Meta.parse(model.layers.config.X))`
        depth = depth,
        f_depth = f_depth,
        mask_embedder = Embedding(2 => dim),
        break_embedder = Embedding(2 => dim),
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias=false),
        AApre_t_encoding = Dense(dim => dim, bias=false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias=false),
        AA_embedder = Embedding(21 => dim),
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = oldAdaLN(dim, dim), ln2 = oldAdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
        indelpre_t_encoding = Dense(dim => 3dim),
        count_decoder = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish),
        del_decoder   = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish),
        feature_embedder = Dense(64 => dim)
    )
    return BranchChainV3(layers)
end
function (fc::BranchChainV3)(t, BSXt, chainids, resinds, breaks, chain_features; sc_frames = nothing)
    l = fc.layers
    Xt = BSXt.state
    cmask = BSXt.flowmask
    pmask = Flux.Zygote.@ignore self_att_padding_mask(BSXt.padmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))    
    x = l.AA_embedder(tensor(Xt[3])) .+ l.mask_embedder(cmask .+ 1) .+ reshape(l.break_embedder(breaks .+ 1), :, 1, size(t,2)) .+ l.feature_embedder(chain_features .+ 0)
    x_a,x_b,x_c = nothing, nothing, nothing
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, pmask)
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = 1 .- (1 .- t .* 0.95f0).*cmask)
        end
        if i==4 (x_a = x) end
        if i==5 (x_b = x) end
        if i==6 (x_c = x) end
    end
    aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t,2)))
    catted = vcat(x_a,x_b,x_c)
    indel_pre_t = reshape(l.indelpre_t_encoding(t_rff), :, 1, size(t,2))
    count_log = reshape(l.count_decoder(catted .+ indel_pre_t, false),  :, length(t))
    del_logits = reshape(l.del_decoder(catted .+ indel_pre_t, false),  :, length(t))
    return frames, aa_logits, count_log, del_logits
end



P = CoalescentFlow((OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0), 
                     ManifoldProcess(OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0)), 
                     DistNoisyInterpolatingDiscreteFlow(D1=Beta(3.0,1.5)),
                     NullProcess()), 
                    Beta(1,2))

const rotM = Flowfusion.Rotations(3)

"""
    X0sampler(root)

Sample a random initial branching state `X₀` from the prior.
The returned tuple contains random Cartesian coordinates,
random SO(3) rotations, and a single dummy amino-acid state (index 21).
"""
X0sampler(root) = (ContinuousState(randn(Float32, 3, 1, 1)), 
                    ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, 1)), 1)),
                    DiscreteState(21, [21]),
                    DiscreteState(0, [1]) #<-INDEXING STATE.
)

"""
    compoundstate(rec)

Convert a flattened `DLProteinFormats` record into a masked `BranchingState`.

This chooses a random design mask (via `rand_mask` and helpers in
`design_mask.jl`), builds masked states for locations, rotations and
amino-acid identities, and wraps them in a `BranchingState` together with the
chain grouping information.

Returns `(X1, breaks)`, where:

- `X1`: target state with Boolean `flowmask`/`branchmask` indicating which
  residues are currently designable.
- `breaks`: Boolean indicating whether there are any sequence breaks inside
  the masked region (used as an additional conditioning signal).
"""
#NEEDS TO RETURN PDB AND CHAIN LABELS
function compoundstate(rec)
    L = length(rec.AAs)
    cmask = rand_mask(rec.chainids)
    breaks = nobreaks(rec.resinds, rec.chainids, cmask)
    X1locs = MaskedState(ContinuousState(rec.locs), cmask, cmask)
    X1rots = MaskedState(ManifoldState(rotM,eachslice(rec.rots, dims=3)), cmask, cmask)
    X1aas = MaskedState((DiscreteState(21, rec.AAs)), cmask, cmask)
    index_state = MaskedState((DiscreteState(0, [1:L;])), cmask, cmask)
    X1 = BranchingState((X1locs, X1rots, X1aas, index_state), rec.chainids, flowmask = cmask, branchmask = cmask) #<- .state, .groupings
    return X1, breaks, DLProteinFormats.pdbid_clean(rec.name), rec.chain_labels
end

"""
    losses(P, X1hat, ts)

Compute the per-component training losses for a single branching-flow step.

# Arguments
- `P`: process collection, typically the global `P` defined internally.
- `X1hat`: model predictions, as a tuple
  `(frames, aa_logits, split_logits, del_logits)` from `BranchChainV1`.
- `ts`: training batch struct returned by `training_prep`, containing targets
  and masks.

Returns a 5-tuple `(l_loc, l_rot, l_aas, splits_loss, del_loss)` with weighted
losses for locations, rotations, amino-acid identities, split counts, and
deletion decisions respectively.
"""
function losses(P, X1hat, ts)
    hat_frames, hat_aas, hat_splits, hat_del = X1hat
    rotangent = Flowfusion.so3_tangent_coordinates_stack(values(linear(hat_frames)), tensor(ts.Xt.state[2]))
    hat_loc, hat_rot, hat_aas = (values(translation(hat_frames)), rotangent, hat_aas)
    l_loc = floss(P.P[1], hat_loc, ts.X1_locs_target,                scalefloss(P.P[1], ts.t, 1, 0.2f0)) * 20
    l_rot = floss(P.P[2], hat_rot, ts.rotξ_target,                   scalefloss(P.P[2], ts.t, 1, 0.2f0)) * 2
    l_aas = floss(P.P[3], hat_aas, onehot(ts.X1aas_target),          scalefloss(P.P[3], ts.t, 1, 0.2f0)) / 10
    splits_loss = floss(P, hat_splits, ts.splits_target, ts.Xt.padmask .* ts.Xt.branchmask, scalefloss(P, ts.t, 1, 0.2f0))
    del_loss = floss(P.deletion_policy, hat_del, ts.del, ts.Xt.padmask .* ts.Xt.branchmask, scalefloss(P, ts.t, 1, 0.2f0))
    return l_loc, l_rot, l_aas, splits_loss, del_loss
end


export BranchChainV1, losses, X0sampler, compoundstate