
function frame_transform(frames, t)
    l = size(frames.composed.inner.values, 3)
    translate, rotate = t
    tr = repeat(translate, Onion.Einops.einops"a b c d -> a b (c r) d", r=l)
    ro = repeat(rotate, Onion.Einops.einops"a b c d -> a b (c r) d", r=l)
    return (Translation(tr) ∘ Rotation(ro)) ∘ frames
end

function expand_frames(frames, transform_array)
    if isempty(transform_array)
        return frames
    end
    base_L = size(frames.composed.inner.values, 3)
    copies = length(transform_array)
    tr = repeat(frames.composed.outer.values, Onion.Einops.einops"a b c d -> a b (c r) d", r=copies + 1)
    ro = repeat(frames.composed.inner.values, Onion.Einops.einops"a b c d -> a b (c r) d", r=copies + 1)
    for i in 1:copies
        transformed_frames = frame_transform(frames, transform_array[i])
        tr[:, :, (i)*base_L + 1:(i+1)*base_L, :] .= transformed_frames.composed.outer.values
        ro[:, :, (i)*base_L + 1:(i+1)*base_L, :] .= transformed_frames.composed.inner.values
    end
    return Translation(tr) ∘ Rotation(ro)
end

#MUST HANDLE GROUPING OFFSETS CORRECTLY!
function expand_state(bf, transform_array) #Assumes the master copy is the identity.
    if isempty(transform_array)
        return bf
    end
    copies = length(transform_array)
    base_L = size(bf.groupings, 1)
    expanded_L = base_L * (copies + 1)
    groupings = repeat(bf.groupings, Onion.Einops.einops"a b -> (a r) b", r=copies + 1)
    max_groupings = maximum(groupings)
    for i in 1:copies #Must offset groupings so the model doesn't think the chains are shared.
        groupings[i*base_L + 1:(i+1)*base_L, :] .= groupings[1:base_L, :] .+ max_groupings * i
    end
    flowmask = repeat(bf.flowmask, Onion.Einops.einops"a b -> (a r) b", r=copies + 1)
    branchmask = repeat(bf.branchmask, Onion.Einops.einops"a b -> (a r) b", r=copies + 1)
    cmask = repeat(bf.state[1].cmask, Onion.Einops.einops"a b -> (a r) b", r=copies + 1)
    locs = bf.state[1].S.state
    rots = bf.state[2].S.state
    frames = Translation(locs) ∘ Rotation(tensor(rots))
    expanded_frames = expand_frames(frames, transform_array)
    AAs = repeat(bf.state[3].S.state, Onion.Einops.einops"a b -> (a r) b", r=copies + 1)
    loc_state = MaskedState(ContinuousState(expanded_frames.composed.outer.values), cmask, cmask)
    rotStates = eachslice(expanded_frames.composed.inner.values[:,:,:,1], dims=3)
    rot_state = MaskedState(ManifoldState(BranchChain.rotM,reshape(rotStates, size(rotStates)..., 1)), cmask, cmask)
    aas_state = MaskedState((DiscreteState(21, AAs)), cmask, cmask)
    index_state = MaskedState((DiscreteState(0, [1:expanded_L;])), cmask, cmask)
    expanded_bf = BranchingState((loc_state, rot_state, aas_state, index_state), groupings, flowmask = cmask, branchmask = branchmask) #<- .state, .groupings
end


