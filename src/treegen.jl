# ─── DTS* Tree Search for BranchChain ──────────────────────────────────────────
#
# Discrete Tree Search with soft Bellman backups, ported from ChainStorm.jl
# to work with BranchChain's variable-length branching flow (CoalescentFlow).
#
# "Internal branching" (BranchingFlows): splits/deletions that change sequence
#   length during diffusion — part of the model.
# "External branching" (DTS tree): exploring multiple diffusion trajectories
#   and selecting the best via reward — what this file implements.

# ─── Snapshotable model wrapper ─────────────────────────────────────────────────

mutable struct BranchTreePredictor
    model               # BranchChainV1, BranchChainV2, or BranchChainV3
    pdb_id
    chain_labels
    feature_func
    sc_frames           # Current self-conditioning frames (variable-dim, mutable)
    device              # Device transfer function (e.g., gpu)
    recycles::Int
    feature_override    # Optional feature override
end

function BranchTreePredictor(model, X0, pdb_id, chain_labels, feature_func;
                             device = identity, recycles = 0, feature_override = nothing)
    resinds = similar(X0.groupings) .= 1:size(X0.groupings, 1)
    input_bundle = if model isa BranchChainV1
        ([0f0]', X0, X0.groupings, resinds, [true]) |> device
    elseif model isa BranchChainV2 || model isa BranchChainV3
        chain_features = broadcast_features([pdb_id], [chain_labels], X0.groupings,
            (a, b) -> feature_func(a, b, override = feature_override))
        ([0f0]', X0, X0.groupings, resinds, [true], chain_features) |> device
    else
        error("Unsupported model type $(typeof(model)) for design_treegen predictor")
    end

    pred = model(input_bundle...) |> cpu
    sc_frames = deepcopy(pred[1])
    BranchTreePredictor(model, pdb_id, chain_labels, feature_func, sc_frames, device, recycles, feature_override)
end

function (pred::BranchTreePredictor)(t, Xt)
    Xtstate = Xt.state
    frominds = Xtstate[4].S.state[:]

    if !isnothing(pred.sc_frames)
        pred.sc_frames = Translation(pred.sc_frames.composed.outer.values[:, :, frominds, :]) ∘
                         Rotation(pred.sc_frames.composed.inner.values[:, :, frominds, :])
    end

    resinds = similar(Xt.groupings) .= 1:size(Xt.groupings, 1)
    input_bundle = if pred.model isa BranchChainV1
        ([t]', Xt, Xt.groupings, resinds, [true]) |> pred.device
    elseif pred.model isa BranchChainV2 || pred.model isa BranchChainV3
        chain_features = broadcast_features([pred.pdb_id], [pred.chain_labels], Xt.groupings,
            (a, b) -> pred.feature_func(a, b, override = pred.feature_override))
        ([t]', Xt, Xt.groupings, resinds, [true], chain_features) |> pred.device
    else
        error("Unsupported model type $(typeof(pred.model)) for design_treegen predictor")
    end

    for _ in 1:pred.recycles
        pred.sc_frames, _ = pred.model(input_bundle..., sc_frames = pred.device(pred.sc_frames))
    end
    result = pred.model(input_bundle..., sc_frames = pred.device(pred.sc_frames)) |> cpu

    pred.sc_frames = deepcopy(result[1])

    state_pred = (ContinuousState(values(translation(result[1]))),
                  ManifoldState(rotM, eachslice(values(linear(result[1])), dims = (3, 4))),
                  result[2], nothing)

    # Reset indexing state after reading it
    Xtstate[4].S.state .= 1:length(Xtstate[4].S.state)

    return state_pred, result[3], result[4]
end

snapshot(pred::BranchTreePredictor) = deepcopy(pred.sc_frames)

function restore!(pred::BranchTreePredictor, snap)
    pred.sc_frames = deepcopy(snap)
    return pred
end

# ─── Tree node ────────────────────────────────────────────────────────────────

mutable struct DTSNode
    state               # BranchingState at this node
    t_index::Int        # index into segments (1-based; depth in tree)
    sc_snapshot         # self-conditioning snapshot at this node
    value::Float64      # soft value estimate v̂
    visit_count::Int    # N(x)
    reward::Union{Nothing, Float64}  # terminal reward (leaf only)
    children::Vector{DTSNode}
    parent::Union{Nothing, DTSNode}
end

function DTSNode(state, t_index, sc_snapshot; parent = nothing)
    DTSNode(state, t_index, sc_snapshot, 0.0, 0, nothing, DTSNode[], parent)
end

is_terminal(node::DTSNode, n_levels::Int) = node.t_index > n_levels
is_fully_expanded(node::DTSNode, max_children::Int) = length(node.children) >= max_children

# ─── Step schedule partitioning ───────────────────────────────────────────────

function partition_steps(stps::AbstractVector, branching_points::AbstractVector)
    boundaries = vcat(branching_points, [1.0])
    segments = Vector{Vector{Float32}}()
    for i in 1:length(boundaries)-1
        lo, hi = boundaries[i], boundaries[i+1]
        seg = Float32[s for s in stps if s >= lo && s <= hi]
        if isempty(seg) || seg[1] > lo
            pushfirst!(seg, Float32(lo))
        end
        if seg[end] < hi
            push!(seg, Float32(hi))
        end
        seg = sort(unique(seg))
        push!(segments, seg)
    end
    return segments
end

# ─── Run a segment of generation steps ───────────────────────────────────────
#
# Unlike ChainStorm's run_segment, this does NOT mask against the initial state.
# CoalescentFlow's step handles masking internally via branchmask and flowmask.

function run_segment(P, state, predictor::BranchTreePredictor, segment_steps::AbstractVector)
    Xt = deepcopy(state)
    for (s1, s2) in zip(segment_steps, segment_steps[begin+1:end])
        hat = predictor(s1, Xt)
        Xt = Flowfusion.step(P, Xt, hat, s1, s2)
    end
    return Xt
end

# ─── Rollout: run from a node to terminal ────────────────────────────────────

function rollout(P, node::DTSNode, predictor::BranchTreePredictor, segments, n_levels)
    restore!(predictor, node.sc_snapshot)
    state = node.state
    for seg_idx in node.t_index:n_levels
        state = run_segment(P, state, predictor, segments[seg_idx])
    end
    return state
end

# ─── UCT selection ────────────────────────────────────────────────────────────

function uct_score(child::DTSNode, parent_visits::Int, c_uct::Float64)
    if child.visit_count == 0
        return Inf
    end
    return child.value + c_uct * sqrt(log(parent_visits) / child.visit_count)
end

function select(node::DTSNode, max_children::Int, c_uct::Float64, n_levels::Int)
    while !is_terminal(node, n_levels)
        if !is_fully_expanded(node, max_children)
            return node
        end
        node = argmax(c -> uct_score(c, node.visit_count, c_uct), node.children)
    end
    return node
end

# ─── Expansion ────────────────────────────────────────────────────────────────

function expand!(P, node::DTSNode, predictor::BranchTreePredictor, segments, n_levels)
    if is_terminal(node, n_levels)
        return node
    end
    restore!(predictor, node.sc_snapshot)
    child_state = run_segment(P, node.state, predictor, segments[node.t_index])
    child_snap = snapshot(predictor)
    child = DTSNode(child_state, node.t_index + 1, child_snap; parent = node)
    push!(node.children, child)
    return child
end

# ─── Soft Bellman backup ─────────────────────────────────────────────────────

function _logsumexp(x)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

function backup!(node::DTSNode, reward::Float64, lambda::Float64)
    node.reward = reward
    node.value = reward
    node.visit_count += 1
    current = node.parent
    while current !== nothing
        current.visit_count += 1
        if lambda == 0.0
            current.value = maximum(c.value for c in current.children)
        else
            lse = _logsumexp(lambda .* [c.value for c in current.children])
            current.value = lse / lambda
        end
        current = current.parent
    end
end

# ─── Best leaf extraction ────────────────────────────────────────────────────

function best_leaf(node::DTSNode)
    if isempty(node.children)
        return node
    end
    return best_leaf(argmax(c -> c.value, node.children))
end

# ─── Main entry point ────────────────────────────────────────────────────────

"""
    design_treegen(model, X1, pdb_id, chain_labels, feature_func; reward, kwargs...)

Run DTS* tree search over the branching flow to find high-reward designs.

Uses Discrete Tree Search with soft Bellman backups to explore multiple
diffusion trajectories through the CoalescentFlow process, selecting the
one that maximizes the given reward function.

- `model`: trained BranchChainV1, BranchChainV2, or BranchChainV3 model.
- `X1`: masked terminal BranchingState (from `X1_from_pdb` etc.).
- `pdb_id`, `chain_labels`, `feature_func`: feature arguments for conditional models.
- `reward`: function `reward(samp::BranchingState) -> Float64` scoring a sample.
- `branching_points`: time points where the DTS tree branches (not BranchingFlows splits).
- `max_children`: maximum children per tree node.
- `n_iterations`: number of DTS* iterations (select → expand → rollout → backup).
- `c_uct`: UCT exploration constant.
- `lambda`: soft Bellman temperature (0 = hard max).
- `steps`: diffusion step schedule (Number, Vector, or default).
- `device`: device function (e.g., `gpu`).
- `P`: CoalescentFlow process.
- `path`: optional PDB output path for best design.
"""
function design_treegen(
    model, X1, pdb_id, chain_labels, feature_func;
    reward,
    branching_points = [0.0, 0.05, 0.15, 0.4, 0.7],
    max_children = 4,
    n_iterations = 100,
    c_uct = 1.0,
    lambda = 1.0,
    steps = step_sched.(0f0:0.005f0:1f0),
    device = identity,
    P = P,
    X0_mean_length = model.layers.config.X0_mean_length_minus_1,
    deletion_pad = model.layers.config.deletion_pad,
    recycles = 0,
    feature_override = nothing,
    t0 = 0f0,
    path = nothing,
)
    # Build step schedule
    if steps isa Number
        stps = step_sched.(t0:Float32((1 - t0) / steps):1f0)
    elseif steps isa AbstractVector
        stps = Float32.(steps)
    else
        stps = step_sched.(t0:0.005f0:1f0)
    end

    n_levels = length(branching_points)
    segments = partition_steps(stps, branching_points)

    # Initialize X0 via branching bridge (same pattern as design())
    t = [Float32(t0)]
    bat = branching_bridge(P, X0sampler, [X1], t,
                            coalescence_factor = 1.0,
                            use_branching_time_prob = 0.0,
                            merger = BranchingFlows.canonical_anchor_merge,
                            length_mins = Poisson(X0_mean_length),
                            deletion_pad = deletion_pad,
                            X1_modifier = X1_modifier)
    X0 = bat.Xt

    # Create predictor and root node
    predictor = BranchTreePredictor(model, X0, pdb_id, chain_labels, feature_func;
                                    device, recycles, feature_override)
    root_snap = snapshot(predictor)
    root = DTSNode(X0, 1, root_snap)

    # DTS* main loop
    for iter in 1:n_iterations
        print("*")

        # Selection
        node = select(root, max_children, c_uct, n_levels)

        # Expansion (if not terminal)
        if !is_terminal(node, n_levels) && !is_fully_expanded(node, max_children)
            node = expand!(P, node, predictor, segments, n_levels)
        end

        # Rollout to terminal
        terminal_state = rollout(P, node, predictor, segments, n_levels)

        # Evaluate reward and backup
        r = Float64(reward(terminal_state))
        backup!(node, r, lambda)
    end
    println()

    # Return the best terminal trajectory
    leaf = best_leaf(root)
    restore!(predictor, leaf.sc_snapshot)
    if is_terminal(leaf, n_levels)
        final_state = leaf.state
    else
        final_state = rollout(P, leaf, predictor, segments, n_levels)
    end

    if !isnothing(path)
        export_pdb(path, final_state.state, final_state.groupings, collect(1:length(final_state.groupings)))
    end

    return final_state
end

# Convenience wrapper matching design() short-form signature
design_treegen(model, X1; kwargs...) = design_treegen(model, X1, "", [""], _default_feature_func(model); kwargs...)

export design_treegen
