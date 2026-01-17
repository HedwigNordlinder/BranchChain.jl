using Pkg
Pkg.activate(".")
using Flowfusion, ForwardBackward, Plots


θ, v0, v1, dec = 100f0, 50f0, 0.000000001f0, -0.1f0
P = OUBridgeExpVar(θ, v0, v1, dec = dec)

#Visualizing the continuous process:
spread = 0.5f0
#baseX0 = ContinuousState(.-ones(Float32, 10) .* spread/2)
baseX0 = ContinuousState(randn(Float32, 10) .* spread)
baseX1 = ContinuousState(randn(Float32, 10) .* 0.2) #ContinuousState(ones(Float32, 10) .* spread/2)
baseXt = deepcopy(baseX0)
traj = []
for t in 0f0:0.001f0:0.999f0
    baseXt = bridge(P, baseXt, baseX1, t, t+0.001f0)
    push!(traj, copy(tensor(baseXt)))
end
pl = plot(stack(traj)', label = :none)
mkpath("processplots")
savefig(pl, "processplots/P_θ$(θ)_v0$(v0)_v1$(v1)_dec$(dec).pdf")


θ, v0, v1, dec = 100f0, 50f0, 0.000000001f0, -0.1f0
P = OUBridgeExpVar(θ, v0, v1, dec = dec)
println("Std at t=0: ", sqrt(v0/(2θ)))

spread = 0.5f0;
baseX0 = ContinuousState(randn(Float32, 1000) .* spread);
baseX1 = ContinuousState(zeros(Float32, 1000));
baseXt = deepcopy(baseX0);
traj = []
for t in 0f0:0.001f0:0.999f0
    baseXt = bridge(P, baseXt, baseX1, t, t+0.001f0)
    push!(traj, copy(tensor(baseXt)))
end
pl = plot(std.(traj));
savefig(pl, "processplots/stds_θ$(θ)_v0$(v0)_v1$(v1)_dec$(dec).pdf");

