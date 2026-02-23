using Random
using LinearAlgebra
using Plots
using SlicedWasserstein

rng = MersenneTwister(431943)

println("OT1d demo")

# Build two 1D discrete measures (small, for dense plan demo)
x = [0.0, 1.0, 3.0, 5.0, 7.0]
w = [0.1, 0.4, 0.2, 0.1, 0.2]
μ = DiscreteMeasure(x, w; normalize=false)

y = [0.0, 2.0, 4.0, 6.0, 8.0]
v = [0.1, 0.4, 0.2, 0.2, 0.1]
ν = DiscreteMeasure(y, v; normalize=false)

println("Total mass μ = ", sum(μ.w), "   ν = ", sum(ν.w))

# Cost-only
s = OT1d(μ, ν)  # default squared cost
println("OT1d cost (squared): ", s)

# Custom cost: L1
abs_cost(x, y) = abs(x - y)
s_abs = OT1d(μ, ν; cost=abs_cost)
println("OT1d cost (abs):     ", s_abs)

# Dense plan (small only)
sP, P = OT1d(μ, ν; compute_plan=true)
println("Dense plan computed. Sum(P) = ", sum(P), " (should match total mass)")

# Visualize the dense plan
p_plan = heatmap(P;
    title="Transport plan",
    xlabel="ν index",
    ylabel="μ index",
    colorbar_title="mass"
)

# Edge plan
sE, I, J, Tm = OT1d_edge(μ, ν; compute_cost=true, compute_edge=true)
println("Edge plan computed: nonzeros = ", length(Tm), ", cost = ", sE)

# A simple 1D “transport arrows” plot:
# plot atoms on two horizontal lines and draw segments with opacity proportional to transported mass
xμ = vec(μ.X)
xν = vec(ν.X)

p_edges = plot(; title="Edge transport plan (1D)", legend=false, yticks=([0,1], ["μ","ν"]))
scatter!(p_edges, xμ, fill(0.0, length(xμ)); markersize=6 .* (μ.w ./ maximum(μ.w)), label="μ")
scatter!(p_edges, xν, fill(1.0, length(xν)); markersize=6 .* (ν.w ./ maximum(ν.w)), label="ν")

# draw edges
tmax = maximum(Tm)
for k in eachindex(Tm)
    α = 0.1 + 0.9 * (Tm[k] / tmax)  # scale opacity
    plot!(p_edges, [xμ[I[k]], xν[J[k]]], [0.0, 1.0]; linewidth=2, alpha=α)
end

# Unnormalized mass
μu = DiscreteMeasure([0.0, 1.0], [1.0, 1.0]; normalize=false)     # total 2
νu = DiscreteMeasure([0.0, 2.0], [0.5, 1.5]; normalize=false)     # total 2
println("\nUnnormalized masses: sum(μu.w)=", sum(μu.w), ", sum(νu.w)=", sum(νu.w))
println("OT1d(μu, νu) = ", OT1d(μu, νu))

# Show plots
plot(p_plan, p_edges; layout=(1,2), size=(1000, 380))
