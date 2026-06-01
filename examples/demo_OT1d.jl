using Random
using LinearAlgebra
using Plots
using SlicedWasserstein

rng = Xoshiro(431943)

println("OT1d demo")

x = [0.0, 1.0, 3.0, 5.0, 7.0]
w = [0.1, 0.4, 0.2, 0.1, 0.2]
μ = DiscreteMeasure(x, w; normalize=false)

y = [0.0, 2.0, 4.0, 6.0, 8.0]
v = [0.1, 0.4, 0.2, 0.2, 0.1]
ν = DiscreteMeasure(y, v; normalize=false)

println("Total mass μ = ", sum(μ.w), "   ν = ", sum(ν.w))

s = OT1d(μ, ν).cost
println("OT1d cost (squared): ", s)

abs_cost(x, y) = abs(x - y)
s_abs = OT1d(μ, ν; cost=abs_cost).cost
println("OT1d cost (abs):     ", s_abs)

r = OT1d(μ, ν; compute_edge=true)
I, J, Tm = r.I::Vector{Int}, r.J::Vector{Int}, r.Tm::Vector{Float64}
println("Edge plan computed: nonzeros = ", length(Tm), ", cost = ", r.cost)

n, m = size(μ.X, 2), size(ν.X, 2)
P = zeros(Float64, n, m)
for k in eachindex(Tm)
    P[I[k], J[k]] += Tm[k]
end
println("Dense plan. Sum(P) = ", sum(P), " (should match total mass)")

p_plan = heatmap(P;
    title="Transport plan",
    xlabel="ν index",
    ylabel="μ index",
    colorbar_title="mass"
)

xμ = vec(μ.X)
xν = vec(ν.X)

p_edges = plot(; title="Edge transport plan (1D)", legend=false, yticks=([0,1], ["μ","ν"]))
scatter!(p_edges, xμ, fill(0.0, length(xμ)); markersize=6 .* (μ.w ./ maximum(μ.w)), label="μ")
scatter!(p_edges, xν, fill(1.0, length(xν)); markersize=6 .* (ν.w ./ maximum(ν.w)), label="ν")

tmax = maximum(Tm)
for k in eachindex(Tm)
    α = 0.1 + 0.9 * (Tm[k] / tmax)
    plot!(p_edges, [xμ[I[k]], xν[J[k]]], [0.0, 1.0]; linewidth=2, alpha=α)
end

μu = DiscreteMeasure([0.0, 1.0], [1.0, 1.0]; normalize=false)
νu = DiscreteMeasure([0.0, 2.0], [0.5, 1.5]; normalize=false)
println("\nUnnormalized masses: sum(μu.w)=", sum(μu.w), ", sum(νu.w)=", sum(νu.w))
println("OT1d(μu, νu) = ", OT1d(μu, νu).cost)

plot(p_plan, p_edges; layout=(1,2), size=(1000, 380))
