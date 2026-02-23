using Random
using Plots
using SlicedWasserstein

rng = MersenneTwister(1234)

n = 150
μ1 = DiscreteMeasure(randn(rng, 2, n) .+ [ 10.0, 10.0])
μ2 = DiscreteMeasure(randn(rng, 2, n) .+ [-10.0,-10.0])
μ3 = DiscreteMeasure(randn(rng, 2, n) .+ [ 10.0,  0.0])
μ4 = DiscreteMeasure(randn(rng, 2, n) .+ [  0.0, 10.0])

measures = [μ1, μ2, μ3, μ4]

# Plot inputs
p = plot(μ1; label="μ1", markersize=3, color=:blue, title="SW barycenter (free support)")
plot!(p, μ2; label="μ2", markersize=3, color=:red)
plot!(p, μ3; label="μ3", markersize=3, color=:orange)
plot!(p, μ4; label="μ4", markersize=3, color=:green)

iters, bar = SWBarycenters_free_supp(measures; save_it=true)

for (k, ϕ) in enumerate(iters)
    plot!(p, DiscreteMeasure(ϕ); label="it $(k*20)", markersize=4, color=:gray, alpha=0.3)

    display(p)
    sleep(2)
end

plot!(p, bar; label="barycenter", markersize=6, color=:yellow)

display(p)