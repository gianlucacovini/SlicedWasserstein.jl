using Random
using SlicedWasserstein
using Plots
using StatsBase

rng = Xoshiro(431943)

M = 2000
Z = sample_directions(M, 2; rng=rng)

x = @view Z[1, :]
y = @view Z[2, :]

θ = range(0, 2π; length=400)
p = plot(cos.(θ), sin.(θ); label="unit circle", linewidth=2)

scatter!(p, x, y;
    aspect_ratio=:equal,
    label="samples",
    markersize=2,
    alpha=0.4,
    title="sample_directions(M=$M, d=2)"
)

display(p)

println("Mean direction ≈ ", (mean(x), mean(y)))
