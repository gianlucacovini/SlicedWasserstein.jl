using SlicedWasserstein
using Random
using StatsBase
using LinearAlgebra

# Problem size 
d         = 2
npoints   = 30      # atoms per measure
nmeasures = 20      # number of measures
K         = 4       # clusters

# Algorithm params 
itmax      = 50
M_SOT      = 200
M_bary     = 200
itmax_bary = 20
seed_algo  = 1234

# Demo params
noise_range = -25:25

# Data seed
rng = Xoshiro(42)

measures = Vector{DiscreteMeasure{Float64}}(undef, nmeasures)
noise = rand(rng, noise_range, d, nmeasures)
for i in 1:nmeasures
    points = randn(rng, d, npoints) .+ noise[:, i]
    measures[i] = DiscreteMeasure(points)
end

assignments, centroids = SWk_means(measures, K;
    itmax=itmax, M_SOT=M_SOT, M_bary=M_bary, itmax_bary=itmax_bary, seed=seed_algo
)

println(assignments)

for μ in centroids
    display(μ)
end

p = plot(measures[1]; color=:black, ms=assignments[1])

colors = [:red, :green, :orange, :purple, :brown, :pink, :gray, :cyan, :magenta, :yellow, :lime, :teal, :navy, :maroon, :olive, :coral, :gold, :indigo, :violet, :turquoise, :salmon, :lavender, :beige, :mint, :peach, :apricot, :charcoal, :tan, :ivory, :plum, :mustard, :cerulean]

for i in 2:nmeasures
    plot!(p, measures[i]; color=colors[i], ms=assignments[i])
end

for i in 1:K
    plot!(p, centroids[i]; ma=0.3, ms=i)
end

display(p)
