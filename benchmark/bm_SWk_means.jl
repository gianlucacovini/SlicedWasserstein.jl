using Random
using BenchmarkTools
using SlicedWasserstein

# Problem size 
d         = 2
npoints   = 30      # atoms per measure
nmeasures = 10      # number of measures
K         = 2       # clusters

# Algorithm params 
itmax      = 10
M_SOT      = 200
M_bary     = 200
itmax_bary = 20
seed_algo  = 1234

# Data seed
rng = MersenneTwister(42)

measures = Vector{DiscreteMeasure{Float64}}(undef, nmeasures)
noise = rand(rng, -25:25, d, nmeasures)
for i in 1:nmeasures
    points = randn(rng, d, npoints) .+ noise[:, i]
    measures[i] = DiscreteMeasure(points)
end

# debug plot
"""
p = plot(measures[1])
for i in 2:nmeasures
    plot!(p, measures[i])
end

display(p)
"""

# Warm-up (compile)
SWk_means(measures, K; itmax=1, M_SOT=10, M_bary=10, itmax_bary=1, seed=seed_algo)

println("Julia threads: ", Threads.nthreads())
println("d=$d, npoints=$npoints, nmeasures=$nmeasures, K=$K")
println("itmax=$itmax, M_SOT=$M_SOT, M_bary=$M_bary, itmax_bary=$itmax_bary, seed=$seed_algo")

# Benchmark
b = @benchmark SWk_means($measures, $K;
    itmax=$itmax, M_SOT=$M_SOT, M_bary=$M_bary, itmax_bary=$itmax_bary, seed=$seed_algo
)

display(b)

nothing
