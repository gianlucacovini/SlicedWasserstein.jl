using Random
using BenchmarkTools
using SlicedWasserstein
using LinearAlgebra

rng = Xoshiro(431943)

d  = 10
nX = 1000
nY = 1500

X = rand(rng, d, nX)
w = rand(rng, nX)
Y = rand(rng, d, nY)
v = rand(rng, nY)

μ = DiscreteMeasure(X, w)  
ν = DiscreteMeasure(Y, v)

M_small = 100
M_big   = 1000
seed    = 12345

println("Benchmarking SOT (includes sorting in OT1d), d=$d, n=$nX, m=$nY")
println("Threads: ", Threads.maxthreadid())

# warm-up
SOT(μ, ν; M=10, seed=seed)

println("SOT: M = $M_small (seed fixed)")
display(@benchmark SOT($μ, $ν; M=$M_small, seed=$seed))

println("\nSOT: M = $M_big (seed fixed)")
display(@benchmark SOT($μ, $ν; M=$M_big, seed=$seed))

# Custom cost
abs_cost(x, y) = abs(x - y)
println("\nSOT: custom cost abs(x-y), M = $M_small")
display(@benchmark SOT($μ, $ν; M=$M_small, cost=$abs_cost, seed=$seed))

# Baselines
println("Baseline: OT1d on 1D measures of sizes n=$nX, m=$nY (includes sorting)")
μ1 = DiscreteMeasure(rand(rng, 1, nX))
ν1 = DiscreteMeasure(rand(rng, 1, nY))
OT1d(μ1, ν1)  # warm-up
display(@benchmark OT1d($μ1, $ν1))

println("Baseline: one radon projection (mul!) cost, sizes d=$d, n=$nX")
θ = randn(rng, d); θ ./= norm(θ)
pr = Vector{Float64}(undef, nX)
radon_project!(pr, μ.X, θ) # warm-up
display(@benchmark radon_project!($pr, $μ.X, $θ))

nothing