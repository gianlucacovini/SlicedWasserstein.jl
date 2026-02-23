using Random
using BenchmarkTools
using LinearAlgebra
using SlicedWasserstein

rng = MersenneTwister(1)

X1 = randn(rng, 2, 600) .+ [-2.0; 0.0]
X2 = randn(rng, 2, 600) .+ [ 2.0; 0.0]
X3 = randn(rng, 2, 600) .+ [ 0.0; 2.0]

μ1 = DiscreteMeasure(X1)
μ2 = DiscreteMeasure(X2)
μ3 = DiscreteMeasure(X3)

measures = [μ1, μ2, μ3]
w = [0.2, 0.5, 0.3]

seed = 123

println("Threads: ", Threads.maxthreadid())
println("SWBarycenters_free_supp() benchmarks")

# Warm-up
SWBarycenters_free_supp(measures;
    w=w, η=0.05, n_supp=200, itmax=3, tol=0.0, M=5,
    seed=seed
)

# Core component benchmarks (helpful to interpret barycenter runtime)
println("Benchmark: one SOT call (bar vs μ1), M=100")
bar0 = DiscreteMeasure(randn(rng, 2, 200))  # just a stand-in barycenter measure
SOT(bar0, μ1; M=10, seed=seed)              # warm-up
display(@benchmark SOT($bar0, $μ1; M=100, seed=$seed))

println("Benchmark: one grad call (bar vs μ1), M=100")
grad(bar0, μ1; M=10, seed=seed)             # warm-up
display(@benchmark grad($bar0, $μ1; M=100, seed=$seed))

# Barycenter
println("Short run: itmax=10, M=50")
display(@benchmark SWBarycenters_free_supp($measures;
    w=$w, η=0.05, n_supp=200, itmax=10, tol=0.0, M=50,
    seed=$seed
))

println("Longer run: itmax=50, M=100")
display(@benchmark SWBarycenters_free_supp($measures;
    w=$w, η=0.03, n_supp=400, itmax=50, tol=0.0, M=100,
    seed=$seed
))
