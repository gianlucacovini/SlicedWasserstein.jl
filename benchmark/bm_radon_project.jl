using BenchmarkTools
using SlicedWasserstein
using Random
using LinearAlgebra

rng = MersenneTwister(431943)

X = rand(rng, 5, 100)
w = rand(rng, 100)
μ = DiscreteMeasure(X, w)  

θ = randn(rng, 5)
θ ./= norm(θ)

println("Benchmarking radon_project (allocating) on n=$(size(X,2)), d=$(size(X,1))")
display(@benchmark radon_project($μ, $θ))

pr_X = similar(w) 
println("Benchmarking radon_project! (in-place)")
display(@benchmark radon_project!($pr_X, $X, $θ))

nothing