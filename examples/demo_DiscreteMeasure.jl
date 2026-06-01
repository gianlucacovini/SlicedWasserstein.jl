using Random
using LinearAlgebra
using SlicedWasserstein
using Plots

rng = Xoshiro(431943)

# Construct a 2D discrete measure
n = 200
X = randn(rng, 2, n)
w = rand(rng, n)

μ = DiscreteMeasure(X, w)   # normalized by default

println("print(μ):")
println(μ)

println("show(μ) (same as print):")
show(μ)
println("\n")

println("REPL-style display (rich show):")
display(μ)
println("\n")

# Plot the measure support
p1 = plot(μ;
    title="DiscreteMeasure support (2D)",
    label="μ",
    color=:blue
)

# small measure example
x_small = [0.0 1.0 2.0]
w_small = [0.2, 0.5, 0.3]
μ_small = DiscreteMeasure(x_small, w_small)

println("Small measure example print:")
println(μ_small)

println("Small measure example show:")
show(μ_small)
println("\n")   

println("Small measure example display:")
display(μ_small)
println("\n")

# 1D measure example
x = randn(rng, 100)
w1 = rand(rng, 100)

ν = DiscreteMeasure(x, w1; normalize=false)

println("Unnormalized 1D measure:")
display(ν)

p2 = plot(ν;
    dims=(1,),
    title="1D DiscreteMeasure (unnormalized weights)",
    label="ν",
    color=:red
)

plot(p1, p2; layout=(1,2), size=(900, 350))
