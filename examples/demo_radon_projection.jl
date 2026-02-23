using Random
using LinearAlgebra
using SlicedWasserstein

rng = MersenneTwister(431943)

n1, n2 = 150, 150
X1 = randn(rng, 2, n1) .+ [ -2.0, 0.0 ]
X2 = randn(rng, 2, n2) .+ [  2.0, 0.5 ]
X  = hcat(X1, X2)

w = vcat(fill(2.0, n1), fill(1.0, n2))  
μ = DiscreteMeasure(X, w; normalize=true)

θ = randn(rng, 2)
θ ./= norm(θ)

ν = radon_project(μ, θ)

p1 = plot(μ; aspect_ratio=:equal, label="μ", title="2D measure and projection direction")

t = range(-6, 6; length=2)
plot!(p1, t .* θ[1], t .* θ[2]; linewidth=2, label="direction θ")
p2 = plot(ν; dims=(1,), markersize=6, color=:red, label="projection ν", title="Projection onto θ")

plot(p1, p2; layout=(1,2), size=(900,400))