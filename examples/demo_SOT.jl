using Random
using LinearAlgebra
using Statistics
using Plots
using SlicedWasserstein

rng  = Xoshiro(431943)
seed = 12345

# Build two 2D discrete measures (two clusters)
n = 300

Xμ = hcat(randn(rng, 2, n) .+ [-5.0; 0.0],
          randn(rng, 2, n) .+ [ 5.0; 0.0])

Xν = hcat(randn(rng, 2, n) .+ [0.0; -5.0],
          randn(rng, 2, n) .+ [0.0;  5.0])

wμ = fill(1.0, size(Xμ,2))
wν = fill(1.0, size(Xν,2))

μ = DiscreteMeasure(Xμ, wμ)   
ν = DiscreteMeasure(Xν, wν)

println("mass(μ) = ", sum(μ.w), "   mass(ν) = ", sum(ν.w))

p0 = plot(μ; dims=(1,2), label="μ", color=:blue, title="Two 2D measures")
plot!(p0, ν; dims=(1,2), label="ν", color=:red)
display(p0)

# SOT vs number of directions M (Monte Carlo convergence)
Ms = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
vals = Float64[]

for M in Ms
    sd = SOT(μ, ν; M=M, seed=seed)
    push!(vals, sd)
end

println("SOT estimate vs M (seed fixed):")
for (M, sd) in zip(Ms, vals)
    println("  M = $(lpad(M,4))   SOT = $sd")
end

p1 = plot(Ms, vals; marker=:circle, xscale=:log10,
          xlabel="M (number of random directions, log scale)",
          ylabel="SOT estimate",
          title="MC convergence",
          label="SOT(M)")

# Reproducibility (same seed)
M = 500
s1 = SOT(μ, ν; M=M, seed=777)
s2 = SOT(μ, ν; M=M, seed=777)

println("Reproducibility check (same seed):")
println("  run1 = ", s1)
println("  run2 = ", s2)
println("  |diff| = ", abs(s1 - s2))

# Different costs: squared (default) vs absolute distance
abs_cost(x, y) = abs(x - y)
s_sq  = SOT(μ, ν; M=500, seed=seed)
s_abs = SOT(μ, ν; M=500, cost=abs_cost, seed=seed)

println("Different costs (M=500):")
println("  squared cost SOT = ", s_sq)
println("  abs cost     SOT = ", s_abs)

# One random slice (Radon projection) + OT1d cost
θ = randn(rng, 2); θ ./= norm(θ)
μp = radon_project(μ, θ)
νp = radon_project(ν, θ)
ot_slice = OT1d(μp, νp).cost

p2 = plot(μp; dims=(1,), label="μ projected", color=:blue,
          title="One random 1D slice")
plot!(p2, νp; dims=(1,), label="ν projected", color=:red)

plot(p0, p1, p2, layout=(3,1), size=(900,900))
