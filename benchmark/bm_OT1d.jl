using BenchmarkTools
using SlicedWasserstein
using Random

rng = Xoshiro(431943)

n_big = 1000
m_big = 1500

X = rand(rng, 1, n_big)
w = rand(rng, n_big)
Y = rand(rng, 1, m_big)
v = rand(rng, m_big)

μ = DiscreteMeasure(X, w)
ν = DiscreteMeasure(Y, v)

println("OT1d (cost only), n=$n_big m=$m_big")
display(@benchmark OT1d($μ, $ν))

let
    n, m = 200, 200
    Xs = rand(rng, 1, n); ws = rand(rng, n)
    Ys = rand(rng, 1, m); vs = rand(rng, m)
    μs = DiscreteMeasure(Xs, ws)
    νs = DiscreteMeasure(Ys, vs)
    println("OT1d (compute_edge=true), n=$n m=$m")
    display(@benchmark OT1d($μs, $νs; compute_edge=true))
end

println("OT1d (compute_edge=true), n=$n_big m=$m_big")
display(@benchmark OT1d($μ, $ν; compute_edge=true))

abs_cost(x, y) = abs(x - y)
println("OT1d with custom cost abs(x-y), n=$n_big m=$m_big")
display(@benchmark OT1d($μ, $ν; cost=$abs_cost))

μs, _ = sort_1d(μ)
νs, _ = sort_1d(ν)
println("OT1d n=$n_big m=$m_big on presorted inputs")
display(@benchmark OT1d($μs, $νs))

nothing
