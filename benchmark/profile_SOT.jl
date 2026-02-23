using Random, SlicedWasserstein, ProfileView

rng = MersenneTwister(431943)
d, nX, nY = 2, 30, 50
X = rand(rng, d, nX); w = rand(rng, nX)
Y = rand(rng, d, nY); v = rand(rng, nY)
μ = DiscreteMeasure(X, w)
ν = DiscreteMeasure(Y, v)

@profview SOT(μ, ν; M=100, seed=12345)