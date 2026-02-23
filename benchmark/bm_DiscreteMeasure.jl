using BenchmarkTools
using Random
using LinearAlgebra
using Statistics
using SlicedWasserstein

rng = MersenneTwister(431943)

function make_data(TX::Type, TW::Type, d::Int, n::Int; normalize_w::Bool=true)
    X = rand(rng, TX, d, n)
    w = rand(rng, TW, n)
    if normalize_w
        w ./= sum(w)
    end
    return X, w
end

# Constructors
let
    d, n = 10, 100_000
    X, w = make_data(Float64, Float64, d, n; normalize_w=false)

    println("Constructor: DiscreteMeasure(X,w) (normalize=true default), d=$d n=$n")
    display(@benchmark DiscreteMeasure($X, $w))

    println("Constructor: DiscreteMeasure(X,w; normalize=false), d=$d n=$n")
    display(@benchmark DiscreteMeasure($X, $w; normalize=false))

    println("@btime comparisons (same sizes):")
    @btime DiscreteMeasure($X, $w)
    @btime DiscreteMeasure($X, $w; normalize=false)
end

# Promotion (Float32 X, Float64 w)
let
    d, n = 10, 100_000
    X, _ = make_data(Float32, Float32, d, n; normalize_w=false)
    _, w = make_data(Float64, Float64, d, n; normalize_w=false)

    println("Promotion: X Float32, w Float64 -> DiscreteMeasure{Float64}, d=$d n=$n")
    @btime DiscreteMeasure($X, $w)
end

# Uniform-weight constructor DiscreteMeasure(X)
let
    d, n = 10, 200_000
    X = rand(rng, Float64, d, n)

    println("Uniform weights: DiscreteMeasure(X), d=$d n=$n")
    @btime DiscreteMeasure($X)
end

# sort_1d
let
    n = 200_000
    x = randn(rng, Float64, n)
    w = rand(rng, Float64, n)
    μ = DiscreteMeasure(x, w; normalize=false)

    println("sort_1d: n=$n")
    @btime sort_1d($μ)
end

# mean
let
    d, n = 50, 50_000
    X, w = make_data(Float64, Float64, d, n; normalize_w=true)
    μ = DiscreteMeasure(X, w; normalize=false)  # weights already normalized here

    println("mean(μ): d=$d n=$n")
    @btime mean($μ)
end

nothing