
"""
    SOT(μ, ν; M=1000, cost=(x, y) -> (x - y)^2, rng=Random.default_rng(), seed=nothing)

Compute the Sliced Optimal Transport (SOT) distance between two discrete measures `μ` and `ν`.

# Arguments
- `μ::DiscreteMeasure`: First discrete measure in `ℝ^d`.
- `ν::DiscreteMeasure`: Second discrete measure in `ℝ^d`.
- `M::Integer=1000`: Number of random projections to use.
- `cost::Function=(x, y) -> (x - y)^2`: Cost function to use in 1D optimal transport.
- `rng::AbstractRNG=Random.default_rng()`: Random number generator to use.
- `seed::Union{Integer, Nothing}=nothing`: If provided, it overrides `rng` and initializes a new `MersenneTwister` with the given seed.

# Returns
- `d::Float64`: The estimated SOT distance between `μ` and `ν
"""
function SOT(
    μ::DiscreteMeasure, 
    ν::DiscreteMeasure; 
    M::Integer=1000, 
    cost::F= (x, y) -> (x - y)^2, 
    rng::AbstractRNG=Random.default_rng(), 
    seed::Union{Integer, Nothing}=nothing
    ) where {F}
    (sum(μ.w) ≈ sum(ν.w)) || throw(ArgumentError("Different total masses"))
    size(μ.X,1) == size(ν.X,1) || throw(ArgumentError("μ and ν must have the same dimension"))
    M > 0 || throw(ArgumentError("M must be > 0"))

    d = size(μ.X, 1)
    z = sample_directions(M, d; rng=rng, seed=seed)

    T = Threads.maxthreadid()

    promote_type(eltype(μ.X), Float64)

    pr_X = [Vector{eltype(μ.X)}(undef, size(μ.X, 2)) for _ in 1:T]
    pr_Y = [Vector{eltype(ν.X)}(undef, size(ν.X, 2)) for _ in 1:T]
    sums = zeros(Float64, T)
    Threads.@threads for i in 1:M
        tid = Threads.threadid()

        θ = @view(z[:, i])

        radon_project!(pr_X[tid], μ.X, θ)
        μ_proj = DiscreteMeasure(pr_X[tid], μ.w; normalize=false)

        radon_project!(pr_Y[tid], ν.X, θ)
        ν_proj = DiscreteMeasure(pr_Y[tid], ν.w; normalize=false)

        sums[tid] += OT1d(μ_proj, ν_proj; cost=cost)
    end

    return sum(sums) / M
end