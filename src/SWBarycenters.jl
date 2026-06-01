
"""
    grad(μ, ν; M=1000, rng=Random.default_rng(), seed=nothing)

Compute the gradient of the Sliced Optimal Transport quadratic cost between two discrete measures `μ` and `ν` with respect to the support points of `μ`.

# Arguments
- `μ::DiscreteMeasure`: First discrete measure.
- `ν::DiscreteMeasure`: Second discrete measure.
- `M::Integer=1000`: Number of random projections to use.
- `rng::AbstractRNG=Random.default_rng()`: Random number generator.
- `seed::Union{Integer, Nothing}=nothing`: Seed for the random number generator. If provided, it overrides `rng`.
# Returns
- `∇::Array{Float64,2}`: Gradient of the Sliced Optimal Transport quadratic cost with respect to the support points of `μ`.
# References
- K. Nguyen, "An Introduction to Sliced Optimal Transport", 2023, Remark 4.2
"""
function grad(
    μ::DiscreteMeasure, 
    ν::DiscreteMeasure; 
    M::Integer=1_000, 
    rng::AbstractRNG=Random.default_rng(), 
    seed::Union{Integer, Nothing}=nothing
    )
    d, n = size(μ.X)
    e, m = size(ν.X)
    (d == e) || throw(ArgumentError("Dimensional mismatch between the two arguments"))
    isapprox(sum(μ.w), sum(ν.w)) || throw(ArgumentError("Different total masses"))

    z = sample_directions(M, d; rng=rng, seed=seed)

    T = Threads.maxthreadid()

    pr_X = [Vector{eltype(μ.X)}(undef, n) for _ in 1:T]
    pr_Y = [Vector{eltype(ν.X)}(undef, m) for _ in 1:T]
    grads = [zeros(eltype(μ.X), d, n) for _ in 1:T]

    Threads.@threads :static for i_dir in 1:M
        tid = Threads.threadid()

        θ = @view(z[:, i_dir])  
        
        radon_project!(pr_X[tid], μ.X, θ)
        μ_proj = DiscreteMeasure(pr_X[tid], μ.w; normalize=false)

        radon_project!(pr_Y[tid], ν.X, θ)
        ν_proj = DiscreteMeasure(pr_Y[tid], ν.w; normalize=false)
        
        r = OT1d(μ_proj, ν_proj; compute_cost=false, compute_edge=true)
        I  = r.I::Vector{Int}
        J  = r.J::Vector{Int}
        Tm = r.Tm::Vector{Float64}

        g = grads[tid]
        px = pr_X[tid]
        py = pr_Y[tid]

        @inbounds for k in eachindex(Tm)
            i = I[k]
            j = J[k]
            t = Tm[k]

            s = px[i] - py[j] # Use projections to compute s 

            c = 2 * s * t

            @simd for a in 1:d
                g[a, i] += c * θ[a]
            end
        end
    end

    ∇ = zeros(eltype(μ.X), d, n)
    for tid in 1:T
        ∇ .+= grads[tid]
    end

    return ∇ / M
end

"""
    SWBarycenters_free_supp(measures; η=nothing, w=nothing, n_supp=nothing, n_supp_max=500, n_supp_min=1, itmax=100, tol=1e-6, M=500, rng=Random.default_rng(), seed=nothing, normalize=true, save_it=false)

Compute the Sliced-Wasserstein barycenter of a set of discrete measures with free support using stochastic gradient descent.

# Arguments
- `measures::AbstractVector`: Vector of discrete measures.
- `η::Union{Real, Nothing}=nothing`: Learning rate. If `nothing`, it is set automatically based on the measures.
- `w::Union{AbstractVector{<:Real}, Nothing}=nothing`: Weights
- `n_supp::Union{Integer, Nothing}=nothing`: Number of support points of the barycenter. If `nothing`, it is set automatically based on the measures.
- `n_supp_max::Union{Integer, Nothing}=500`: Maximum number of support
points if `n_supp` is not provided.
- `n_supp_min::Union{Integer, Nothing}=1`: Minimum number of support points if `n_supp` is not provided.
- `itmax::Integer=100`: Maximum number of iterations.
- `tol::Real=1e-6`: Tolerance for the stopping criterion.
- `M::Integer=500`: Number of random projections to use in the gradient computation.
- `rng::AbstractRNG=Random.default_rng()`: Random number generator.
- `seed::Union{Integer, Nothing}=nothing`: Seed for the random number generator. If provided, it overrides `rng`.
- `normalize::Bool=true`: Whether to normalize the barycenter measure.
- `save_it::Bool=false`: Whether to save the support points at every 20 iterations.

# Returns
- `DiscreteMeasure`: The computed Sliced-Wasserstein barycenter.
If `save_it` is true, returns a tuple `(iterations, barycenter)` where `iterations` is a vector of support points at every 20 iterations.
"""
function SWBarycenters_free_supp(
    measures::AbstractVector;
    η::Union{Real, Nothing}=nothing,
    w::Union{AbstractVector{<:Real}, Nothing}=nothing, 
    n_supp::Union{Integer, Nothing}=nothing,
    n_supp_max::Union{Integer, Nothing}=500,
    n_supp_min::Union{Integer, Nothing}=1,
    itmax::Integer = 100,
    tol::Real=1e-4,
    M::Integer=500, 
    rng::AbstractRNG=Random.default_rng(), 
    seed::Union{Integer, Nothing}=nothing, 
    normalize::Bool=true,
    save_it::Bool =false
    )

    (n_supp_min ≤ n_supp_max) || throw(ArgumentError("n_supp_min should be lower than n_supp_max"))

    len_meas = length(measures)
    (len_meas ≥ 1) || throw(ArgumentError("The argument must contain at least one measure"))

    if w === nothing
        w = fill(1.0/len_meas, len_meas)
    else
        (length(w) == len_meas) || throw(ArgumentError("The weights vector must have the same length as the measures vector"))
        (sum(w) ≈ 1.0) || throw(ArgumentError("The weights must sum to 1"))
    end

    if seed !== nothing
        local_rng = Xoshiro(seed)
    else
        local_rng = rng
    end

    measures = normalize ? [DiscreteMeasure(μ.X, μ.w; normalize=true) for μ in measures] : measures

    d, _ = size(measures[1].X)

    if n_supp === nothing
        n_i = [size(μ.X, 2) for μ in measures]
        n_supp = min(round(Int, median(n_i)), n_supp_max)
        n_supp = max(n_supp, n_supp_min)
    end

    # warm-start 
    ϕ = zeros(d, n_supp)
    for j in 1:len_meas
        ϕ .+= w[j] .* mean(measures[j])
    end

    # weighted average second moment around each measure mean
    s2 = 0.0
    for j in 1:len_meas
        μ = measures[j]
        mj = μ.X * μ.w
        for i in 1:size(μ.X,2)
            s2 += w[j] * μ.w[i] * sum(abs2, μ.X[:,i] .- mj)
        end
    end
    σ = sqrt(s2 / d)

    if η === nothing
        η = maximum([0.5*σ^2, 0.5])
    end

    # add small noise to avoid collapse
    ϕ = ϕ .+ 0.5 * σ^2 .* randn(local_rng, d, n_supp)

    # SGD update
    prev_ϕ = fill(Inf, size(ϕ))
    i = 1

    iterations = Matrix{eltype(ϕ)}[]

    gacc = zeros(size(ϕ))
    while (norm(ϕ - prev_ϕ) > tol) && (i ≤ itmax) # stopping conditions
        bar = DiscreteMeasure(ϕ)

        copy!(prev_ϕ, ϕ)
        fill!(gacc, 0)

        for k in eachindex(measures)
            (size(measures[k].X, 1) == d) || throw(ArgumentError("All measures must have the same dimension"))
            gk = grad(bar, measures[k]; M=M, rng=local_rng, seed=nothing)
            @inbounds gacc .+= w[k] .* gk
        end

        @inbounds ϕ .-= η/√i .* gacc

        if save_it && (i % 20 == 1)
            push!(iterations, copy(ϕ))
        end

        i += 1
    end

    if save_it
        return iterations, DiscreteMeasure(ϕ)
    end

    return DiscreteMeasure(ϕ)
end