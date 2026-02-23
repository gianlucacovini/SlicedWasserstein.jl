
"""
    sample_directions(M, d; rng=Random.default_rng(), seed=nothing)

Sample `M` random directions in `d` dimensions, uniformly from the unit sphere.

# Arguments
- `M::Integer`: Number of directions to sample (must be `≥ 0`).
- `d::Integer`: Dimension of the space (must be `> 0`).
- `rng::AbstractRNG`: Random number generator to use (default: `Random.default_rng()`).
- `seed::Union{Integer, Nothing}`: If provided, it overrides `rng` and initializes a new `MersenneTwister` with the given seed.
- `T::Type{<:AbstractFloat}`: The floating-point type for the output (default: `Float64`).

# Returns
- `Z::Matrix{T}`: A `d x M` matrix whose columns are the sampled directions.
"""
function sample_directions(
    M::Integer, 
    d::Integer; 
    rng::AbstractRNG=Random.default_rng(), 
    seed::Union{Integer, Nothing}=nothing,
    T::Type{<:AbstractFloat}=Float64
    )
    (M ≥ 0 && d ≥ 1) || throw(ArgumentError("The first argument must be ≥ 0 and the second > 0"))

    local_rng = seed === nothing ? rng : Random.MersenneTwister(seed)
    
    Z = randn(local_rng, T, d, M)

    @inbounds for j in 1:M
            norm = sqrt(sum(abs2, @view(Z[:, j])))
            @view(Z[:, j]) ./= norm
    end
        
    return Z
end

"""
    radon_project(μ, θ)
Compute the Radon projection of a discrete measure `μ` along direction `θ`.

# Arguments
- `μ::DiscreteMeasure{T}`: Discrete measure in `ℝ^d`.
- `θ::AbstractVector{T}`: Direction in `ℝ^d`.

# Returns
- `pr_μ::DiscreteMeasure{T}`: The Radon projection of `μ` along `θ`, a discrete measure in `ℝ`.
"""
function radon_project(
    μ::DiscreteMeasure{T}, 
    θ::AbstractVector{T}
    ) where T<:Real
    d, n = size(μ.X)
    (d == length(θ)) || throw(ArgumentError("The points and the direction have different dimension"))

    S = promote_type(eltype(μ.X), eltype(θ))
    pr_X = Vector{S}(undef, n)

    mul!(pr_X, S.(μ.X'), S.(θ))

    return DiscreteMeasure(pr_X, μ.w; normalize=false)
end

"""
    radon_project!(pr_X, X, θ)
Compute the Radon projection of points `X` along direction `θ`, storing the result in `pr_X`.

# Arguments
- `pr_X::AbstractVector{T}`: Output vector to store the projected points (must have length equal to the number of columns of `X`).
- `X::AbstractMatrix{T}`: Points in `ℝ^d` (each column is a point).
- `θ::AbstractVector{T}`: Direction in `ℝ^d`.

# Returns
- `pr_X::AbstractVector{T}`: The projected points along `θ`.
"""
function radon_project!(
    pr_X::AbstractVector{T}, 
    X::AbstractMatrix{T}, 
    θ::AbstractVector{T}
    ) where T<:Real
    (size(X, 1) == length(θ)) || throw(ArgumentError("The points and the direction have different dimension"))
    (size(X, 2) == length(pr_X)) || throw(ArgumentError("The output vector has incorrect length"))

    mul!(pr_X, X', θ)

    return pr_X
end
