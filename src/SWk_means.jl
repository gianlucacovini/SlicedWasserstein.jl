
"""
    SWk_means(measures, k; itmax=50, M_SOT=1000, M_bary=500, itmax_bary=100, tol_bary=1e-6, rng=Random.default_rng(), seed=nothing)

Performs k-means clustering of the input discrete measures using the Sliced Wasserstein distance.

# Arguments
- `measures::AbstractVector{DiscreteMeasure}`: Vector of discrete measures to cluster.
- `k::Integer`: Number of clusters.
- `itmax::Integer=50`: Maximum number of k-means iterations.
- `M_SOT::Integer=1000`: Number of random projections for SOT distance computation.
- `M_bary::Integer=500`: Number of random projections for barycenter computation.
- `itmax_bary::Integer=100`: Maximum number of iterations for barycenter computation.
- `tol_bary::Real=1e-6`: Tolerance for barycenter computation.
- `rng::AbstractRNG=Random.default_rng()`: Random number generator.
- `seed::Union{Integer, Nothing}=nothing`: Seed for the random number generator. If provided, it overrides `rng`.

# Returns
- `assignments::Vector{Int}`: Vector indicating the cluster assignment for each measure.
- `centroids::Vector{DiscreteMeasure}`: Vector of computed cluster centroids (barycenters).
"""
function SWk_means(
    measures::AbstractVector{<:DiscreteMeasure}, 
    k::Integer; 
    itmax::Integer=50,
    M_SOT::Integer=1000,
    M_bary::Integer=500,
    itmax_bary::Integer=100,
    tol_bary::Real=1e-6,
    rng::AbstractRNG=Random.default_rng(), 
    seed::Union{Integer, Nothing}=nothing
    )

    n_meas = length(measures)

    (k ≤ n_meas) || throw(ArgumentError("The number of clusters must be smaller or equal than the number of measures"))
    (k ≥ 1) || throw(ArgumentError("The number of clusters must be at least 1"))
    (itmax ≥ 1) || throw(ArgumentError("The maximum number of iterations must be at least 1"))
    
    local_rng = seed === nothing ? rng : MersenneTwister(seed)

    # Warm-initialize (k-means++)
    centroids = Vector{DiscreteMeasure{eltype(measures[1].X)}}(undef, k)

    sampled = falses(n_meas) 
    sampled_idx = Vector{Int}(undef, k)

    idx = sample(local_rng, 1:n_meas)
    centroids[1] = measures[idx]
    sampled_idx[1] = idx
    sampled[idx] = true

    dist = zeros(Float64, n_meas)
    for j in 1:n_meas
        if !sampled[j]
            dist[j] = SOT(centroids[1], measures[j]; rng=local_rng, M=M_SOT)
        end
    end

    for i in 2:k
        idx = sample(local_rng, 1:n_meas, Weights(dist)) 
        centroids[i] = measures[idx]
        sampled_idx[i] = idx
        sampled[idx] = true

        for j in 1:n_meas
            if !sampled[j]
                dj = SOT(centroids[i], measures[j]; rng=local_rng, M=M_SOT)
                if dj < dist[j]
                    dist[j] = dj
                end
            else
                dist[j] = 0.0
            end
        end
    end


    # Assign measures
    assignments = zeros(Int, n_meas)    
    for _ in 1:itmax
        flag = false

        for idx in 1:n_meas
            μ = measures[idx]
            old_assign = assignments[idx]

            best_SOT = Inf
            for j in 1:k
                curr_dist = SOT(μ, centroids[j]; rng=local_rng, M=M_SOT)
                if curr_dist < best_SOT
                    best_SOT = curr_dist
                    assignments[idx] = j
                end
            end

            if !flag && old_assign != assignments[idx]
                flag = true
            end
        end

        # check no cluster is empty o/w fill it with the worst assignment
        counts = zeros(Int, k)
        @inbounds for j in 1:n_meas
            counts[assignments[j]] += 1
        end

        for c in 1:k
            if counts[c] == 0
                worst_dist = SOT(measures[1], centroids[assignments[1]]; rng=local_rng, M=M_SOT)
                worst_idx = 1
                for j in 2:n_meas
                    curr_dist = SOT(measures[j], centroids[assignments[j]]; rng=local_rng, M=M_SOT)
                    if curr_dist > worst_dist
                        worst_dist = curr_dist
                        worst_idx = j
                    end
                end
                
                assignments[worst_idx] = c
                centroids[c] = measures[worst_idx]

                flag = true
            end
        end

        if !flag
            return assignments, centroids
        end

        # Update centroids
        for j in 1:k
            centroids[j] = SWBarycenters_free_supp(measures[findall(assignments .== j)]; rng=local_rng, M=M_bary, itmax=itmax_bary, tol=tol_bary)
        end
    end

    return assignments, centroids
end