
"""
    OT1d(μ::DiscreteMeasure, ν::DiscreteMeasure; cost::F= (x, y) -> (x - y)^2, compute_plan::Bool=false) where {F}

Compute the optimal transport cost between two 1-dimensional discrete measures `μ` and `ν`.
If `compute_plan` is true, also return the optimal transport plan as a matrix (intended for small measures; for large measures use OT1d_edge).

# Arguments
- `μ::DiscreteMeasure`: The source discrete measure.
- `ν::DiscreteMeasure`: The target discrete measure.
- `cost::F`: A cost function that takes two arguments (default is squared Euclidean
    distance).
- `compute_plan::Bool`: Whether to compute and return the optimal transport plan (default is false).

# Returns
- If `compute_plan` is false, returns the optimal transport cost as a scalar.
- If `compute_plan` is true, returns a tuple `(cost, P)` where `cost` is the optimal transport cost and `P` is the optimal transport plan matrix.
"""
function OT1d(μ::DiscreteMeasure, ν::DiscreteMeasure; cost::F= (x, y) -> (x - y)^2, compute_plan::Bool=false) where {F}
    ((size(μ.X, 1) == 1) && (size(ν.X, 1) == 1)) || throw(ArgumentError("μ and ν must be 1-dim"))
    (sum(μ.w) ≈ sum(ν.w)) || throw(ArgumentError("Different total masses"))

    n = size(μ.X, 2)
    m = size(ν.X, 2)

    (n > 0) || throw(ArgumentError("μ has no support points"))
    (m > 0) || throw(ArgumentError("ν has no support points"))

    μ_sorted, p1 = sort_1d(μ)
    ν_sorted, p2 = sort_1d(ν)

    s = zero(promote_type(eltype(μ.w), Float64))

    i = 1
    j = 1
    a = μ_sorted.w[1]
    b = ν_sorted.w[1]

    ε = 100 * eps(eltype(a)) * max(sum(μ.w), one(eltype(μ.w)))

    if compute_plan
        P = zeros(Float64, n, m)
    end


    while (i ≤ n) && (j ≤ m)
        t = min(a, b)
        if compute_plan
            P[i, j] = t
        end
        s += t*cost(μ_sorted.X[1,i], ν_sorted.X[1,j])
        a = a - t
        b = b - t
        
        if a ≤ ε
            i += 1
            if i ≤ n
                a = μ_sorted.w[i]
            end
        end

        if b ≤ ε
            j += 1
            if j ≤ m
                b = ν_sorted.w[j]
            end
        end
    end

    if compute_plan
        P_orig = zeros(eltype(P), n, m)

        @inbounds for i in 1:n, j in 1:m
            t = P[i,j]

            if t != 0
                P_orig[p1[i], p2[j]] = t
            end
        end

        return s, P_orig
    end

    return s
end

"""
    OT1d_edge(μ::DiscreteMeasure, ν::DiscreteMeasure; cost::F= (x, y) -> (x - y)^2, compute_cost::Bool=true, compute_edge::Bool=true) where {F}

Compute the optimal transport cost and/or transport plan between two 1-dimensional discrete measures `μ` and `ν` using an edge list representation.

# Arguments
- `μ::DiscreteMeasure`: The source discrete measure.
- `ν::DiscreteMeasure`: The target discrete measure.
- `cost::F`: A cost function that takes two arguments (default is squared Euclidean distance).
- `compute_cost::Bool`: Whether to compute and return the optimal transport cost (default is true).
- `compute_edge::Bool`: Whether to compute and return the optimal transport plan as an edge list (default is true).

# Returns
- If both `compute_cost` and `compute_edge` are true, returns a tuple `(cost, I, J, Tm)` where `cost` is the optimal transport cost, `I` and `J` are vectors of indices representing the source and target points, and `Tm` is a vector of transported masses.
- If only `compute_edge` is true, returns the edge list as `(I, J, Tm)`.
- If only `compute_cost` is true, returns the optimal transport cost as a scalar.
"""
function OT1d_edge(μ::DiscreteMeasure, ν::DiscreteMeasure; cost::F= (x, y) -> (x - y)^2, compute_cost::Bool=true, compute_edge::Bool=true) where {F}
    (compute_cost || compute_edge) || throw(ArgumentError("At least one between compute_cost and compute_edge must be true"))
    ((size(μ.X, 1) == 1) && (size(ν.X, 1) == 1)) || throw(ArgumentError("μ and ν must be 1-dim"))
    (sum(μ.w) ≈ sum(ν.w)) || throw(ArgumentError("Different total masses"))

    ε = 100 * eps(eltype(μ.w))

    n = size(μ.X, 2)
    m = size(ν.X, 2)

    (n > 0) || throw(ArgumentError("μ has no support points"))
    (m > 0) || throw(ArgumentError("ν has no support points"))

    μ_sorted, p1 = sort_1d(μ)
    ν_sorted, p2 = sort_1d(ν)

    i = 1
    j = 1
    a = μ_sorted.w[1]
    b = ν_sorted.w[1]

    ε = 100 * eps(eltype(a)) * max(sum(μ.w), one(eltype(μ.w)))

    if compute_cost
        s = zero(promote_type(eltype(μ.w), Float64))
    end

    if compute_edge
        I = Vector{Int}()
        J = Vector{Int}()
        Tm = Vector{eltype(μ.w)}()

        sizehint!(I, n+m)
        sizehint!(J, n+m)
        sizehint!(Tm, n+m)
    end

    while (i ≤ n) && (j ≤ m)
        t = min(a, b)
        if compute_edge
            push!(I, p1[i])
            push!(J, p2[j])
            push!(Tm, t)
        end

        if compute_cost
            s += t*cost(μ_sorted.X[1,i], ν_sorted.X[1,j])
        end

        a -= t
        b -= t
        
        if a ≤ ε
            i += 1
            if i ≤ n
                a = μ_sorted.w[i]
            end
        end

        if b ≤ ε
            j += 1
            if j ≤ m
                b = ν_sorted.w[j]
            end
        end
    end

    if compute_cost && compute_edge
        return  s, I, J, Tm
    elseif compute_edge
        return I, J, Tm
    elseif compute_cost
        return s
    end
end