
"""
    OT1dResult{T<:Real}

Result returned by [`OT1d`](@ref).

# Fields
- `cost`: optimal transport cost, or `nothing` if not requested.
- `I`: source indices of each transported edge, or `nothing`.
- `J`: target indices of each transported edge, or `nothing`.
- `Tm`: transported masses per edge, or `nothing`.
"""
struct OT1dResult{T<:Real}
    cost::Union{T, Nothing}
    I::Union{Vector{Int}, Nothing}
    J::Union{Vector{Int}, Nothing}
    Tm::Union{Vector{T}, Nothing}
end

"""
    OT1d(μ, ν; cost=(x,y)->(x-y)^2, compute_cost=true, compute_edge=false)

Compute the optimal transport between 1-dimensional discrete measures `μ` and `ν`.

Returns an `OT1dResult` whose fields are `nothing` when not requested.
The edge-list fields `I`, `J`, `Tm` represent the transport plan: `Tm[k]` mass is moved
from support point `I[k]` of `μ` to support point `J[k]` of `ν`.

# Arguments
- `μ::DiscreteMeasure`: Source 1-dimensional measure.
- `ν::DiscreteMeasure`: Target 1-dimensional measure.
- `cost::F`: Cost function (default: squared distance).
- `compute_cost::Bool=true`: Compute the optimal transport cost.
- `compute_edge::Bool=false`: Compute the transport plan as an edge list.

# Returns
An `OT1dResult{T}` with:
- `cost`: optimal transport cost, or `nothing`.
- `I`, `J`: source/target indices of each transported edge, or `nothing`.
- `Tm`: transported masses per edge, or `nothing`.
"""
function OT1d(μ::DiscreteMeasure, ν::DiscreteMeasure;
              cost::F=(x,y)->(x-y)^2,
              compute_cost::Bool=true,
              compute_edge::Bool=false) where {F}
    ((size(μ.X,1)==1) && (size(ν.X,1)==1)) || throw(ArgumentError("μ and ν must be 1-dim"))
    (sum(μ.w) ≈ sum(ν.w)) || throw(ArgumentError("Different total masses"))
    (compute_cost || compute_edge) || throw(ArgumentError("At least one of compute_cost, compute_edge must be true"))

    n = size(μ.X, 2)
    m = size(ν.X, 2)
    (n > 0) || throw(ArgumentError("μ has no support points"))
    (m > 0) || throw(ArgumentError("ν has no support points"))

    μ_sorted, p1 = sort_1d(μ)
    ν_sorted, p2 = sort_1d(ν)

    S = promote_type(eltype(μ.w), Float64)
    ε = 100 * eps(eltype(μ.w)) * max(sum(μ.w), one(eltype(μ.w)))
    s = compute_cost ? zero(S) : nothing

    I_vec = Vector{Int}()
    J_vec = Vector{Int}()
    Tm_vec = Vector{S}()
    if compute_edge
        sizehint!(I_vec, n + m)
        sizehint!(J_vec, n + m)
        sizehint!(Tm_vec, n + m)
    end

    i = 1
    j = 1
    a = μ_sorted.w[1]
    b = ν_sorted.w[1]

    while (i ≤ n) && (j ≤ m)
        t = min(a, b)

        if compute_edge
            push!(I_vec, p1[i])
            push!(J_vec, p2[j])
            push!(Tm_vec, S(t))
        end

        if compute_cost
            s += t * cost(μ_sorted.X[1, i], ν_sorted.X[1, j])
        end

        a -= t
        b -= t

        if a ≤ ε
            i += 1
            i ≤ n && (a = μ_sorted.w[i])
        end

        if b ≤ ε
            j += 1
            j ≤ m && (b = ν_sorted.w[j])
        end
    end

    return OT1dResult{S}(
        s,
        compute_edge ? I_vec : nothing,
        compute_edge ? J_vec : nothing,
        compute_edge ? Tm_vec : nothing,
    )
end
