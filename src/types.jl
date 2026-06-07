import Base: convert, show, sort, summary
import Statistics: mean

"""
    DiscreteMeasure(X, w; normalize=true)
    DiscreteMeasure(X)

A discrete measure supported on a finite set of points.

# Representation
- `X::Matrix{T}`: a `d × n` matrix where each column represents a point in `ℝ^d`.
- `w::Vector{T}`: a vector of length `n` representing the weights associated with each point.
- `T<:Real`: the numeric type of the points and weights.
- When `w` is provided, weights are normalized to sum to 1 by default; set `normalize=false` to disable. When `w` is omitted, uniform weights are used (already normalized).

# Examples
```julia
X = [0.0 1.0 2.0;
     0.0 0.0 0.0]
w = [1.0, 1.0, 2.0]
μ = DiscreteMeasure(X, w)  # creates a discrete measure with normalized weights
μ_uniform = DiscreteMeasure(X)  # creates a discrete measure with uniform weights
```
"""
struct DiscreteMeasure{T<:Real}
    X::Matrix{T}
    w::Vector{T}

    function DiscreteMeasure{T}(X::AbstractMatrix{T}, w::AbstractVector{T}; normalize::Bool=true) where {T<:Real}
        (size(X, 2) ≥ 1) || throw(ArgumentError("The measure must contain at least one point"))
        (size(X, 2) == length(w)) || throw(ArgumentError("The number of points must be the same as the number of weights w"))
        all(isfinite, w) || throw(ArgumentError("Weights must be finite"))
        all(w .>= 0) || throw(ArgumentError("Weights must be nonnegative"))

        Xc = Matrix{T}(X)
        wc = Vector{T}(w)

        if normalize
            s = sum(wc)
            (s > 0) || throw(ArgumentError("The sum of the weights must be positive"))
            wc ./= s
        end

        return new(Xc, wc)
    end
end

function DiscreteMeasure(X::AbstractMatrix{TX}, w::AbstractVector{TW}; normalize::Bool=true) where {TX<:Real, TW<:Real}
    T = promote_type(TX, TW)
    return DiscreteMeasure{T}(T.(X), T.(w); normalize=normalize)
end

function DiscreteMeasure(X::AbstractMatrix{T}) where {T <: Real}
    n = size(X,2)
    w = fill(one(T)/n, n)
    return DiscreteMeasure(X, w; normalize=false)
end

function DiscreteMeasure(X::AbstractVector{TX}, w::AbstractVector{TW}; normalize::Bool=true) where {TX <: Real, TW <: Real}
    X_mat = reshape(X, 1, :)
    return DiscreteMeasure(X_mat, w; normalize=normalize)
end

function DiscreteMeasure(X::AbstractMatrix{TX}, w::AbstractMatrix{TW}; normalize::Bool=true) where {TX <: Real, TW <: Real }
    ((size(w, 1) == 1) || (size(w, 2) == 1)) || throw(ArgumentError("The set of weights w must be a 1dim"))

    w_vec = vec(w)
    return DiscreteMeasure(X, w_vec; normalize=normalize)
end

function DiscreteMeasure(X::AbstractVector{T}) where {T <: Real}
    return DiscreteMeasure(reshape(X, 1, :))
end

DiscreteMeasure(μ::DiscreteMeasure) = μ

function convert(::Type{DiscreteMeasure{T}}, μ::DiscreteMeasure) where {T <: Real}
    return DiscreteMeasure(T.(μ.X), T.(μ.w); normalize=false)
end

round3(x) = round(x; digits=3)

function show(io::IO, μ::DiscreteMeasure)
    n = size(μ.X, 2)
    println(io, "DiscreteMeasure($(size(μ.X,1))D, n=$n)")

    # :limit=true means "summarize/truncate"
    limit = get(io, :limit, true)

    if !limit || n ≤ 10
        iofull = IOContext(io, :limit => false)
        print(io, "  X = ")
        show(iofull, "text/plain", round3.(μ.X))
        println(io)
        print(io, "  w = ")
        show(iofull, "text/plain", round3.(μ.w'))
        println(io)
    else
        println(io, "  support size: ", size(μ.X,1), " × ", n)
        println(io, "  weights: min=$(round3(minimum(μ.w))), max=$(round3(maximum(μ.w)))")
    end
end

"""
    print_full(μ)
    print_full(μs)

Print a `DiscreteMeasure` (or a vector of them) without truncating the support.
"""
print_full(μ::DiscreteMeasure) = show(IOContext(stdout, :limit => false), μ)
print_full(io::IO, μ::DiscreteMeasure) = show(IOContext(io, :limit => false), μ)

function print_full(io::IO, μs::AbstractVector{<:DiscreteMeasure})
    for (k, μ) in pairs(μs)
        k == firstindex(μs) || println(io)
        println(io, "[$k]")
        print_full(io, μ)
    end
end

print_full(μs::AbstractVector{<:DiscreteMeasure}) = print_full(stdout, μs)


"""
    sort_1d(μ::DiscreteMeasure)

Sorts a 1-dimensional discrete measure `μ` in ascending order of its support points.
Returns a tuple `(μ_sorted, p)` where `μ_sorted` is the sorted discrete measure and `p` is the permutation vector such that `μ_sorted.X = μ.X[:, p]`.
"""
function sort_1d(μ::DiscreteMeasure)
    (size(μ.X, 1) == 1) || throw(ArgumentError("μ must be 1-dim"))

    x = vec(μ.X)

    p = sortperm(vec(μ.X))
    x = vec(μ.X)[p]
    w = μ.w[p]
    return DiscreteMeasure(reshape(x, 1, :), w; normalize=false), p
end

"""
    mean(μ::DiscreteMeasure)

Computes the weighted mean of the discrete measure `μ`.
"""
mean(μ::DiscreteMeasure) = μ.X * μ.w
