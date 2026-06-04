module PlotsExt

using SlicedWasserstein
using Plots

"""
    plot(μ::DiscreteMeasure; dims = (1, 2), markersize = 6, color = :blue, label = nothing, kwargs...)

Plots the discrete measure `μ` using scatter plot. The size of each marker is proportional to the corresponding weight.
"""
function Plots.plot(μ::DiscreteMeasure; dims = (1, 2), markersize = 6, color = :blue, label = nothing, kwargs...)
    X = μ.X
    d, n = size(X)

    if d < maximum(dims)
        throw(ArgumentError("Cannot plot dims=$dims for dimension d=$d"))
    end

    mass = sum(μ.w)
    (mass > 0) || throw(ArgumentError("Cannot plot a measure with zero total mass"))

    if length(dims) == 1
        scatter(
            X[dims[1], :],
            zeros(n);
            markersize = markersize .* μ.w .* n ./ mass,
            color = color,
            label = label,
            kwargs...
        )
    elseif length(dims) == 2
        scatter(
            X[dims[1], :],
            X[dims[2], :];
            markersize = markersize .* μ.w .* n ./ mass,
            color = color,
            label = label,
            aspect_ratio = :equal,
            kwargs...
        )
    else
        throw(ArgumentError("Can only plot 1D or 2D projections"))
    end
end

"""
    plot!(plt::Plots.Plot, μ::DiscreteMeasure; dims = (1, 2), markersize = 6, color = :blue, label = nothing, kwargs...)

Adds the discrete measure `μ` to an existing plot `plt` using scatter plot. The size of each marker is proportional to the corresponding weight.
"""
function Plots.plot!(plt::Plots.Plot, μ::DiscreteMeasure; dims=(1,2), markersize=6, color=:blue, label=nothing, kwargs...)
    X = μ.X
    d, n = size(X)
    d < maximum(dims) && throw(ArgumentError("Cannot plot dims=$dims for dimension d=$d"))

    mass = sum(μ.w)
    (mass > 0) || throw(ArgumentError("Cannot plot a measure with zero total mass"))

    ms = markersize .* μ.w .* n ./ mass

    if length(dims) == 1
        return scatter!(plt, X[dims[1], :], zeros(n); markersize=ms, color=color, label=label, kwargs...)
    elseif length(dims) == 2
        return scatter!(plt, X[dims[1], :], X[dims[2], :]; markersize=ms, color=color, label=label,
                        aspect_ratio=:equal, kwargs...)
    else
        throw(ArgumentError("Can only plot 1D or 2D projections"))
    end
end

end # module PlotsExt
