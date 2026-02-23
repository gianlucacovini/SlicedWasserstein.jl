module SlicedWasserstein

using LinearAlgebra
using Random
using Statistics
using StatsBase
using Plots

export DiscreteMeasure, sort_1d, print_full
export sample_directions, radon_project, radon_project!
export OT1d, OT1d_edge, SOT
export SWBarycenters_free_supp, grad
export SWk_means

include("types.jl")
include("projection.jl")
include("OT1d.jl")
include("SOT.jl")
include("SWBarycenters.jl")
include("SWk_means.jl")

end # module SlicedWasserstein