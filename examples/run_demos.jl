using Plots

outdir = joinpath(@__DIR__, "plots")
mkpath(outdir)

for file in readdir(@__DIR__)
    if startswith(file, "demo_") && endswith(file, ".jl")
        include(file)
        savefig(joinpath(outdir, replace(file, ".jl" => ".png")))
    end
end

nothing