
begin
    for file in readdir(@__DIR__)
        if startswith(file, "bm_") && endswith(file, ".jl")
            include(file)
        end
    end
end

nothing