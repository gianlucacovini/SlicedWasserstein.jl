using Test

@testset "All tests" begin
    for file in readdir(@__DIR__)
        if startswith(file, "test_") && endswith(file, ".jl")
            include(file)
        end
    end
end

nothing