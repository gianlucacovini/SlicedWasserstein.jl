using Test
using SlicedWasserstein

@testset "OT1d" begin
    @testset "basic value (squared cost)" begin
        μ = DiscreteMeasure([0.0, 1.0], [0.5, 0.5])
        ν = DiscreteMeasure([0.0, 2.0], [0.5, 0.5])
        @test OT1d(μ, ν) ≈ 0.5
    end

    @testset "invariance to input ordering" begin
        μ1 = DiscreteMeasure([0.0, 1.0, 3.0], [0.2, 0.5, 0.3]; normalize=false)
        ν1 = DiscreteMeasure([0.0, 2.0, 4.0], [0.2, 0.5, 0.3]; normalize=false)

        μ2 = DiscreteMeasure([3.0, 0.0, 1.0], [0.3, 0.2, 0.5]; normalize=false)
        ν2 = DiscreteMeasure([4.0, 2.0, 0.0], [0.3, 0.5, 0.2]; normalize=false)

        @test OT1d(μ1, ν1) ≈ OT1d(μ2, ν2)
    end

    @testset "custom cost" begin
        μ = DiscreteMeasure([0.0, 1.0], [0.5, 0.5])
        ν = DiscreteMeasure([0.0, 2.0], [0.5, 0.5])
        c = (x, y) -> abs(x - y)
        @test OT1d(μ, ν; cost=c) ≈ 0.5
    end

    @testset "unnormalized equal mass works" begin
        μ = DiscreteMeasure([0.0, 1.0], [1.0, 1.0]; normalize=false)      # mass 2
        ν = DiscreteMeasure([0.0, 2.0], [0.5, 1.5]; normalize=false)      # mass 2
        @test OT1d(μ, ν) ≥ 0
    end

    @testset "compute_plan returns consistent plan (unsorted inputs)" begin
        # Deliberately unsorted supports to catch plan reindexing bugs
        μ = DiscreteMeasure([1.0, 0.0], [0.25, 0.75]; normalize=false)
        ν = DiscreteMeasure([2.0, 0.0], [0.50, 0.50]; normalize=false)

        s, P = OT1d(μ, ν; compute_plan=true)

        @test size(P) == (2, 2)
        @test sum(P) ≈ sum(μ.w) ≈ sum(ν.w)
        @test all(P .>= 0)

        @test vec(sum(P, dims=2)) ≈ μ.w
        @test vec(sum(P, dims=1)) ≈ ν.w

        cost = (x, y) -> (x - y)^2
        s_from_P = 0.0
        for i in 1:2, j in 1:2
            s_from_P += P[i, j] * cost(μ.X[1, i], ν.X[1, j])
        end
        @test s ≈ s_from_P
    end

    @testset "OT1d_edge matches OT1d and has correct marginals" begin
        μ = DiscreteMeasure([3.0, 0.0, 1.0], [0.3, 0.2, 0.5]; normalize=false)
        ν = DiscreteMeasure([4.0, 2.0, 0.0], [0.3, 0.5, 0.2]; normalize=false)

        s = OT1d(μ, ν)
        s2, I, J, Tm = OT1d_edge(μ, ν; compute_cost=true, compute_edge=true)

        @test s2 ≈ s
        @test length(I) == length(J) == length(Tm)
        @test all(Tm .>= 0)
        @test sum(Tm) ≈ sum(μ.w) ≈ sum(ν.w)

        margμ = zeros(length(μ.w))
        margν = zeros(length(ν.w))
        for k in eachindex(Tm)
            margμ[I[k]] += Tm[k]
            margν[J[k]] += Tm[k]
        end
        @test margμ ≈ μ.w
        @test margν ≈ ν.w
    end

    @testset "errors" begin
        μ = DiscreteMeasure([0.0, 1.0], [0.5, 0.5])
        ν_bad_mass = DiscreteMeasure([0.0, 2.0], [0.25, 0.25]; normalize=false)
        @test_throws ArgumentError OT1d(μ, ν_bad_mass)

        μ2d = DiscreteMeasure([0.0 1.0; 0.0 1.0], [0.5, 0.5])
        ν = DiscreteMeasure([0.0, 2.0], [0.5, 0.5])
        @test_throws ArgumentError OT1d(μ2d, ν)
    end

    @testset "Float32 behavior" begin
        μ = DiscreteMeasure(Float32[0,1], Float32[0.5,0.5])
        ν = DiscreteMeasure(Float32[0,2], Float32[0.5,0.5])
        @test OT1d(μ, ν) ≈ 0.5
    end
end

nothing