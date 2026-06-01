using Test
using Random
using SlicedWasserstein
using StatsBase

@testset "SWk_means tests" begin

    @testset "basic 2-cluster separation" begin
        rng = Xoshiro(37)
        d = 2

        measures = Vector{DiscreteMeasure{Float64}}(undef, 6)
        for i in 1:3
            X = 0.05 * randn(rng, d, 6) .+ 0.01 * rand(rng) 
            measures[i] = DiscreteMeasure(X)
        end
        for i in 4:6
            X = 0.05 * randn(rng, d, 6) .+ 5.0 .+ 0.01 * rand(rng)  
            measures[i] = DiscreteMeasure(X)
        end

        assignments, centroids = SWk_means(
            measures, 2;
            itmax=10, M_SOT=200, M_bary=200, itmax_bary=20, seed=1234
        )

        @test length(assignments) == length(measures)
        @test !(all(assignments .== assignments[1]))  # not all in same cluster
        @test length(centroids) == 2
        @test all(c isa DiscreteMeasure for c in centroids)
    end

    @testset "reproducibility with seed" begin
        rng = Xoshiro(123)
        d = 2

        measures = Vector{DiscreteMeasure{Float64}}(undef, 4)
        for i in 1:4
            X = 0.1 * randn(rng, d, 5) .+ (i <= 2 ? 0.0 : 3.0)  # two groups
            measures[i] = DiscreteMeasure(X)
        end

        assignments1, centroids1 = SWk_means(
            measures, 2;
            itmax=8, M_SOT=100, M_bary=100, itmax_bary=10, seed=2026
        )
        assignments2, centroids2 = SWk_means(
            measures, 2;
            itmax=8, M_SOT=100, M_bary=100, itmax_bary=10, seed=2026
        )

        @test assignments1 == assignments2
        @test length(centroids1) == length(centroids2) == 2

        # Compare centroid means computed from support + weights
        for i in 1:2
            @test size(centroids1[i].X) == size(centroids2[i].X)
            m1 = mean(centroids1[i])
            m2 = mean(centroids2[i])
            @test isapprox(m1, m2; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "K = 1 (single cluster) behaviour" begin
        rng = Xoshiro(7)
        d = 3
        measures = [DiscreteMeasure(0.1 * randn(rng, d, 5) .+ i) for i in 1:3]

        assignments, centroids = SWk_means(
            measures, 1;
            itmax=5, M_SOT=80, M_bary=80, itmax_bary=8, seed=9
        )

        @test length(assignments) == length(measures)
        @test all(assignments .== 1)
        @test length(centroids) == 1
        @test centroids[1] isa DiscreteMeasure
    end

    @testset "K equals number of measures - loose check" begin
        rng = Xoshiro(55)
        d = 2
        N = 4
        measures = [DiscreteMeasure(rand(rng, d, 6) .+ i) for i in 1:N]

        assignments, centroids = SWk_means(
            measures, N;
            itmax=6, M_SOT=80, M_bary=80, itmax_bary=8, seed=42
        )

        @test length(assignments) == N
        @test length(centroids) == N
        @test length(unique(assignments)) ≥ 2
    end

    @testset "output shapes and types" begin
        rng = Xoshiro(101)
        d = 2
        measures = [DiscreteMeasure(rand(rng, d, 4)) for _ in 1:5]

        assignments, centroids = SWk_means(
            measures, 3;
            itmax=4, M_SOT=60, M_bary=60, itmax_bary=6, seed=11
        )

        @test assignments isa Vector{Int}
        @test length(assignments) == length(measures)
        @test centroids isa Vector{<:DiscreteMeasure}
        @test length(centroids) == 3
        for c in centroids
            @test c.X isa AbstractArray
            @test c.w isa AbstractVector
        end
    end

    @testset "dispatch: Vector{DiscreteMeasure{Float64}} vs Vector{DiscreteMeasure}" begin
        rng = Xoshiro(202)
        d = 2

        measures_typed = Vector{DiscreteMeasure{Float64}}(undef, 3)
        for i in 1:3
            measures_typed[i] = DiscreteMeasure(rand(rng, d, 5))
        end

        measures_untyped = Vector{DiscreteMeasure}(undef, 3)
        for i in 1:3
            measures_untyped[i] = measures_typed[i]
        end

        @test begin
            SWk_means(measures_typed, 2; itmax=2, M_SOT=20, M_bary=20, itmax_bary=2, seed=1)
            true
        end

        @test begin
            SWk_means(measures_untyped, 2; itmax=2, M_SOT=20, M_bary=20, itmax_bary=2, seed=1)
            true
        end
    end

end

nothing
