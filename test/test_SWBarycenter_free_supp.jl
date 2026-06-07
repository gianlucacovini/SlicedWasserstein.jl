using Test
using Random
using LinearAlgebra
using StatsBase

using SlicedWasserstein

@testset "SWBarycenters Fixed Support" begin
    @testset "SWBarycenters_free_supp argument validation" begin
        rng = Xoshiro(0)
        μ = DiscreteMeasure(randn(rng, 2, 10))
        ν = DiscreteMeasure(randn(rng, 2, 10))

        # wrong length
        @test_throws ArgumentError SWBarycenters_free_supp([μ, ν]; w=[1.0])

        # does not sum to 1
        @test_throws ArgumentError SWBarycenters_free_supp([μ, ν]; w=[0.2, 0.2])

        # correct — light run
        bar = SWBarycenters_free_supp([μ, ν]; w=[0.25, 0.75], itmax=1, M=10, rng=rng)
        @test size(bar.X, 1) == 2
        @test isapprox(sum(bar.w), 1.0; atol=1e-10)
    end

    @testset "SWBarycenters_free_supp reproducibility with seed" begin
        rng = Xoshiro(999)

        X1 = randn(rng, 2, 40) .+ [-2.0; 0.0]
        X2 = randn(rng, 2, 40) .+ [ 2.0; 0.0]
        μ1 = DiscreteMeasure(X1)
        μ2 = DiscreteMeasure(X2)

        # small M and small itmax to keep test fast but deterministic
        b1 = SWBarycenters_free_supp([μ1, μ2];
            w=[0.5, 0.5], n_supp=25, itmax=3, tol=0.0, M=30,
            rng=rng, seed=1234
        )
        b2 = SWBarycenters_free_supp([μ1, μ2];
            w=[0.5, 0.5], n_supp=25, itmax=3, tol=0.0, M=30,
            rng=rng, seed=1234
        )

        # use approximate equality — threading/FP may create tiny differences
        @test isapprox(b1.X, b2.X; atol=1e-10, rtol=1e-12)
        @test isapprox(b1.w, b2.w; atol=1e-12, rtol=0)
    end

    @testset "SWBarycenters_free_supp smoke test" begin
        rng = Xoshiro(2024)

        X1 = randn(rng, 2, 80) .+ [-2.0; 0.0]
        X2 = randn(rng, 2, 80) .+ [ 2.0; 0.0]
        μ1 = DiscreteMeasure(X1)
        μ2 = DiscreteMeasure(X2)

        # modest M and itmax for CI; increase M for more accurate runs
        bar = SWBarycenters_free_supp([μ1, μ2];
            η=0.01, itmax=10, tol=0.0, M=50,
            rng=rng, normalize=true
        )

        @test size(bar.X, 1) == 2
        @test size(bar.X, 2) >= 1
        @test isapprox(sum(bar.w), 1.0; atol=1e-10)
    end

    @testset "Identity: barycenter of identical measures" begin
        rng = Xoshiro(0)
        X = randn(rng, 2, 50)
        μ = DiscreteMeasure(X)

        bar = SWBarycenters_free_supp(
            [μ, μ, μ]; w=[1/3, 1/3, 1/3], itmax=500, tol=1e-12,
            M=500, rng=rng, seed=42
        )
        @test size(bar.X) == size(μ.X)
        @test SOT(bar, μ; M=1_000, rng=rng, seed=999) ≤ 1e-2
    end

    @testset "Dirac measures: barycenter = weighted mean (analytic)" begin
        rng = Xoshiro(1)

        x1 = [-2.0, 0.0]
        x2 = [ 3.0, 1.0]
        μ1 = DiscreteMeasure(reshape(x1, :, 1))
        μ2 = DiscreteMeasure(reshape(x2, :, 1))

        w = [0.25, 0.75]
        target = w[1]*x1 .+ w[2]*x2

        bar = SWBarycenters_free_supp([μ1, μ2];
            w=w, n_supp=1, η=0.5, itmax=50, tol=1e-12,
            M=200, rng=rng, seed=7
        )

        @test size(bar.X) == (2, 1)
        @test isapprox(vec(bar.X), target; atol=1e-2)
    end

    @testset "Symmetry: mirrored measures -> centered barycenter" begin
        rng = Xoshiro(2024)
        X1 = randn(rng, 2, 200) .+ [-2.0; 0.0]
        X2 = randn(rng, 2, 200) .+ [ 2.0; 0.0]
        μ1 = DiscreteMeasure(X1)
        μ2 = DiscreteMeasure(X2)

        bar = SWBarycenters_free_supp([μ1, μ2];
            w=[0.5, 0.5], n_supp=50, η=0.05, itmax=40, tol=1e-12,
            M=300, rng=rng, seed=11
        )

        m = vec(mean(bar.X; dims=2))
        @test abs(m[1]) ≤ 0.2
        @test abs(m[2]) ≤ 0.2
    end

    @testset "Weight influence: barycenter shifts toward heavier measure" begin
        rng = Xoshiro(10)
        X1 = randn(rng, 2, 150) .+ [-3.0; 0.0]
        X2 = randn(rng, 2, 150) .+ [ 3.0; 0.0]
        μ1 = DiscreteMeasure(X1)
        μ2 = DiscreteMeasure(X2)

        barA = SWBarycenters_free_supp([μ1, μ2];
            w=[0.8, 0.2], n_supp=40, η=0.05, itmax=30, tol=1e-12,
            M=200, rng=rng, seed=5
        )
        barB = SWBarycenters_free_supp([μ1, μ2];
            w=[0.2, 0.8], n_supp=40, η=0.05, itmax=30, tol=1e-12,
            M=200, rng=rng, seed=5
        )

        mxA = mean(barA.X[1, :])
        mxB = mean(barB.X[1, :])

        @test mxA < mxB
    end

    @testset "Objective decreases with more iterations (approx)" begin
        rng = Xoshiro(77)
        X1 = randn(rng, 2, 120) .+ [-2.0; 0.0]
        X2 = randn(rng, 2, 120) .+ [ 2.0; 0.0]
        μ1 = DiscreteMeasure(X1)
        μ2 = DiscreteMeasure(X2)
        w = [0.5, 0.5]

        # light runs for CI; increase M for stronger guarantee outside CI
        bar1 = SWBarycenters_free_supp([μ1, μ2];
            w=w, n_supp=30, η=0.05, itmax=1, tol=0.0,
            M=150, rng=rng, seed=9
        )
        bar10 = SWBarycenters_free_supp([μ1, μ2];
            w=w, n_supp=30, η=0.05, itmax=10, tol=0.0,
            M=150, rng=rng, seed=9
        )

        J(b) = w[1]*SOT(b, μ1; M=300, rng=rng, seed=123) + w[2]*SOT(b, μ2; M=300, rng=rng, seed=123)

        @test J(bar10) ≤ J(bar1) + 1e-5
    end
end

nothing