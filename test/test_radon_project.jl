using Test
using SlicedWasserstein
using LinearAlgebra

@testset "radon_project" begin
    @testset "dimension check" begin
        X = [0.0 1.0 2.0;
             0.0 0.0 0.0]
        w = [1.0, 1.0, 2.0]
        μ = DiscreteMeasure(X, w; normalize=false)

        θ_bad = [1.0, 0.0, 0.0]
        @test_throws ArgumentError radon_project(μ, θ_bad)

        buf = zeros(3)
        @test_throws ArgumentError radon_project!(buf, X, θ_bad)
    end

    @testset "correct projection values (2D)" begin
        X = [0.0 1.0 2.0;
             0.0 0.0 0.0]
        w = [1.0, 1.0, 2.0]              # sum=4 (unnormalized)
        μ = DiscreteMeasure(X, w; normalize=false)

        θx = [1.0, 0.0]
        νx = radon_project(μ, θx)

        @test size(νx.X) == (1, 3)
        @test vec(νx.X) ≈ [0.0, 1.0, 2.0]

        # mass preserved and NOT renormalized
        @test νx.w == μ.w
        @test sum(νx.w) ≈ sum(μ.w)

        # in-place matches
        buf = similar(vec(νx.X))
        radon_project!(buf, X, θx)
        @test buf ≈ vec(νx.X)
    end

    @testset "dot-product correctness (general θ)" begin
        X = [1.0  0.0 -1.0;
             0.0  2.0  0.0]
        w = [2.0, 1.0, 1.0]
        μ = DiscreteMeasure(X, w; normalize=false)

        θ = [2.0, -1.0]
        ν = radon_project(μ, θ)

        expected = [dot(θ, X[:, i]) for i in 1:3]
        @test vec(ν.X) ≈ expected
        @test ν.w == μ.w
    end

    @testset "type behavior (promotion sanity)" begin
        X32 = Float32[0 1 2; 0 0 0]
        w = [1.0, 1.0, 2.0]
        μ32 = DiscreteMeasure(X32, w; normalize=false)

        θ64 = [1.0, 0.0]  # Float64
        ν = radon_project(μ32, θ64)

        @test eltype(vec(ν.X)) == promote_type(eltype(X32), eltype(θ64))
    end
end

nothing