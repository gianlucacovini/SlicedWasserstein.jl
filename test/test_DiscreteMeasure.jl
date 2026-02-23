using Test
using Random
using SlicedWasserstein

@testset "DiscreteMeasure" begin
    @testset "2D constructor and normalization" begin
        X = [0.0 1.0 2.0;
             0.0 0.0 0.0]
        w = [1.0, 1.0, 2.0]

        μ = DiscreteMeasure(X, w)
        @test size(μ.X) == (2, 3)
        @test length(μ.w) == 3
        @test isapprox(sum(μ.w), 1.0; atol=1e-15)

        μ2 = DiscreteMeasure(X, w; normalize=false)
        @test isapprox(sum(μ2.w), 4.0; atol=1e-15)

        @test DiscreteMeasure(μ) === μ
    end

    @testset "Uniform weights constructors" begin
        X = randn(2, 10)
        μ = DiscreteMeasure(X)
        @test size(μ.X) == (2, 10)
        @test length(μ.w) == 10
        @test isapprox(sum(μ.w), 1.0; atol=1e-15)
        @test all(isapprox.(μ.w, 0.1; atol=1e-15))

        x = randn(7)
        μ1 = DiscreteMeasure(x)
        @test size(μ1.X) == (1, 7)
        @test length(μ1.w) == 7
        @test isapprox(sum(μ1.w), 1.0; atol=1e-15)
    end

    @testset "1D constructor with weights" begin
        x = [3.0, 1.0, 2.0]
        w = [0.2, 0.3, 0.5]
        μ = DiscreteMeasure(x, w; normalize=false)
        @test size(μ.X) == (1, 3)
        @test vec(μ.X) == x
        @test μ.w == w

        μn = DiscreteMeasure(x, w)
        @test isapprox(sum(μn.w), 1.0; atol=1e-15)
    end

    @testset "Matrix weights constructor" begin
        X = randn(2, 3)
        wrow = reshape([1.0, 1.0, 2.0], 1, :)
        wcol = reshape([1.0, 1.0, 2.0], :, 1)

        μr = DiscreteMeasure(X, wrow)
        μc = DiscreteMeasure(X, wcol)
        @test length(μr.w) == 3
        @test μr.w == μc.w

        @test_throws ArgumentError DiscreteMeasure(X, ones(2,2))
    end

    @testset "Type promotion" begin
        X32 = rand(Float32, 2, 5)
        w64 = ones(Float64, 5)

        μ = DiscreteMeasure(X32, w64)
        @test eltype(μ.X) == Float64
        @test eltype(μ.w) == Float64

        μu = DiscreteMeasure(X32)
        @test eltype(μu.X) == Float32
        @test eltype(μu.w) == Float32
    end

    @testset "convert" begin
        X = randn(2, 4)
        w = rand(4)
        μ = DiscreteMeasure(X, w; normalize=false)

        μ32 = convert(DiscreteMeasure{Float32}, μ)
        @test eltype(μ32.X) == Float32
        @test eltype(μ32.w) == Float32
        @test sum(μ32.w) ≈ sum(μ.w)
    end

    @testset "Errors" begin
        X = randn(2, 3)
        @test_throws ArgumentError DiscreteMeasure(X, [1.0, 2.0])
        @test_throws ArgumentError DiscreteMeasure(X, [0.0, 0.0, 0.0])
        @test_throws ArgumentError DiscreteMeasure(X, [-1.0, 0.0, 0.0])
    end
end

nothing