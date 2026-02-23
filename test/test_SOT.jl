using Test
using SlicedWasserstein
using Random

@testset "SOT" begin
    @testset "throws on different total masses" begin
        rng = MersenneTwister(1)
        Xμ = rand(rng, 3, 5)
        Xν = rand(rng, 3, 5)
        μ = DiscreteMeasure(Xμ, fill(1.0, 5); normalize=false)
        ν = DiscreteMeasure(Xν, fill(0.9, 5); normalize=false)

        @test_throws ArgumentError SOT(μ, ν; M=10, seed=123)
    end

    @testset "throws on dimension mismatch" begin
        rng = MersenneTwister(10)
        μ = DiscreteMeasure(rand(rng, 3, 10))
        ν = DiscreteMeasure(rand(rng, 4, 10))
        @test_throws ArgumentError SOT(μ, ν; M=10, seed=1) 
    end

    @testset "deterministic when seed is fixed (up to floating error)" begin
        rng = MersenneTwister(2)
        d, n = 5, 50
        Xμ = rand(rng, d, n)
        Xν = rand(rng, d, n)
        wμ = rand(rng, n); wμ ./= sum(wμ)
        wν = rand(rng, n); wν ./= sum(wν)

        μ = DiscreteMeasure(Xμ, wμ)
        ν = DiscreteMeasure(Xν, wν)

        s1 = SOT(μ, ν; M=200, seed=777)
        s2 = SOT(μ, ν; M=200, seed=777)

        @test isapprox(s1, s2; rtol=1e-12, atol=1e-12)
    end

    @testset "identity: SOT(μ,μ) ≈ 0" begin
        rng = MersenneTwister(3)
        d, n = 4, 40
        X = rand(rng, d, n)
        w = rand(rng, n); w ./= sum(w)
        μ = DiscreteMeasure(X, w)

        s = SOT(μ, μ; M=200, seed=1234)
        @test isapprox(s, 0.0; atol=1e-10, rtol=0.0)
    end

    @testset "nonnegativity (squared cost)" begin
        rng = MersenneTwister(4)
        d, n = 4, 40
        Xμ = rand(rng, d, n)
        Xν = rand(rng, d, n)
        wμ = rand(rng, n); wμ ./= sum(wμ)
        wν = rand(rng, n); wν ./= sum(wν)

        μ = DiscreteMeasure(Xμ, wμ)
        ν = DiscreteMeasure(Xν, wν)

        s = SOT(μ, ν; M=200, seed=42)
        @test s ≥ -1e-10
    end

    @testset "d = 1 matches OT1d (squared cost)" begin
        rng = MersenneTwister(5)
        n = 30
        x = rand(rng, n)
        y = rand(rng, n)
        wμ = rand(rng, n); wμ ./= sum(wμ)
        wν = rand(rng, n); wν ./= sum(wν)

        μ = DiscreteMeasure(x, wμ)
        ν = DiscreteMeasure(y, wν)

        ot = OT1d(μ, ν)
        sot = SOT(μ, ν; M=500, seed=7)

        @test isapprox(sot, ot; atol=1e-8, rtol=1e-8)
    end

    @testset "M=0 should throw (recommended behavior)" begin
        rng = MersenneTwister(6)
        μ = DiscreteMeasure(rand(rng, 3, 10))
        ν = DiscreteMeasure(rand(rng, 3, 10))
        @test_throws ArgumentError SOT(μ, ν; M=0, seed=1)
    end
end

nothing