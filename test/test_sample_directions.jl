using Test
using Random
using SlicedWasserstein

@testset "Sample Directions" begin
    rng = Xoshiro(431943)
    M, d = 200, 5
    Z = sample_directions(M, d; rng=rng)

    @test size(Z) == (d, M)
    @test eltype(Z) == Float64

    colnorms = sqrt.(sum(abs2, Z; dims=1))
    @test all(isapprox.(vec(colnorms), 1.0; atol=1e-10))

    # Reproducibility with seed
    Z1 = sample_directions(50, 3; seed=7)
    Z2 = sample_directions(50, 3; seed=7)
    @test Z1 == Z2

    # Same RNG state => same draws
    rngA = Xoshiro(999)
    rngB = Xoshiro(999)
    Z3 = sample_directions(50, 3; rng=rngA)
    Z4 = sample_directions(50, 3; rng=rngB)
    @test Z3 == Z4

    # Seed overrides rng 
    rngX = Xoshiro(123)
    @test sample_directions(10, 3; rng=rngX, seed=7) == sample_directions(10, 3; seed=7)

    # Special cases
    Z5 = sample_directions(100, 1; seed=1)
    @test all(isapprox.(abs.(Z5), 1.0; atol=1e-15))

    Z0 = sample_directions(0, 5; seed=1)
    @test size(Z0) == (5, 0)

    # Argument errors
    @test_throws ArgumentError sample_directions(-1, 3; seed=1)
    @test_throws ArgumentError sample_directions(10, 0; seed=1)
end

nothing