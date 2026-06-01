using BenchmarkTools
using Random
using SlicedWasserstein

M = 1_000
d = 2
rng = Xoshiro(431943)

println("Detailed benchmark (M=$M, d=$d):")
b = @benchmark sample_directions($M, $d; rng=$rng)
display(b)

println("Scaling checks (M=$M):")
println("d=2:")
@btime sample_directions($M, 2; rng=$rng)
println("d=10:")
@btime sample_directions($M, 10; rng=$rng)
println("d=100:")
@btime sample_directions($M, 100; rng=$rng)
println("d=500:")
@btime sample_directions($M, 500; rng=$rng)

println("Scaling checks (d=$d):")
println("M=500:")
@btime sample_directions(500, $d; rng=$rng)
println("M=1000:")
@btime sample_directions(1000, $d; rng=$rng)
println("M=5000:")
@btime sample_directions(5000, $d; rng=$rng)
println("M=10000:")
@btime sample_directions(10000, $d; rng=$rng)

nothing