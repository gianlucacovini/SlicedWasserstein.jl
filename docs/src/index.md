# SlicedWasserstein.jl

`SlicedWasserstein.jl` is a lightweight Julia package for computations with
discrete measures using sliced optimal transport.

The package currently provides:

- a `DiscreteMeasure` type for finite weighted measures;
- one-dimensional optimal transport for discrete measures;
- Monte Carlo estimates of sliced Wasserstein costs;
- free-support sliced Wasserstein barycenters;
- sliced Wasserstein k-means clustering.

!!! note
    The package is currently released as `v0.1.0`. The public API is still
    evolving.

## Installation

Before registration, install the package directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/gianlucacovini/SlicedWasserstein.jl")
```

After registration, it can be installed with:

```julia
using Pkg
Pkg.add("SlicedWasserstein")
```

## Quick start

```@example quickstart
using SlicedWasserstein

μ = DiscreteMeasure(randn(2, 50))
ν = DiscreteMeasure(randn(2, 50) .+ 1)

SOT(μ, ν; M = 100, seed = 1)
```

## Contents

```@contents
Pages = ["index.md", "api.md"]
Depth = 2
```
