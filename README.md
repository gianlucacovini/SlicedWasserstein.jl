# SlicedWasserstein.jl

`SlicedWasserstein.jl` is a lightweight Julia package for computations with
finite discrete measures using sliced optimal transport.

The package currently provides:

- finite weighted discrete measures;
- one-dimensional optimal transport for discrete measures;
- Monte Carlo estimates of sliced Wasserstein costs;
- free-support sliced Wasserstein barycenters;
- sliced Wasserstein k-means clustering.

The package is currently in an early `v0.1` stage, and the public API may still
evolve.

## Installation

Before registration, install the package directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/gianlucacovini/SlicedWasserstein.jl")
```

After registration in the Julia General registry, it can be installed with:

```julia
using Pkg
Pkg.add("SlicedWasserstein")
```

## Quick start

```julia
using SlicedWasserstein

μ = DiscreteMeasure(randn(2, 50))
ν = DiscreteMeasure(randn(2, 50) .+ 1)

sot = SOT(μ, ν; M = 100, seed = 1)
```

## Documentation

The package includes Documenter-based documentation in the `docs/` directory.

To build the documentation locally, run:

```bash
julia --project=docs docs/make.jl
```

The documentation contains:

- a short background section on optimal transport and sliced Wasserstein methods;
- examples for the main functionality;
- an API reference for exported functions and types.

## Status

This package is under active development. It is intended for computations with
finite discrete measures and currently focuses on clarity, reproducibility, and
a small dependency footprint.

## Use of generative AI

Some documentation and repository-maintenance changes were prepared with
assistance from generative AI tools. All code, documentation, and package
metadata are reviewed, edited, and validated by the human maintainer, who takes
responsibility for the contents of the package.

## References

- G. Peyré and M. Cuturi. *Computational Optimal Transport*. Foundations and
  Trends in Machine Learning, 2019.
- K. Nguyen. *An Introduction to Sliced Optimal Transport*. Foundations and
  Trends in Computer Graphics and Vision, 2025.
- N. Bonneel, G. Peyré, M. Cuturi, et al. *Sliced and Radon Wasserstein
  Barycenters of Measures*. Journal of Mathematical Imaging and Vision, 2015.
