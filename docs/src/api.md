# API Reference

```@meta
CurrentModule = SlicedWasserstein
```

## Measures

```@docs
DiscreteMeasure
sort_1d
```

## Projections

```@docs
sample_directions
radon_project
radon_project!
```

## One-dimensional optimal transport

```@docs
OT1d
```

`OT1d` returns an `OT1dResult` with fields `cost`, `I`, `J`, and `Tm`.

## Sliced optimal transport

```@docs
SOT
```

## Barycenters

```@docs
grad
SWBarycenters_free_supp
```

## Clustering

```@docs
SWk_means
```

## Display helpers

`print_full(μ)` prints a `DiscreteMeasure` without truncating its support.

`print_full(μs)` prints a vector of discrete measures without truncation.
