# Background and methodology

This page gives a short overview of the mathematical ideas behind
`SlicedWasserstein.jl`.

## Optimal transport

Optimal transport provides a framework for comparing probability measures by
quantifying the minimal cost required to transport mass from one distribution to
another.

Given two probability measures ``\mu`` and ``\nu`` on ``\mathbb{R}^d`` with equal
total mass, the Kantorovich formulation of optimal transport is

```math
\pi^\star =
\arg\min_{\pi \in \Pi(\mu,\nu)}
\int_{\mathbb{R}^d \times \mathbb{R}^d} c(x,y) \, d\pi(x,y),
```

where ``\Pi(\mu,\nu)`` is the set of couplings with marginals ``\mu`` and
``\nu``, and ``c(x,y)`` is a cost function, typically
``c(x,y) = \|x-y\|^2``.

The quadratic Wasserstein cost is

```math
W_2^2(\mu,\nu)
=
\min_{\pi \in \Pi(\mu,\nu)}
\int \|x-y\|^2 \, d\pi(x,y).
```

Although optimal transport has appealing theoretical properties, computing it in
high dimension can be expensive.

## Sliced optimal transport

Sliced optimal transport reduces a high-dimensional optimal transport problem to
a collection of one-dimensional problems.

Given a direction ``\theta \in \mathbb{S}^{d-1}``, let
``\mathcal{R}_\theta \mu`` denote the projection of ``\mu`` onto the line
spanned by ``\theta``. The squared sliced Wasserstein cost is

```math
\mathrm{SOT}(\mu,\nu)
=
\mathbb{E}_{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}
\left[
W_2^2\bigl(\mathcal{R}_\theta \mu,\mathcal{R}_\theta \nu\bigr)
\right].
```

In practice, the expectation is approximated by Monte Carlo sampling:

```math
\mathrm{SOT}_M(\mu,\nu)
=
\frac{1}{M}
\sum_{m=1}^M
W_2^2\bigl(\mathcal{R}_{\theta_m}\mu,
           \mathcal{R}_{\theta_m}\nu\bigr),
\qquad
\theta_m \sim \mathcal{U}(\mathbb{S}^{d-1}).
```

For each sampled direction, the package:

1. projects both discrete measures onto a line;
2. solves a one-dimensional optimal transport problem;
3. averages the resulting costs.

The function `SOT` returns the squared sliced Wasserstein cost, not its square
root.

## Discrete measures

All measures in this package are represented as finite discrete measures. A
measure supported on ``n`` points in ``\mathbb{R}^d`` is written as

```math
\mu = \sum_{i=1}^n w_i \delta_{x_i},
```

where ``x_i \in \mathbb{R}^d`` are support points and ``w_i \ge 0`` are weights.

In the implementation, a discrete measure is represented by:

- a matrix `X` of size `d × n`, whose columns are the support points;
- a vector `w` of length `n`, containing the weights.

See [`DiscreteMeasure`](@ref) for the API.

## One-dimensional optimal transport

For one-dimensional discrete measures, the optimal transport problem can be
solved efficiently after sorting the support points.

`SlicedWasserstein.jl` implements this through `OT1d`, using a cumulative mass
matching algorithm. When requested, the transport plan is returned in an
edge-list representation rather than as a dense matrix.

This is useful for gradient computations and avoids storing large dense
transport matrices.

## Sliced Wasserstein barycenters

A barycenter is a measure that summarizes a collection of input measures.
Given measures ``\mu_1,\dots,\mu_K`` and weights ``w_1,\dots,w_K`` with
``\sum_k w_k = 1``, a sliced Wasserstein barycenter solves

```math
\mu^\star =
\arg\min_{\mu}
\sum_{k=1}^K
w_k \mathrm{SOT}(\mu,\mu_k).
```

The package implements free-support sliced Wasserstein barycenters for discrete
measures. In this setting, the barycenter is parametrized by support points:

```math
\mu_\phi =
\frac{1}{n}
\sum_{i=1}^n \delta_{\phi_i}.
```

The support locations are optimized by a stochastic gradient descent scheme.
Because the optimization problem is generally non-convex, results may depend on
initialization and hyperparameters.

See [`SWBarycenters_free_supp`](@ref).

## Sliced Wasserstein k-means

Sliced Wasserstein barycenters can be used as centroids in a k-means-style
clustering algorithm for discrete measures.

The algorithm alternates between:

1. assigning each measure to the nearest centroid according to the sliced
   Wasserstein cost;
2. updating each centroid by computing a sliced Wasserstein barycenter of the
   assigned measures.

The implementation uses a k-means++-style initialization and handles empty
clusters by reassigning a poorly represented measure when needed.

See [`SWk_means`](@ref).

## Computational notes

Sliced Wasserstein distances and barycenters can still be computationally
intensive for large problems. The main parameters affecting runtime are:

- the number of random projections `M`;
- the number of support points;
- the number of barycenter or k-means iterations;
- whether Julia is run with multiple threads.

For larger examples, consider running Julia with:

```bash
julia --threads auto
```

## References

- G. Peyré and M. Cuturi. *Computational Optimal Transport*. Foundations and
  Trends in Machine Learning, 2019.
- K. Nguyen. *An Introduction to Sliced Optimal Transport*. Foundations and
  Trends in Computer Graphics and Vision, 2025.
- N. Bonneel, G. Peyré, M. Cuturi, et al. *Sliced and Radon Wasserstein
  Barycenters of Measures*. Journal of Mathematical Imaging and Vision, 2015.
