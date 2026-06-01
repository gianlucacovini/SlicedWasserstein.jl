# Sliced Optimal Transport

## Optimal Transport: a framework for distances between measures

Optimal Transport (OT) provides a powerful framework to compare probability
measures by quantifying the minimal cost required to transport mass from one
distribution to another [Peyré & Cuturi, 2019].
Given two probability measures $\mu$ and $\nu$ on $\mathbb{R}^d$ with equal total
mass, the Kantorovich formulation of optimal transport consists in solving

$$
\pi^\star = \arg\min_{\pi \in \Pi(\mu,\nu)}
\int_{\mathbb{R}^d \times \mathbb{R}^d} c(x,y) d\pi(x,y),
$$

where $\Pi(\mu,\nu)$ is the set of couplings with marginals $\mu$ and $\nu$, and
$c(x,y)$ is a cost function, typically $c(x,y)=\|x-y\|^2$.

A fundamental application of OT is the definition of the
**Kantorovich–Rubinstein–Wasserstein distance** (also known as Earth Mover’s
Distance or, simply, Wasserstein distance). In particular, the quadratic Wasserstein distance is defined as

$$
W_2^2(\mu,\nu)
= \min_{\pi \in \Pi(\mu,\nu)}
\int \|x-y\|^2 d\pi(x,y).
$$

Despite its appealing theoretical properties, computing $W_2$ in high dimension
is computationally expensive, as it requires solving a large-scale linear
program whose complexity grows quickly with the number of support points.

---

## Sliced Optimal Transport: an efficient framework for distances between measures

**Sliced Optimal Transport (SOT)** addresses these computational challenges by
reducing the high-dimensional OT problem to a collection of one-dimensional
problems.
Given a direction $\theta \in \mathbb{S}^{d-1}$, let
$\mathcal{R}_\theta \mu$ denote the projection of the measure $\mu$ onto the line
spanned by $\theta$.
The **squared sliced Wasserstein distance (SW distance)** is defined as

$$
\mathrm{SOT}(\mu,\nu)=
\mathbb{E}_{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}
\left[
W_2^2\bigl(\mathcal{R}_\theta \mu,\mathcal{R}_\theta \nu\bigr)
\right].
$$

Sliced OT is computationally efficient and theoretical appealing because:
- the one-dimensional Wasserstein distance admits a closed-form solution based
  on sorting and cumulative mass matching,
- each slice can be computed independently,
- many geometric properties of optimal transport are preserved, such as
  translation invariance and meaningful interpolation between measures.

Since its introduction, SOT has become a widely used alternative to classical OT
in high-dimensional applications. However, to the best of our knowledge, no
self-contained and efficient Julia implementation was available prior to this
project. In this repository, we follow the exposition of sliced optimal transport in Nguyen, 2025.

---

## Sliced Wasserstein Barycenters

A fundamental geometric concept in optimal transport is that of a **barycenter**, intended as a measure which
summarizes information from a given set of measures $\mu_1,\dots,\mu_K$ weighted by $w_1,\dots,w_K$ with
$\sum_k w_k = 1$. Using SW distance, we can define **sliced Wasserstein barycenters**, defined as

$$
\mu^\star=
\arg\min_{\mu}
\sum_{k=1}^K
w_k \mathrm{SOT}(\mu,\mu_k).
$$

Sliced Wasserstein barycenters are attractive from a computational perspective,
but no closed-form solution is available in general.

---

## Sliced Wasserstein K-means clustering

SW barycenters are often used in applications, in particular they can be used to implement K-means clustering
between probability measures. Given a set of measures $\mu_1,\dots,\mu_m$ and a number of clusters k, one can
assign each measure to a cluster by the SW distance from the SW barycenter (intended as centroid) of the cluster.

---

## Our contributions

This project provides a lightweight Julia implementation of:
- the sliced Wasserstein distance for discrete measures,
- stochastic algorithms for computing free-support sliced Wasserstein
  barycenters,
- an algorithm for computing SW K-means clustering.

The implementation is written from scratch and aims to rely only on standard Julia
packages.

### Discrete measures

We introduce a `DiscreteMeasure` type to represent discrete (possibly weighted) probability measures.
A discrete measure $\mu$ supported on $n$ points in $\mathbb{R}^d$ is written as

$$
\mu = \sum_{i=1}^n w_i \delta_{x_i},
$$

where $x_i \in \mathbb{R}^d$ are the support points and $w_i \ge 0$ are the
associated weights.

In the implementation, a discrete measure is represented by:
- a matrix $X \in \mathbb{R}^{d \times n}$ whose columns correspond to the support
  points $x_i$,
- a vector $w \in \mathbb{R}^n$ of non-negative weights.

Depending on the application, the weights can be automatically normalized so
that $\sum_i w_i = 1$, or left unnormalized.

All the measures in our package are intended as discrete measures.

### Sliced Wasserstein distance

The squared sliced 2-Wasserstein distance between two measures $\mu$ and $\nu$ is implemented in `SOT`.
Given a direction $\theta \in \mathbb{S}^{d-1}$, we denote by
$\mathcal{R}_\theta \mu$ the projection of $\mu$ onto the line spanned by
$\theta$, defined as

$$
\mathcal{R}_\theta \mu=
\sum_{i=1}^n w_i \delta_{\langle x_i, \theta \rangle}.
$$

The sliced Wasserstein cost is approximated using Monte Carlo sampling:

$$
\mathrm{SOT}_M(\mu,\nu)=
\frac{1}{M}
\sum_{m=1}^M
W_2^2\bigl(\mathcal{R}_{\theta_m} \mu,\mathcal{R}_{\theta_m} \nu\bigr),
\qquad
\theta_m \sim \mathcal{U}(\mathbb{S}^{d-1}).
$$

In the implementation, this is achieved by:
- sampling random directions on the unit sphere,
- projecting the measures onto each direction,
- solving a one-dimensional optimal transport problem for each projection,
- averaging the resulting costs.

The one-dimensional OT solver is implemented explicitly using a linear-time
algorithm based on sorting and cumulative mass matching (north-west corner
algorithm).
For gradient computations, the transport plan is stored in an **edge-based
representation**, avoiding the construction of dense transport matrices.

Note that the function returns the **sliced Wasserstein cost** (i.e. the squared
distance), not its square root.

### Sliced Wasserstein barycenters

We implement **free-support sliced Wasserstein barycenters** in the function `SWBarycenters_free_supp` for the quadratic
cost, following the approach of Bonneel et al. (2015).
Given measures $\mu_1,\dots,\mu_K$ and weights $w_1,\dots,w_K$ with
$\sum_k w_k = 1$, the free-support sliced Wasserstein barycenter is the uniform measure
parametrized by its support points, namely

$$
\mu^\star=
\arg\min_{\mu_\phi}
\sum_{k=1}^K
w_k \mathrm{SOT}(\mu_\phi,\mu_k),
$$

$$
\text{s.t.}\quad 
\mu_\phi = \frac{1}{n} \sum_{i=1}^n \delta_{\phi_i}.
$$

The optimization is performed over the locations of the support points.
Since no closed-form solution is available, we minimize the barycenter objective
using a (stochastic) gradient descent scheme.

The gradient of the sliced Wasserstein cost with respect to the support points is
computed using the edge-based representation of the one-dimensional transport
plans.
Due to the non-convex nature of the problem, the objective may admit multiple
local minima.
To mitigate this issue, we employ:
- warm-start initialization based on the input measures,
- an adaptive step size schedule that accounts for the scale of the data.

### Sliced Wasserstein K-means

We implemented a **K-means** clustering algorithm for a set of measures in the function `SWk_means`. The 
algorithm proceed iteratively by assigning each measure to a cluster based on the
SW distance from the centroid and then updating the centroids by computing the 
SW barycenter of the cluster.

- Computational demand of the algorithm is mitigated by warm-starting the centroids
  through a **K-means++** technique.
- Empty clustering issue is faced by assigning the worst classified point to the
  empty cluster if necessary.

---

## Repository structure

- `src/`: implementation of sliced OT, barycenters, K-means and supporting utilities  
- `test/`: unit tests covering correctness, invariances, and reproducibility  
- `benchmark/`: performance benchmarks  
- `examples/`: visual demos

### Environments

This repository uses multiple Julia environments:

- **Root project (`Project.toml`)**  
  Main package environment used by end users.

- **Test environment (`test/Project.toml`)**  
  Contains test-only dependencies.

- **Benchmark environment (`benchmark/Project.toml`)**  
  Contains benchmarking tools (e.g. `BenchmarkTools`) and is not required for normal use.

Each environment is independent and must be instantiated separately.

---

## Installation

The package is not registered yet. It can be installed directly from GitHub with:

```julia
using Pkg
Pkg.add(url="https://github.com/gianlucacovini/SlicedWasserstein.jl")
```

Then it can be loaded with:

```julia
using SlicedWasserstein
```

Alternatively, clone the repository and instantiate the project environment: 

```bash
git clone https://github.com/gianlucacovini/SlicedWasserstein.jl.git
julia --project
pkg> instantiate
```

Eventually initialize with threads to exploit parallelization:

```bash
julia --project --threads auto
pkg> instantiate
```

### Example usage

Computing the SW distance between two measures:

```julia
using SlicedWasserstein

μ = DiscreteMeasure(randn(2, 50))
ν = DiscreteMeasure(randn(2, 50))

sot = SOT(μ, ν)
println(sot)
```

Computing a sliced Wasserstein barycenter:

```julia
using SlicedWasserstein

measures = [DiscreteMeasure(randn(2, 50)) for _ in 1:3]
bar = SWBarycenters_free_supp(measures)

print_full(bar)
```

Performing sliced Wasserstein K-means clustering:

```julia
using SlicedWasserstein

measures = [DiscreteMeasure(randn(2, 20)) for _ in 1:3]
K = 2

assignments, centroids = SWk_means(measures, K; M_bary=200, itmax=10, itmax_bary=10)

println(assignments)
print_full(centroids)
```

Plotting 2d measures:

Note: plotting requires Plots.jl
```julia
using SlicedWasserstein
using Plots

μ = DiscreteMeasure(randn(2, 50) .+ 10)
ν = DiscreteMeasure(randn(2, 50) .- 10)

p = plot(μ; label="μ", color=:blue)
plot!(p, ν; label="ν", color=:red)

display(p)
```

**Performance note**  
Sliced Wasserstein distances and barycenters are computationally intensive.
For large problems, we recommend:
- using a moderate number of projections `M`,
- enabling multithreading (`--threads auto`),
- and tuning the barycenter step size and tolerance.

---

## Implementation details

The implementation faced challenges and opportunities: the main critical
points are the **computational demand** of SW distance and of SW barycenters
and the **non-convexity** of SW barycenter problem.

Addressing both problems we obtained good progress but left minor improvements
for future work. We already mentioned how we work on this details, but we 
summarize here the work already done and to be done.

<br>

To improve efficiency:
- we parallelized the computation of the SW distance and of the gradient used
  in the GD step of SW barycenters computation along directions,
- we implemented in-place Radon projection of measures along directions
- we implemented the transport plan when solving 1d OT in node-edge structure
  to overcome the problem of storing a dense matrix,
- minor allocation reduction and use of @inbounds, @view, @simd when useful.

A point that is still improvable is the allocation cost in the definition of
DiscreteMeasure which copies the input matrix and vector to create a new object.
Other heuristics to improve performances are possible, such as computing the
gradient on minibatches in the barycenter computation.

<br>

To overcome the non-convexity of barycenter computation:
- we warm-start initialization based on the input measures,
- we introduced an adaptive step size schedule that accounts
  for the scale of the data.

Despite these solutions, convergence of barycenter computation
could still be sensitive to hyperparameters choice, such as the 
starting step-size. A finer approach to the problemwould require 
the use of specific heuristics.

### Other considerations

Along the main code in the `src` folder, we gave much attention to
the developement of a complete set of tests, benchmarks and demos.

- A complete overview of tests and benchmarks can be visualized by
  running `runtests.jl` and `benchmark_all.jl` respectively.
  **Note:** `test` and `benchmark` folder have their own environment
  which include dependencies with `Test` and `BenchmarkTools` and
  `ProfileView` respectively.
- The demos, which can be found in the `example` folder cover
  all the main functions implemented through visual representation
  of 2d scatter plots of measures and can be run independently.

---

## References

- Peyré, G., & Cuturi, M. (2019).  
  *Computational Optimal Transport*.  
  Foundations and Trends® in Machine Learning, 11(5–6), 355–607.  

- Nguyen, K. (2025).  
  *An Introduction to Sliced Optimal Transport*.
  Foundations and Trends® in Computer Graphics and Vision 17 (3-4), 171-406.

- Bonneel, N., Peyré, G., Cuturi, M., et al. (2015).  
  *Sliced and Radon Wasserstein Barycenters of Measures*.  
  Journal of Mathematical Imaging and Vision, 51(1), 22–45.  

---
