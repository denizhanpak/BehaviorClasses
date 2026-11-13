# The estimator of the category information and TE (te_work/category/cat_est.jl): Ross's (2014) nearest-neighbour
# mutual information between continuous variables and a discrete label, `ross_mi_nd` (with `PSI` and `tmap`), from the
# cells of CategorizationCategoryAnalysis.jl, where it gives the category information of the evolved agent
# (CATEGORY_INFO), verbatim. The conditional form is the chain rule I(C; Z | W) = I(C; Z, W) − I(C; W), both terms on
# the same noisy points.

"""Runs `f` on each element of `xs` in parallel (one task each)."""
tmap(f, xs) = fetch.([Threads.@spawn f(x) for x in xs])

"""ψ(n), the digamma function at the integers 1…20 000: −γ + Σ_{j<n} 1/j."""
const PSI = let n = 20_000, h = zeros(n)
    for i in 2:n
        h[i] = h[i-1] + 1 / (i - 1)
    end
    h .- Base.MathConstants.eulergamma
end

"""
    ross_mi(x, labels; k = 5)

Ross's (2014) nearest-neighbour estimate of the mutual information (bits) between a continuous variable `x` and a
discrete one `labels`. For each point i of class c: d = the distance to its k-th nearest neighbour of the same class,
m = the number of points of any class within d; Iᵢ = ψ(N) − ψ(N_c) + ψ(k) − ψ(m), averaged over the points. A class
with no more than k points uses k = N_c − 1; a class of one point adds 0. One-dimensional `x`, from sorted arrays.
"""
function ross_mi(x, labels; k=5)
    N = length(x)
    p = sortperm(x)
    xs, ls = x[p], labels[p]
    total = 0.0
    for c in unique(ls)
        xc = xs[ls.==c]
        Nc = length(xc)
        kc = min(k, Nc - 1)
        kc < 1 && continue
        for j in 1:Nc
            xi, lo, hi, d = xc[j], j - 1, j + 1, 0.0
            for _ in 1:kc
                dl = lo >= 1 ? xi - xc[lo] : Inf
                dh = hi <= Nc ? xc[hi] - xi : Inf
                if dl <= dh
                    d, lo = dl, lo - 1
                else
                    d, hi = dh, hi + 1
                end
            end
            dd = d * (1 + 1e-12) + 1e-300
            m = searchsortedlast(xs, xi + dd) - searchsortedfirst(xs, xi - dd)
            total += PSI[N] - PSI[Nc] + PSI[kc] - PSI[max(m, kc)]
        end
    end
    total / N / log(2)
end

"""
The thread count `ross_mi_nd` cuts its work for (0: this session's `Threads.nthreads()`, as the notebook's cell does).
The per-chunk sums are added in chunk order, so the estimate depends on the cut in its last bits (≈ 1e-16 bits); set
it (or `PIPELINE_ROSS_THREADS`) to the thread count of an earlier run to reproduce that run bit for bit.
"""
const ROSS_PARTITION_THREADS = Ref(0)

"""
    ross_mi_nd(X, labels; k = 5)

`ross_mi` for a many-dimensional variable: the columns of `X` (D × N) are the points, with the largest coordinate
difference as the distance. All pairwise distances, in parallel over the points.
"""
function ross_mi_nd(X, labels; k=5)
    D, N = size(X)
    Nc = Dict(c => count(==(c), labels) for c in unique(labels))
    nt = ROSS_PARTITION_THREADS[] > 0 ? ROSS_PARTITION_THREADS[] : Threads.nthreads()
    chunks = Iterators.partition(1:N, cld(N, 4nt))
    parts = tmap(collect(chunks)) do idx
        dist, same, acc = zeros(N), Float64[], 0.0
        for i in idx
            @inbounds for j in 1:N
                m = 0.0
                for a in 1:D
                    m = max(m, abs(X[a, j] - X[a, i]))
                end
                dist[j] = m
            end
            empty!(same)
            for j in 1:N
                j != i && labels[j] == labels[i] && push!(same, dist[j])
            end
            kc = min(k, length(same))
            δ = partialsort!(same, kc)
            m = count(j -> j != i && dist[j] <= δ, 1:N)
            acc += PSI[N] - PSI[Nc[labels[i]]] + PSI[kc] - PSI[max(m, kc)]
        end
        acc
    end
    sum(parts) / N / log(2)
end

"""I(C; Z | W) in bits, for the columns of `Z` and `W` as points (rows = variables) and labels `C`."""
ross_cmi(Z, W, C; k=5) = ross_mi_nd(vcat(Z, W), C; k) - ross_mi_nd(W, C; k)
