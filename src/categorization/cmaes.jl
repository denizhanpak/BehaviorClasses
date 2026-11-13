# The CMA-ES of ActiveCategorizationCTRNN.jl (the cell the edge-law fits evaluated by its text), verbatim.

"""
    cmaes(f, m0, scale; σ0, λ, maxgen, rng, stall)

CMA-ES (`CategorizationLawV2.jl` §4, with a stall stop) minimizing `f` over m0 + scale .* z, the λ candidates of a
generation evaluated in parallel. It stops after `maxgen` generations, when the step size collapses, or when the best
value has improved by less than 0.1% in `stall` generations. Returns the best point, its value and the generations run.
The candidates are drawn from `rng` in one block per generation and their values gathered in order, so the search is
the same at any thread count.
"""
function cmaes(f, m0, scale; σ0=1.0, λ=48, maxgen=1500, rng=MersenneTwister(1), stall=200)
    n = length(m0)
    μ = λ ÷ 2
    w = log(μ + 0.5) .- log.(1:μ)
    w ./= sum(w)
    μeff = 1 / sum(abs2, w)
    cc, cs = (4 + μeff / n) / (n + 4 + 2μeff / n), (μeff + 2) / (n + μeff + 5)
    c1 = 2 / ((n + 1.3)^2 + μeff)
    cμ = min(1 - c1, 2 * (μeff - 2 + 1 / μeff) / ((n + 2)^2 + μeff))
    damps = 1 + 2 * max(0, sqrt((μeff - 1) / (n + 1)) - 1) + cs
    χn = sqrt(n) * (1 - 1 / (4n) + 1 / (21n^2))
    m, σ = zeros(n), σ0
    pc, ps, C = zeros(n), zeros(n), Matrix{Float64}(I, n, n)
    point(z) = m0 .+ scale .* z
    best = (f=f(point(m)), z=copy(m))
    history = Float64[]
    g = 0
    while g < maxgen
        g += 1
        E = eigen(Symmetric(C))
        D, B = sqrt.(max.(E.values, 1e-20)), E.vectors
        Y = B * (D .* randn(rng, n, λ))
        Zs = m .+ σ .* Y
        F = fetch.([Threads.@spawn f(point(Zs[:, j])) for j in 1:λ])
        idx = sortperm(F)
        F[idx[1]] < best.f && (best = (f=F[idx[1]], z=Zs[:, idx[1]]))
        yw = Y[:, idx[1:μ]] * w
        m = m .+ σ .* yw
        ps = (1 - cs) .* ps .+ sqrt(cs * (2 - cs) * μeff) .* (B * ((B' * yw) ./ D))
        hσ = norm(ps) / sqrt(1 - (1 - cs)^(2g)) / χn < 1.4 + 2 / (n + 1)
        pc = (1 - cc) .* pc .+ hσ * sqrt(cc * (2 - cc) * μeff) .* yw
        Ysel = Y[:, idx[1:μ]]
        C = (1 - c1 - cμ) .* C .+ c1 .* (pc * pc' .+ (1 - hσ) * cc * (2 - cc) .* C) .+ cμ .* (Ysel * Diagonal(w) * Ysel')
        C = (C + C') / 2
        σ *= exp((cs / damps) * (norm(ps) / χn - 1))
        push!(history, best.f)
        stalled = g > stall && history[g-stall] - best.f < 1e-3 * max(best.f, 1.0)
        (σ * maximum(D) < 1e-6 || F[idx[end]] - F[idx[1]] < 1e-9 || stalled) && break
    end
    (p=point(best.z), f=best.f, gens=g)
end
