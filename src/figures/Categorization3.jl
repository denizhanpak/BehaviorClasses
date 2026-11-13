# Figure Categorization3: the hold angle, agent A's force map, the hold rule and the hold angles by sign row (A–D).
# Loading this module draws and saves it.
# The code of notebooks/Categorization.jl, which loads these modules and explains each panel; run there or by
# pipeline/analyze_categorization.jl (Categorization2–4) and pipeline/fit_categorization.jl (Categorization1).
module Categorization3

using Markdown
using MathematicalSystems
import HybridSystems, LazySets
using OrdinaryDiffEqTsit5
using Plots
using LinearAlgebra
import JLD2
import KernelDensity, StatsBase, Roots, Random
using ..CategorizationCommon
using ..BehaviorClasses.BehaviorTools, ..BehaviorClasses.AgentModels
using ..CategorizationLarger

# ── cell e42c9cec ────────────────────────────────────────────────────────────
begin
    """Angles of the outer, middle and inner ray pairs from the middle of the view (°), from AgentModels' `ray_angles`."""
    const PAIR_ANGLES = [abs(rad2deg(ray_angles()[j])) for j in 1:3]

    """
        hold_bands(row)

    The ranges of angles (°) in which a circuit with sign row `row` (the signs of a₁, a₂, a₃) holds the circle's
    outer edge, read from the middle outward: a + pair with a − pair just inside it holds the edge from its own angle
    to the next pair's (beyond the outermost pair, without bound); a + pair with no − pair inside it (ray 4 does not
    steer) lets the circle in to the middle, from 0 to its angle. Returns a vector of (lo, hi).
    """
    function hold_bands(row)
        s, ψ = reverse(collect(row)), reverse(PAIR_ANGLES)      # inner, middle, outer
        bands = Tuple{Float64,Float64}[]
        for i in 1:3
            s[i] == '+' || continue
            if i == 1
                push!(bands, (0.0, ψ[1]))
            elseif s[i-1] == '−'
                push!(bands, (ψ[i], i < 3 ? ψ[i+1] : Inf))
            end
        end
        bands
    end

    """Each sign row's range(s) of the held outer edge (`hold_bands`)."""
    ROW_BANDS = Dict(r => hold_bands(r) for r in BAND_ORDER)
    in_bands(x, bands) = any(((lo, hi),) -> lo <= x < hi, bands)
end

# ── cell ad72fbe9 ────────────────────────────────────────────────────────────
"""
One entry per circuit, sorted by α₀⁺: its short name `s`, `family`, action `mode`, free vector `q` (log τ, θ, W_s, W_c,
U₁ … U₇), ray weights `U`, odd parts `a` (a₁, a₂, a₃: the outer, middle and inner pair), sign `row`, the outer-edge
fit's hold angle `α0p` (α₀⁺), and the kept fit's edge `e` and angle `α0`.
"""
CIRCUIT_TABLE = let rows = map(collect(keys(ALL_FITS))) do s
        f = ALL_FITS[s]
        q = CIRCUITS[f.k].q
        U = q[5:11]
        a = [(U[j] - U[8-j]) / 2 for j in 1:3]
        (; s, f.family, f.mode, q, U, a, row=join(x > 0 ? "+" : "−" for x in a), α0p=f.circle[1].L.α0, f.e,
            α0=f.circle[f.e].L.α0)
    end
    @assert length(rows) == length(CIRCUITS) "a circuit has no fit"
    @assert all(r -> r.row in BAND_ORDER, rows) "a circuit's sign row is outside the three rows"
    sort(rows; by=r -> r.α0p)
end

# ── cell cb288196 ────────────────────────────────────────────────────────────
"""
For each circuit (in `CIRCUIT_TABLE`'s order, by α₀⁺): its short name `s`, the outer-edge circle law's hold angle `α0p`
and `speed`, the speed that law reaches on its own circle paths: the median over d = 1 … 50 of each path's largest |v|
(`run_edge_law`, steps of 0.01).
"""
HOLD_SPEED = map(CIRCUIT_TABLE) do c
    L = ALL_FITS[c.s].circle[1].L
    (; c.s, c.α0p, speed=median(maximum(abs, diff(run_edge_law(L, Float64(d)))) / LAW_DT for d in 1:50))
end

# ── cell 1d58ab04 ────────────────────────────────────────────────────────────
"""
    three_clusters(a)

The exact one-dimensional k-means of the sorted values `a` with k = 3: the cut at the pair of places that leaves the
smallest sum of squares within the clusters (every pair tried). Returns the three index ranges.
"""
function three_clusters(a)
    n = length(a)
    ss(r) = (m = mean(a[r]); sum(x -> (x - m)^2, a[r]))
    _, i, j = minimum((ss(1:i) + ss(i+1:j) + ss(j+1:n), i, j) for i in 1:n-2 for j in i+1:n-1)
    [1:i, i+1:j, j+1:n]
end

# ── cell 9c8e3136 ────────────────────────────────────────────────────────────
"""
Every agent in 3A, two-node and larger, sorted by α₀⁺: its name `s`, size `N`, sign `row`, the outer-edge circle law's
hold angle `α0p` and `speed` (as `HOLD_SPEED`; the larger agents' laws are the fits of `pipeline/fit_categorization.jl`, step fit_larger,
cached).
"""
HOLD_ALL = let F = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "cache", "categorization_edge_law_nnode.jld2"), "data")
    two = [(; h.s, N=2, c.row, h.α0p, h.speed) for (h, c) in zip(HOLD_SPEED, CIRCUIT_TABLE)]
    larger = map(LARGER.agents) do a
        L = F[a.name].circle[1].L
        (; s=a.name, a.N, a.row, α0p=L.α0, speed=median(maximum(abs, diff(run_edge_law(L, Float64(d)))) / LAW_DT for d in 1:50))
    end
    sort([two; larger]; by=r -> r.α0p)
end

# ── cell 07dc2670 ────────────────────────────────────────────────────────────
"""
The three clusters of every agent's α₀⁺ in 3A (`three_clusters` over `HOLD_ALL`): for each cluster its range `lo`–`hi`,
`median`, number `n`, the range of its laws' speeds `speeds`, and how many of its agents are of each sign row (`rows`).
"""
HOLD_CLUSTERS = let a = [h.α0p for h in HOLD_ALL]
    map(three_clusters(a)) do r
        rows = Dict(row => count(h -> h.row == row, HOLD_ALL[r]) for row in ROW_ORDER)
        (; lo=a[first(r)], hi=a[last(r)], median=median(a[r]), n=length(r), speeds=extrema(h.speed for h in HOLD_ALL[r]), rows)
    end
end

# ── cell 64c5fa79 ────────────────────────────────────────────────────────────
"""3A: every agent's hold angle α₀⁺ against its law's speed, in the grey of its brain size, with the medians of the three clusters (dashed)."""
panel_hold_speed = let xlims = (0, 5.5), ylims = (-1, 25)
    p = plot(; title="hold angle and speed of the edge law", xlabel="speed the law reaches (median peak |v|)",
        ylabel="hold angle α₀⁺ (°)", xlims, ylims, legend=:topright, legendfontsize=7)
    for c in HOLD_CLUSTERS
        hline!(p, [c.median]; color=INK_MUTED, linestyle=:dash, linewidth=1, label="")
        annotate!(p, xlims[2] - 0.05, c.median + 0.7, text(fmt(c.median; d=1) * "°, " * string(c.n) * " agents", 7, INK_SECONDARY, :right))
    end
    for h in HOLD_ALL
        scatter!(p, [h.speed], [h.α0p]; color=SIZE_COLORS[h.N], markeralpha=0.6, markerstrokewidth=0, markersize=5, label="")
    end
    for N in 2:8
        plot!(p, [NaN], [NaN]; color=SIZE_COLORS[N], linewidth=5, label=string(N) * " neurons")
    end
    p
end

# ── cell 73d517d0 ────────────────────────────────────────────────────────────
begin
    """The force map's grid: the circle's side position relative to the agent `rel` (−100 … 100 in steps of 0.25) and its height `h` (15 … 275 in steps of 0.5), as in `ActiveCategorizationTwoNodeAgents.jl`."""
    HELD_GRID = (rel=-100.0:0.25:100.0, h=15.0:0.5:275.0)

    """
    Agent A's force map under a circle (`held_forces` over `HELD_GRID`): the smallest and largest stable force `lo` and
    `hi` with the agent and the circle held still (length(h) × length(rel), `NaN` where no ray sees the circle) and
    `two`, true where they differ, so that the nodes have two stable states.
    """
    HELD_MAP = let c = AGENT_CIRCUIT, rel = HELD_GRID.rel, h = HELD_GRID.h
        lo, hi = fill(NaN, length(h), length(rel)), fill(NaN, length(h), length(rel))
        Threads.@threads for j in eachindex(rel)
            for i in eachindex(h)
                lo[i, j], hi[i, j] = held_forces(c, :circle, rel[j], h[i])
            end
        end
        (; lo, hi, two=hi .- lo .> 1e-6)
    end
end

# ── cell a8718253 ────────────────────────────────────────────────────────────
"""Agent A's circle trials drawn on the force map: from d = ±5, ±10, …, ±50, Δx at the 868 positions."""
HELD_PATHS = [run_circuit(AGENT_CIRCUIT, :circle, d) for d in [-50.0:5.0:-5.0; 5.0:5.0:50.0]]

# ── cell da435dd6 ────────────────────────────────────────────────────────────
"""The force map's share of the seen area where the nodes have two stable states (A latches), and the force's range."""
HELD_MAP_CHECK = (; two=count(HELD_MAP.two) / count(isfinite, HELD_MAP.lo), forces=extrema(filter(isfinite, HELD_MAP.lo)))

# ── cell 3fe9ced8 ────────────────────────────────────────────────────────────
"""3B: agent A's force map under a circle with its circle trials, and the colour key (right, where no ray sees)."""
panel_force_map = let key = (x=(104.0, 110.0), h=(110.0, 230.0)), swatch = (h=(62.0, 82.0),)
    rel, h = HELD_GRID.rel, HELD_GRID.h
    stripe(r, y) = isodd(floor(Int, 0.575r + 0.375y))     # diagonal stripes, as in the agents notebook
    img = [HELD_MAP.two[i, j] && stripe(rel[j], h[i]) ? HELD_MAP.hi[i, j] : HELD_MAP.lo[i, j] for i in eachindex(h), j in eachindex(rel)]
    p = heatmap(rel, h, img; c=HELD_SCALE, clims=(-5, 5), colorbar=false, title="Agent Example: the force, circle held still",
        xlabel="circle relative to the agent  (← left · right →)", ylabel="height of the circle, h", xlims=(-100, 130),
        xticks=-100:50:100, ylims=(15, 275), grid=false, framestyle=:box)
    vline!(p, [0.0]; color=INK_MUTED, linewidth=0.6, label="")
    for x in HELD_PATHS
        plot!(p, -x, TRIAL_H; color=AGENT_COLOR, linewidth=0.9, alpha=0.9, label="")
    end
    # the key, where no ray sees: the force scale and a striped swatch
    edges = range(key.h..., length=41)
    for k in 1:40
        F = -5 + 10 * (k - 0.5) / 40
        plot!(p, Shape([key.x[1], key.x[2], key.x[2], key.x[1]], [edges[k], edges[k], edges[k+1], edges[k+1]]);
            color=get(HELD_SCALE, (F + 5) / 10), linewidth=0, label="")
    end
    for (y, s) in ((key.h[2], "5 right"), (sum(key.h) / 2, "0"), (key.h[1], "5 left"))
        annotate!(p, key.x[2] + 2, y, text(s, 7, INK_SECONDARY, :left))
    end
    annotate!(p, sum(key.x) / 2, key.h[2] + 9, text("force", 8, INK_SECONDARY, :center))
    for (k, x) in enumerate(range(key.x..., length=7)[1:end-1])
        plot!(p, Shape([x, x + 1, x + 1, x], [swatch.h[1], swatch.h[1], swatch.h[2], swatch.h[2]]);
            color=isodd(k) ? HELD_COLORS.right : HELD_COLORS.left, linewidth=0, label="")
    end
    annotate!(p, key.x[1], swatch.h[1] - 8, text("two states", 7, INK_SECONDARY, :left))
    p
end

# ── cell 293b6695 ────────────────────────────────────────────────────────────
"""
For each sign row: its circuits' α₀⁺ (`x`), Silverman's bandwidth (`bandwidth`, °) and the kernel density on a grid from
−6 to 30° (`grid`, `density`; KernelDensity.jl), with area 1.
"""
ROW_DENSITIES = Dict(map(BAND_ORDER) do r
    x = [c.α0p for c in CIRCUIT_TABLE if c.row == r]
    k = KernelDensity.kde(x; boundary=(-6.0, 30.0), npoints=2048)
    r => (; x, bandwidth=KernelDensity.default_bandwidth(x), grid=collect(k.x), density=collect(k.density))
end)

# ── cell 0fad33c5 ────────────────────────────────────────────────────────────
begin
    """
    Every agent's hold angle and sign row, two-node and larger (3C, 3D): its name `s`, size `N`, `family`, sign `row`
    (for a larger agent, its motor pair's: the rays into L₁) and α₀⁺ (`α0p`), from `CIRCUIT_TABLE` and `LARGER`.
    """
    ALL_HOLDS = [[(; c.s, N=2, c.family, c.row, c.α0p) for c in CIRCUIT_TABLE];
        [(; s=a.name, a.N, a.family, a.row, a.α0p) for a in LARGER.agents]]
    @assert all(h -> h.row in ROW_ORDER, ALL_HOLDS) "an agent's sign row is not in ROW_ORDER"

    """Each sign row's range(s) of the held outer edge (`hold_bands`), for every row in `ROW_ORDER`."""
    ALL_ROW_BANDS = Dict(r => hold_bands(r) for r in ROW_ORDER)

    """
    For each sign row with at least 3 agents of any size: their α₀⁺ (`x`) and the kernel density on a grid from −6 to
    30° (`grid`, `density`; KernelDensity.jl, Silverman's bandwidth), with area 1, as `ROW_DENSITIES` for the two-node
    agents alone.
    """
    ALL_ROW_DENSITIES = Dict(map(filter(r -> count(h -> h.row == r, ALL_HOLDS) >= 3, ROW_ORDER)) do r
        x = [h.α0p for h in ALL_HOLDS if h.row == r]
        k = KernelDensity.kde(x; boundary=(-6.0, 30.0), npoints=2048)
        r => (; x, grid=collect(k.x), density=collect(k.density))
    end)

    """For each sign row: how many agents it has (`n`), and those that hold outside its range (`outside`, names)."""
    ROW_COUNTS = Dict(r => (; n=count(h -> h.row == r, ALL_HOLDS),
        outside=[h.s for h in ALL_HOLDS if h.row == r && !in_bands(h.α0p, ALL_ROW_BANDS[r])]) for r in ROW_ORDER)
end

# ── cell b8930d5b ────────────────────────────────────────────────────────────
"""3C: the hold rule of every sign row on the agent's view, side by side."""
panel_hold_rule = let h = 150.0, reach = 200.0, gap = 205.0
    ψ = rad2deg.(ray_angles())                       # the seven rays, −15 … 15°
    pair(j) = min(j, 8 - j)                          # 1 outer, 2 middle, 3 inner, 4 ray 4
    t = range(0, 2π; length=100)
    n = length(ROW_ORDER)
    p = plot(; aspect_ratio=:equal, xlims=(-(n - 1) / 2 * gap - 100, (n - 1) / 2 * gap + 100), ylims=(-22, reach + 66),
        framestyle=:none, grid=false, legend=false, title="the hold rule of each sign row")
    for (i, row) in enumerate(ROW_ORDER)
        X = (i - (n + 1) / 2) * gap
        xy(r, ang) = (X + r * sind(ang), r * cosd(ang))
        for (lo, hi) in ALL_ROW_BANDS[row]
            hi = min(hi, 24.0)
            for (a, b) in (lo == 0 ? [(-hi, hi)] : [(lo, hi), (-hi, -lo)])
                angs = range(a, b; length=40)
                pts = vcat([xy(15.0, g) for g in angs], [xy(reach, g) for g in reverse(angs)])
                plot!(p, Shape(first.(pts), last.(pts)); color=ROW_COLORS[row], alpha=0.25, linewidth=0)
            end
        end
        for j in 1:7
            k = pair(j)
            sgn = k == 4 ? '0' : collect(row)[k]
            style = sgn == '+' ? (color=INK, linestyle=:solid, linewidth=1.6) :
                    sgn == '−' ? (color=INK_MUTED, linestyle=:dash, linewidth=1.1) : (color=INK_MUTED, linestyle=:dot, linewidth=0.9)
            plot!(p, [xy(15.0, ψ[j])[1], xy(reach, ψ[j])[1]], [xy(15.0, ψ[j])[2], xy(reach, ψ[j])[2]]; style...)
            k == 4 || annotate!(p, xy(reach + 10, ψ[j])..., text(string(sgn), 11, sgn == '+' ? INK : INK_SECONDARY, :center))
            i == 1 && ψ[j] >= 0 && annotate!(p, xy(reach + 26, ψ[j])..., text("$(round(Int, ψ[j]))°", 7, INK_MUTED, :center))
        end
        plot!(p, Shape(X .+ 15cos.(t), 15sin.(t)); color=:white, linecolor=INK_SECONDARY, linewidth=1)
        E = median([a.α0p for a in ALL_HOLDS if a.row == row])
        φ = E <= asind(15 / h) ? 0.0 : Roots.find_zero(φ -> φ + asind(15cosd(φ) / h) - E, (0.0, E))
        plot!(p, Shape(X + h * tand(φ) .+ 15cos.(t), h .+ 15sin.(t)); color=INK_MUTED, alpha=0.25,
            linecolor=INK_SECONDARY, linewidth=1)
        φ > 0 && plot!(p, [xy(15.0, E)[1], xy(reach, E)[1]], [xy(15.0, E)[2], xy(reach, E)[2]]; color=ROW_COLORS[row],
            linewidth=3)
        annotate!(p, X, reach + 56, text(spaced(row), 12, ROW_COLORS[row], :center))
    end
    p
end

# ── cell d0381e8a ────────────────────────────────────────────────────────────
"""3D: the density of α₀⁺ in each sign row over the agents of every size, with the agents as ticks and the two-node rows' ranges shaded."""
panel_hold_density = let xmax = 24.0
    top = maximum(maximum(d.density) for d in values(ALL_ROW_DENSITIES))
    p = plot(; xlims=(0, xmax), ylims=(-0.17top, 1.24top), xticks=0:2:xmax, legend=(0.225, 0.66),
        xlabel="outer-edge hold angle α₀⁺ (°)", ylabel="density (per °; each row's area 1)",
        title="where each sign row holds the circle's outer edge, all brain sizes")
    for r in BAND_ORDER, (lo, hi) in ROW_BANDS[r]
        vspan!(p, [lo, min(hi, xmax)]; color=ROW_COLORS[r], alpha=0.13, label="")
    end
    for (ψ, name) in zip(PAIR_ANGLES, ["outer rays", "middle rays", "inner rays"])
        vline!(p, [ψ]; color=INK_MUTED, linestyle=:dot, linewidth=0.8, label="")
        annotate!(p, ψ + 0.12, 1.12top, text("$name $(round(Int, ψ))°", 8, INK_MUTED, :left))
    end
    hline!(p, [0.0]; color=INK_MUTED, linewidth=0.8, label="")
    for r in ROW_ORDER
        c = ROW_COUNTS[r]
        lab = "$(spaced(r)): $(c.n) agent$(c.n == 1 ? "" : "s")"
        if haskey(ALL_ROW_DENSITIES, r)
            d = ALL_ROW_DENSITIES[r]
            plot!(p, d.grid, d.density; fillrange=0, fillalpha=0.30, color=ROW_COLORS[r], linewidth=2, label=lab)
        else                                          # too few for a density: in the key by its ticks' colour
            plot!(p, [NaN], [NaN]; color=ROW_COLORS[r], linewidth=2, label=lab)
        end
    end
    for a in ALL_HOLDS
        plot!(p, [a.α0p, a.α0p], [-0.14top, -0.03top]; color=ROW_COLORS[a.row], linewidth=1.5, label="")
    end
    p
end

# ── cell 5b6c737c ────────────────────────────────────────────────────────────
"""Figure 3: hold angle and speed (A), A's force map (B), the hold rule (C) and the hold angles by sign row (D), lettered and saved to `figures/`."""
figure_bearing = let
    top = plot(panel_hold_speed, panel_force_map; layout=grid(1, 2))
    plt = plot(top, panel_hold_rule, panel_hold_density; layout=grid(3, 1; heights=[0.42, 0.22, 0.36]), size=(1300, 1500),
        left_margin=6Plots.mm, bottom_margin=5Plots.mm)
    save_figure(panel_letters!(plt), "Categorization3")
end

# Export every name defined here, so the analysis scripts and the notebook see them all.
eval(Expr(:export, filter(n -> !startswith(string(n), '#') && n ∉ (:eval, :include, nameof(@__MODULE__)),
    names(@__MODULE__; all=true))...))

end
