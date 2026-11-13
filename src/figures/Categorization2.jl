# Figure Categorization2: the relief deficit and the category TE (A–D). Loading this module draws and saves it.
# The code of notebooks/Categorization.jl, which loads these modules and explains each panel; run there or by
# pipeline/analyze_categorization.jl (Categorization2–4) and pipeline/fit_categorization.jl (Categorization1).
module Categorization2

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

# ── cell 41a49a2f ────────────────────────────────────────────────────────────
"""
    relief_deficit_drawing(; h = 60, hstar = 45)

2A: a line at height `h` (brown) and the circle (dashed) whose image has the same angular width 2β and whose nearest point
is the line's middle, seen along the sight lines (dotted) from the agent's centre. Red: the arc of the circle that the
angle β covers, from its nearest point to its edge, and the relief the line lacks at its edge. Grey rules: the heights h₀, h₁ and h₂ = `h` of the line as it falls;
black dotted: h*, where the deficit reaches D* (at `hstar`, for illustration). Redrawn from
the development script `deficit_geometry.jl`.
"""
function relief_deficit_drawing(; h=60.0, hstar=45.0)
    r = VISUAL.size / 2                                               # the object's half-width
    ray(θ, d) = (d * sind(θ), d * cosd(θ))                            # a point at bearing θ (degrees), distance d
    seg!(p, a, b; kw...) = plot!(p, [a[1], b[1]], [a[2], b[2]]; label="", kw...)
    note!(p, x, y, s; size=8, align=:left, color=INK_SECONDARY) = annotate!(p, x, y, text(s, size, align, color))
    circle_color, line_color = COLORS.blue, "#8b5a2b"
    β = atand(r / h)
    R = h * sind(β) / (1 - sind(β))                                   # the circle with the same image, touching the line's middle
    C = (0.0, h + R)
    near(θ) = ray(θ, (h + R) * cosd(θ) - sqrt(max(R^2 - (h + R)^2 * sind(θ)^2, 0.0)))   # where the sight line at θ first meets the circle
    T = near(β)                                                       # the circle's edge, the tangent point
    E = ray(β, h / cosd(β))                                           # the line's edge
    p = plot(; aspect_ratio=:equal, xlims=(-42, 42), ylims=(-15, 108), framestyle=:none, grid=false, legend=false)
    plot!(p, disc(0.0, 0.0, 15.0); color=INK_SECONDARY, fillalpha=0.18, linewidth=0, label="")
    note!(p, 0, -9, "agent"; size=7, align=:center, color=INK_MUTED)
    for (i, d) in enumerate((100.0, 80.0, h))                         # the line's heights h₀, h₁, h₂ as it falls; it is at h₂ now
        now = d == h
        seg!(p, (-42, d), (42, d); color=INK_MUTED, linewidth=0.8, alpha=now ? 1.0 : 0.5)
        note!(p, -41, d + 2.6, "h" * string('₀' + (i - 1)) * " = " * string(round(Int, d)); size=10,
            color=now ? INK_SECONDARY : INK_MUTED)
    end
    seg!(p, (-42, hstar), (42, hstar); color=INK, linestyle=:dot, linewidth=1.4)
    note!(p, -41, hstar + 2.6, "h*"; size=10)
    plot!(p, disc(C..., R); color=circle_color, fillalpha=0.10, linecolor=circle_color, linestyle=:dash, linewidth=1, label="")
    plot!(p, Shape([-r, r, r, -r], [h - 1.1, h - 1.1, h + 1.1, h + 1.1]); color=line_color, fillalpha=0.6, linewidth=0, label="")
    seg!(p, (0, 0), (0, h); color=INK, linestyle=:dot, linewidth=1.2)                       # the centre sight line
    seg!(p, (0, h), C; color=INK, linestyle=:dash, linewidth=1)                              # on to the circle's centre
    scatter!(p, [C[1]], [C[2]]; color=INK, markersize=2.5, markerstrokewidth=0, label="")
    for s in (-1, 1)
        seg!(p, (0, 0), ray(s * β, 96); color=INK, linestyle=:dot, linewidth=1.2)           # the edge sight lines
        scatter!(p, [s * E[1]], [E[2]]; color=line_color, markersize=3.5, markerstrokewidth=0, label="")
    end
    seg!(p, E, T; color=LAW_COLOR, linewidth=4)                                              # the relief the line lacks at its edge
    seg!(p, T, C; color=INK, linestyle=:dash, linewidth=1)                                   # the radius to the tangent point
    u, w = (0 .- T) ./ hypot((0 .- T)...), (C .- T) ./ R                                     # the right angle at T
    q = 2.4
    plot!(p, [T[1] + q * u[1], T[1] + q * (u[1] + w[1]), T[1] + q * w[1]], [T[2] + q * u[2], T[2] + q * (u[2] + w[2]), T[2] + q * w[2]];
        color=INK, linewidth=0.8, label="")
    arc = near.(range(0, β; length=40))
    plot!(p, first.(arc), last.(arc); color=LAW_COLOR, linewidth=3, label="")                # the arc β covers
    plot!(p, [18 * sind(t) for t in range(0, β; length=30)], [18 * cosd(t) for t in range(0, β; length=30)];
        color=INK_SECONDARY, linewidth=0.8, label="")
    note!(p, 2.4, 21, "β"; size=10)
    p
end

# ── cell 6fdda6f9 ────────────────────────────────────────────────────────────
"""2A: the relief deficit of a line: the arc β covers and its projection on the line (`relief_deficit_drawing`)."""
panel_relief_deficit = plot!(relief_deficit_drawing(); title="the relief deficit of a line")

# ── cell 0ecefba6 ────────────────────────────────────────────────────────────
begin
    """
    The category TE of `CategorizationStrategies.jl` §7 (`pipeline/analyze_categorization.jl`, step category_te), in `CIRCUITS` order: the heights `h` of its
    moments and `quantities[q][k, m]` for circuit k at moment m, q = `:te_F` (TE from the shape to the force), `:mi_x`
    (what the position holds) and `:mi_state` (what the two nodes hold), among others.
    """
    CATEGORY = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "cache", "categorization_strategies_category.jld2"), "data")
    """The relief ablations of `CategorizationStrategies.jl` §7: `windows[k, 1, i]`, circuit k's catch rate with the circle flattened inside the window centred at `centres[i]`, `width` high."""
    RELIEF = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "cache", "categorization_strategies_relief.jld2"), "data")
    """The edge law's 𝒟* for each circuit, with the split the TE script used (`CategorizationStrategies.jl` §7)."""
    CATEGORY_DSTAR = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "cache", "categorization_strategies_dstar.jld2"), "data")
end

# ── cell 65a6a629 ────────────────────────────────────────────────────────────
begin
    """Bits the position must hold for the paths of circles and lines to have parted (the split)."""
    const SPLIT_BITS = 0.9
    """Both shapes are in range below this height: the line from 235 on, the circle from 250."""
    const BOTH_IN_RANGE = 235.0
    """A relief window counts when flattening the circle inside it leaves fewer than this share of circles caught."""
    const WINDOW_CUT = 0.8

    """
    For each circuit, in `CIRCUITS` order: its `name`, whether it is `latched`, the push height `hstar` = 15/𝒟*, the
    `split` (NaN if the position never holds `SPLIT_BITS`), the centres of its relief windows `windows`, and the most
    damaging window `worst` (NaN if none counts) with its catch rate `worst_rate`.
    """
    CATEGORY_TIMING = map(eachindex(CIRCUITS)) do k
        x, flat = CATEGORY.quantities[:mi_x][k, :], RELIEF.windows[k, 1, :]
        s, i = findfirst(>=(SPLIT_BITS), x), argmin(flat)
        (; name=CIRCUITS[k].short, latched=ACTION_MODES[k].latched, hstar=VISUAL.size / 2 / CATEGORY_DSTAR.Dstar[k],
            split=s === nothing ? NaN : CATEGORY.h[s], windows=RELIEF.centres[flat.<WINDOW_CUT],
            worst=flat[i] < WINDOW_CUT ? RELIEF.centres[i] : NaN, worst_rate=flat[i])
    end
end

# ── cell 6afeeba1 ────────────────────────────────────────────────────────────
"""The caches against each other: the same circuits in the same order (`names`), and the split recomputed here against the one the D* script used (`split`, all equal)."""
CATEGORY_CHECK = (; names=CATEGORY.names == RELIEF.names == CATEGORY_DSTAR.names == [c.short for c in CIRCUITS],
    split=all(isequal(t.split, CATEGORY_DSTAR.split[k]) for (k, t) in enumerate(CATEGORY_TIMING)))

# ── cell 44d54e45 ────────────────────────────────────────────────────────────
begin
    """The fill of the TE panels for what the position holds, as in `CategorizationStrategies.jl`, and the colour of the relief windows."""
    const CATEGORY_FILLS = (position="#e2e0d8",)
    const WINDOW_COLOR = "#77756e"

    """
        te_panel(name; title, legend = false, note = "")

    2B and 2C: what the category tells over the object's height for circuit `name` (falling from left to right), from
    `BOTH_IN_RANGE` down, where both shapes are in range: the TE from the shape to the force (colour of its action mode;
    faint after the split), what the two neurons hold (black) and what the position holds (grey fill), the split (black
    diamond), its relief windows (dark bars at the top, each window's centre ± 2.5), the push height h* (red point and
    dotted line) and the TE's peak before the split (triangle). `note` is written under the top of the panel.
    """
    function te_panel(name; title, legend=false, columns=1, note="", top=1.22)
        k = circuit_index(name)
        h, Q, t = CATEGORY.h, CATEGORY.quantities, CATEGORY_TIMING[k]
        te, x, nodes = Q[:te_F][k, :], Q[:mi_x][k, :], Q[:mi_state][k, :]
        s = something(findfirst(>=(SPLIT_BITS), x), length(h) + 1)
        pre, post = copy(te), copy(te)
        pre[s:end] .= NaN
        post[1:max(s - 2, 0)] .= NaN
        p = plot(; title, xflip=true, xlims=(15, BOTH_IN_RANGE), ylims=(-0.03, top), yticks=0:0.25:1, legend, legend_column=columns,
            xlabel="height of the object, h", ylabel="bits about the relief deficit")
        plot!(p, h, max.(x, 0); fillrange=0, color=CATEGORY_FILLS.position, linewidth=0, label="the position holds")
        plot!(p, h, nodes; color=AGENT_COLOR, linewidth=1.0, label="the two neurons hold")
        plot!(p, h, post; color=mode_color(k), linewidth=0.8, alpha=0.35, label="")
        plot!(p, h, pre; color=mode_color(k), linewidth=1.8, label="TE shape → force")
        isnan(t.split) || scatter!(p, [t.split], [SPLIT_BITS]; color=AGENT_COLOR, marker=:diamond, markersize=5,
            markerstrokecolor=:white, label="split")
        m = findfirst(==(CATEGORY_DSTAR.te_peak[k]), h)                # the TE peak before the split (`CATEGORY_DSTAR`)
        m === nothing || scatter!(p, [h[m]], [te[m]]; color=mode_color(k), marker=:utriangle, markersize=7,
            markerstrokecolor=:white, label="TE peak")
        for (n, c) in enumerate(t.windows)
            plot!(p, [c - 2.5, c + 2.5], [1.12, 1.12]; color=WINDOW_COLOR, linewidth=4, label=n == 1 ? "relief window" : "")
        end
        plot!(p, [t.hstar, t.hstar], [-0.03, 1.07]; color=LAW_COLOR, linestyle=:dot, linewidth=1, label="")
        scatter!(p, [t.hstar], [1.12]; color=LAW_COLOR, markersize=5, markerstrokecolor=:white, label="h*")
        isempty(note) || annotate!(p, BOTH_IN_RANGE - 5, 1.12, text(note, 8, INK_SECONDARY, :left))
        p
    end
end

# ── cell 155bf22b ────────────────────────────────────────────────────────────
"""
For each two-node agent: its four heights, the `split`, the most damaging relief `window`, the TE peak before the split
`te` (`CATEGORY_DSTAR.te_peak`) and the push height `hstar` (`CATEGORY_TIMING`), and the centres of all its relief
`windows`; their `spread` (highest − lowest of the four) and `apart`, the smallest distance between two of the split,
the TE peak and h*, or between one of them and the heights its relief windows flatten (each window's centre ± half its
height, `RELIEF.width`), NaN unless it has all four; and whether all four lie inside the trial (`inside`, h ≤ 255).
`aligned`: the agent whose four heights lie closest together (smallest spread); `unaligned`: the one whose split, TE
peak, h* and relief windows lie farthest apart (largest `apart`), of those with all four inside the trial.
"""
ALIGNMENT = let half = RELIEF.width / 2
    rows = map(eachindex(CIRCUITS)) do k
        t = CATEGORY_TIMING[k]
        v = [t.split, t.worst, CATEGORY_DSTAR.te_peak[k], t.hstar]
        ok = all(isfinite, v)
        pts = v[[1, 3, 4]]                                            # the split, the TE peak and h*
        off(x) = max(minimum(abs.(x .- t.windows)) - half, 0.0)     # a height's distance from the flattened heights
        (; t.name, split=v[1], window=v[2], te=v[3], hstar=v[4], t.windows, spread=ok ? maximum(v) - minimum(v) : NaN,
            apart=ok ? minimum([[abs(pts[i] - pts[j]) for i in 1:3 for j in i+1:3]; off.(pts)]) : NaN,
            inside=ok && maximum(v) <= 255)
    end
    full = filter(r -> isfinite(r.spread), rows)
    (; rows, n=length(full), aligned=argmin(r -> r.spread, full).name, unaligned=argmax(r -> r.apart, filter(r -> r.inside, full)).name)
end

# ── cell dee0fdcb ────────────────────────────────────────────────────────────
"""2B: the category TE of the agent whose split, relief window, TE peak and push height line up best."""
panel_te_aligned = te_panel(ALIGNMENT.aligned; title="agent " * ALIGNMENT.aligned, legend=:outertop, columns=4)

# ── cell fe1adb53 ────────────────────────────────────────────────────────────
"""2C: the category TE of the agent whose split, TE peak, push height and relief windows lie farthest apart."""
panel_te_unaligned = te_panel(ALIGNMENT.unaligned; title="agent " * ALIGNMENT.unaligned)

# ── cell f00b3ad9 ────────────────────────────────────────────────────────────
"""
How well the push height h* predicts, over the 41 two-node agents, (a) the `split` and (b) the most damaging relief
window (`window`, over the agents that have one): Spearman's ρ (StatsBase) and the number of agents `n`.
"""
DSTAR_PREDICTS = let T = CATEGORY_TIMING
    rank(f) = (sel = filter(t -> isfinite(f(t)), T); (ρ=StatsBase.corspearman([t.hstar for t in sel], f.(sel)), n=length(sel)))
    (; split=rank(t -> t.split), window=rank(t -> t.worst))
end

# ── cell d8b0703b ────────────────────────────────────────────────────────────
"""
How well the push height h* predicts the split and the most damaging relief window, for each brain size 2 … 8: Spearman's
ρ (StatsBase) over the agents of that size that have a split (`split`, over `n_split`) and that have a relief window
(`window`, over `n_window`); NaN where fewer than 4 agents have one. Size 2 is `DSTAR_PREDICTS`; sizes 3 … 8 are the
larger agents (`LARGER`). `pooled`: the same over all 60 larger agents together.
"""
DSTAR_BY_SIZE = let A = LARGER.agents
    rank(S, f) = length(S) >= 4 ? StatsBase.corspearman([a.hstar for a in S], f.(S)) : NaN
    sized = map(LARGER_SIZES) do N
        S, W = filter(a -> a.N == N && isfinite(a.split), A), filter(a -> a.N == N && isfinite(a.worst), A)
        (; N, split=rank(S, a -> a.split), n_split=length(S), window=rank(W, a -> a.worst), n_window=length(W))
    end
    S, W = filter(a -> isfinite(a.split), A), filter(a -> isfinite(a.worst), A)
    (; sizes=[(; N=2, split=DSTAR_PREDICTS.split.ρ, n_split=DSTAR_PREDICTS.split.n, window=DSTAR_PREDICTS.window.ρ,
            n_window=DSTAR_PREDICTS.window.n); sized],
        pooled=(split=rank(S, a -> a.split), n_split=length(S), window=rank(W, a -> a.worst), n_window=length(W)))
end

# ── cell 76c35ef2 ────────────────────────────────────────────────────────────
"""The larger agents' data against the two-node procedures (`LARGER`): the runner's largest difference in Δx from `src/categorization/nnode_model.jl`'s trials over the 60 agents, and, run on the 41 two-node agents, the largest differences from their cached relief windows and position information and the number of splits that agree."""
LARGER_CHECK = (; runner=LARGER.reproduce, LARGER.check...)

# ── cell 5c400dda ────────────────────────────────────────────────────────────
"""2D: every agent's split (diamonds) and most damaging relief window (squares) against its push height, ±1 bootstrap standard deviation, in the grey of its brain size."""
panel_dstar_sizes = let xlims = (0.0, 320.0), ylims = (0.0, 275.0)
    two = [(; t.name, N=2, t.hstar, t.split, t.worst) for t in CATEGORY_TIMING]
    larger = [(; a.name, a.N, a.hstar, a.split, a.worst) for a in LARGER.agents]
    A = filter(a -> isfinite(a.hstar), [two; larger])
    p = plot(; title="h* against split and relief window", xlabel="h*", ylabel="height of the object, h",
        xlims, ylims, legend=:topleft, legendfontsize=7)
    plot!(p, [0.0, ylims[2]], [0.0, ylims[2]]; color=INK_MUTED, linestyle=:dash, linewidth=0.8, label="")
    for (key, sdkey, marker, name) in ((:split, :split_sd, :diamond, "split"), (:worst, :worst_sd, :rect, "relief window"))
        for a in sort(A; by=a -> a.N)
            y = getfield(a, key)
            isfinite(y) || continue
            sd = get(BOOT[a.name], sdkey, NaN)                    # ±1 bootstrap standard deviation
            isfinite(sd) && plot!(p, [a.hstar, a.hstar], [y - sd, y + sd]; color=SIZE_COLORS[a.N], linewidth=0.8, alpha=0.35, label="")
            scatter!(p, [a.hstar], [y]; color=SIZE_COLORS[a.N], marker, markersize=5, markeralpha=0.45, markerstrokewidth=0,
                label="")
        end
        scatter!(p, [NaN], [NaN]; color=INK_SECONDARY, marker, markersize=5, markerstrokewidth=0, label=name)
    end
    for N in 2:8
        plot!(p, [NaN], [NaN]; color=SIZE_COLORS[N], linewidth=5, label=string(N) * " neurons")
    end
    p
end

# ── cell c01bc822 ────────────────────────────────────────────────────────────
"""Figure 2: the relief deficit (A), the category TE of the agents where the heights line up and where none do (B, C, equally wide) and h* by brain size (D), lettered and saved to `figures/`."""
figure_category = let
    top = plot(panel_relief_deficit, panel_te_aligned; layout=grid(1, 2; widths=[0.4, 0.6]))
    bottom = plot(panel_te_unaligned, panel_dstar_sizes; layout=grid(1, 2; widths=[0.6, 0.4]))
    plt = plot(top, bottom; layout=grid(2, 1), size=(1400, 1050), left_margin=6Plots.mm, bottom_margin=5Plots.mm)
    save_figure(panel_letters!(plt), "Categorization2")
end

# Export every name defined here, so the analysis scripts and the notebook see them all.
eval(Expr(:export, filter(n -> !startswith(string(n), '#') && n ∉ (:eval, :include, nameof(@__MODULE__)),
    names(@__MODULE__; all=true))...))

end
