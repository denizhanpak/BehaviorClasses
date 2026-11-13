# Figure Categorization4: gain and damping, the latch threshold, and damping against the odd loop gain (A–D).
# Loading this module draws and saves it.
# The code of notebooks/Categorization.jl, which loads these modules and explains each panel; run there or by
# pipeline/analyze_categorization.jl (Categorization2–4) and pipeline/fit_categorization.jl (Categorization1).
module Categorization4

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

# ── cell 8e436ab6 ────────────────────────────────────────────────────────────
"""
Each two-node agent's edge-law constants, keyed by its short name: its index `k`, `family`, the circle law's `V`, `ε`,
`b` and gain `V/ε` (the law of the edge it keeps, `ALL_FITS`), from `ACTION_MODES` whether it is `latched`, whether
through a `fold` (else the pitchfork), its threshold `dcrit` and its `margin` D − D_crit, and its odd loop gain at the
symmetric rest `odd_gain` (`LARGER.two_node_gain`).
"""
LAW_CONSTANTS = Dict(map(collect(keys(ALL_FITS))) do s
    f = ALL_FITS[s]
    L, m = f.circle[f.e].L, ACTION_MODES[f.k]
    s => (; f.k, f.family, L.V, L.ε, L.b, gain=L.V / L.ε, m.latched, fold=m.mechanism == :fold, m.dcrit, m.margin,
        odd_gain=LARGER.two_node_gain[s])
end)

# ── cell 83d4f935 ────────────────────────────────────────────────────────────
"""The action modes the fits were labelled with (`ALL_FITS`) against `ACTION_MODES`: the agents that differ (none)."""
LAW_CHECK = [s for (s, c) in LAW_CONSTANTS if (ALL_FITS[s].mode == "latched") != c.latched]

# ── cell 37f1e7f5 ────────────────────────────────────────────────────────────
begin
    """The two-node agents named in the panels of §4: the fold latches, and those that lie apart."""
    const NAMED = Set(["M", "P", "O", "K", "Q", "#110", "f12", "#107", "F", "G", "H", "I", "A"])

    """The marker of agent constants `c`: a circle (`shape = :circle`, 4A) or its action mode's (`:mode`, 4B, 4C)."""
    agent_marker(c, shape) = shape == :circle ? :circle : c.latched ? MODE_MARKERS.latched : MODE_MARKERS.symmetric

    """
        constants_scatter!(p, x; shape = :circle, named = NAMED, left = Set{String}())

    One marker per two-node agent at (`x(c)`, b) for its `LAW_CONSTANTS` entry c: colour = family, marker = a circle (with a
    black ring around a fold latch) or the action mode's (`shape`, `agent_marker`), and the agents in `named` named (to the
    left of the marker for those in `left`).
    """
    function constants_scatter!(p, x; shape=:circle, named=NAMED, left=Set{String}())
        for (s, c) in sort(collect(LAW_CONSTANTS); by=first)
            isfinite(x(c)) || continue
            m = agent_marker(c, shape)
            scatter!(p, [x(c)], [c.b]; color=FAMILY_COLORS[c.family], marker=m, markerstrokecolor=:white,
                markerstrokewidth=m == :xcross ? 0 : 0.6, markersize=m == :xcross ? 7 : 6, label="")
            shown = s == AGENT ? "Example" : s                            # §1's agent goes by its figure name
            s in named && annotate!(p, x(c), c.b, s in left ? text(shown * "   ", 7, INK_SECONDARY, :right) :
                                                            text("   " * shown, 7, INK_SECONDARY, :left))
        end
        p
    end

    """The families' colours named by their hold-angle range α₀⁺ (the sign rows' ranges of 3C), in increasing order."""
    const FAMILY_RANGES = [("first", "α₀⁺ 0–5°"), ("middle", "α₀⁺ 10–15°"), ("edge", "α₀⁺ above 15°")]

    """The key of 4A (`shape = :circle`: the colours as circles) or of 4B and 4C (`:mode`: the colours as bars and the action modes' markers), each colour named by its hold-angle range."""
    function constants_key!(p; shape=:circle)
        for (f, range) in FAMILY_RANGES
            shape == :circle ? scatter!(p, [NaN], [NaN]; color=FAMILY_COLORS[f], markerstrokewidth=0, markersize=6, label=range) :
            plot!(p, [NaN], [NaN]; color=FAMILY_COLORS[f], linewidth=5, label=range)
        end
        if shape == :mode
            scatter!(p, [NaN], [NaN]; color=INK_SECONDARY, marker=MODE_MARKERS.latched, markerstrokewidth=0, markersize=6, label="latched")
            scatter!(p, [NaN], [NaN]; color=INK_SECONDARY, marker=MODE_MARKERS.symmetric, markerstrokewidth=0, markersize=8, label="symmetric")
        end
        p
    end
end

# ── cell 1a33d4bf ────────────────────────────────────────────────────────────
"""4A: the edge law's gain V/ε against its damping b, by action mode and family."""
panel_gain_damping = let xlims = (5e-4, 12.0), ylims = (0.05, 250.0)
    p = plot(; xscale=:log10, yscale=:log10, xlims, ylims, xlabel="gain at the hold V/ε", ylabel="damping b",
        title="the edge law's gain and damping", legend=:bottomright, legendfontsize=7)
    hline!(p, [100.0]; color=INK_MUTED, linestyle=:dot, linewidth=0.8, label="")
    annotate!(p, 6e-4, 135.0, text("b's bound in the fits", 7, INK_MUTED, :left))
    constants_scatter!(p, c -> c.gain; left=Set(["I"]))
    constants_key!(p; shape=:circle)
end

# ── cell c3dc7d60 ────────────────────────────────────────────────────────────
"""
The latch threshold D_crit(u*, S) on a grid (as `CategorizationStrategies.jl` §4's, narrower in u*): `u` from −7 to 7
in steps of 0.25 and `S` from −40 to 4 in steps of 1, with `D[i, j]` the threshold at S[i], u[j] (`latch_threshold`,
20 bisections), and `pitchfork[i, j]`, whether it is the pitchfork's 1/σ′(u) to within 0.1%.
"""
LATCH_SURFACE = let u = collect(-7.0:0.25:7.0), S = collect(-40.0:1.0:4.0)
    D = zeros(length(S), length(u))
    Threads.@threads :dynamic for j in eachindex(u)
        for i in eachindex(S)
            D[i, j] = latch_threshold(u[j], S[i]; steps=20)
        end
    end
    (; u, S, D, pitchfork=[abs(D[i, j] * dσ(u[j]) - 1) < 1e-3 for i in eachindex(S), j in eachindex(u)])
end

# ── cell adf5b652 ────────────────────────────────────────────────────────────
"""4B: the latch threshold surface D_crit(u*, S) in 3D with the 41 agents and their drop lines to it."""
panel_latch_surface = let zmax = 45.0, camera = (38, 26)
    L = LATCH_SURFACE
    # the x and y labels are padded with spaces to clear the ticks (GR 3D)
    p = surface(L.u, L.S, min.(L.D, zmax); color=cgrad(["#eef2f8", "#9fb3cf"]), colorbar=false, camera, alpha=0.85,
        linewidth=0.2, xlims=extrema(L.u), ylims=extrema(L.S), zlims=(0, zmax), xlabel="rest input u*        ",
        ylabel="          even weight S", zlabel="odd weight D", title="the latch threshold D_crit(u*, S)", legend=false)
    for c in values(LAW_CONSTANTS)
        r = REST[c.k]
        (L.u[1] <= r.u <= L.u[end] && L.S[1] <= r.S <= L.S[end]) || continue
        D, Dc = clamp(r.D, 0, zmax), clamp(c.dcrit, 0, zmax)
        col, m = FAMILY_COLORS[c.family], agent_marker(c, :mode)
        plot3d!(p, [r.u, r.u], [r.S, r.S], [Dc, D]; color=col, linewidth=1, alpha=0.7)
        scatter3d!(p, [r.u], [r.S], [D]; color=col, marker=m, markersize=m == :xcross ? 5 : 4,
            markerstrokecolor=:white, markerstrokewidth=m == :xcross ? 0 : 0.5)
    end
    p
end

# ── cell f07f33a0 ────────────────────────────────────────────────────────────
"""4C: the edge law's damping b against the odd loop gain − 1, two-node agents."""
panel_damping_margin = let ylims = (0.05, 250.0)
    g = [c.odd_gain - 1 for c in values(LAW_CONSTANTS)]
    xlims = (minimum(g) - 0.15, maximum(g) + 0.25)
    p = plot(; yscale=:log10, xlims, ylims, xlabel="odd loop gain − 1", ylabel="damping b",
        title="damping and the odd loop gain", legend=(0.15, 0.66), legendfontsize=7)     # left, a little above the middle
    vline!(p, [0.0]; color=INK_MUTED, linestyle=:dash, linewidth=0.8, label="")
    annotate!(p, 0.03, 0.07, text("pitchfork →", 7, INK_SECONDARY, :left))
    hline!(p, [100.0]; color=INK_MUTED, linestyle=:dot, linewidth=0.8, label="")
    constants_scatter!(p, c -> c.odd_gain - 1; shape=:mode, named=Set(["M", "P", "O", "Q", "f12", "#107", "F", "G", "H", "I"]))
    constants_key!(p; shape=:mode)
end

# ── cell eb5c95ad ────────────────────────────────────────────────────────────
"""
The damping b of the two action modes (`LAW_CONSTANTS`): its range among the `symmetric` and the `latched` agents, and the
agents on the wrong side of a `cut` at b = 0.4 (`misplaced`).
"""
DAMPING_MODES = let C = LAW_CONSTANTS, cut = 0.4
    (; symmetric=extrema(c.b for c in values(C) if !c.latched), latched=extrema(c.b for c in values(C) if c.latched), cut,
        misplaced=sort([s for (s, c) in C if (c.b > cut) != c.latched]))
end

# ── cell eaf96de3 ────────────────────────────────────────────────────────────
"""
How well the distance from the latch threshold D − D_crit predicts the damping b over the 41 two-node agents:
Spearman's ρ (StatsBase; the b at the fits' bound count as ties) over the agents with a finite threshold, `n`.
"""
MARGIN_PREDICTS = let C = filter(c -> isfinite(c.margin), collect(values(LAW_CONSTANTS)))
    (; ρ=StatsBase.corspearman([c.margin for c in C], [c.b for c in C]), n=length(C))
end

# ── cell 7b5a6696 ────────────────────────────────────────────────────────────
"""
How well the odd loop gain's distance from the pitchfork, gain − 1, predicts the damping b, for each brain size 2 … 8:
Spearman's ρ (StatsBase; the b at the fits' bound count as ties) over the agents of that size (`ρ`, `n`): the 41
two-node agents (`LAW_CONSTANTS`) and ten of each larger size (`LARGER`). `pooled`: over the 60 larger agents together.
"""
MARGIN_BY_SIZE = let two = collect(values(LAW_CONSTANTS)), A = LARGER.agents
    rank(g, b) = (ρ=StatsBase.corspearman(g .- 1, b), n=length(g))
    (; sizes=[(; N=2, rank([c.odd_gain for c in two], [c.b for c in two])...);
            [(; N, rank([a.odd_gain for a in A if a.N == N], [a.b for a in A if a.N == N])...) for N in LARGER_SIZES]],
        pooled=rank([a.odd_gain for a in A], [a.b for a in A]))
end

# ── cell a3beab6e ────────────────────────────────────────────────────────────
"""4D: every agent's damping b against its odd loop gain − 1, with bootstrap 95% intervals for b, in the grey of its brain size."""
panel_margin_sizes = let ylims = (5e-4, 250.0)
    two = [(; name=s, N=2, g=c.odd_gain, c.b) for (s, c) in LAW_CONSTANTS]
    larger = [(; a.name, a.N, g=a.odd_gain, a.b) for a in LARGER.agents]
    A = sort([two; larger]; by=a -> a.N)
    p = plot(; title="damping against the odd loop gain, every agent", xlabel="odd loop gain − 1", ylabel="damping b",
        yscale=:log10, ylims, legend=(0.86, 0.86), legendfontsize=7)     # top right, under b's bound
    vline!(p, [0.0]; color=INK_MUTED, linestyle=:dash, linewidth=0.8, label="")
    hline!(p, [100.0]; color=INK_MUTED, linestyle=:dot, linewidth=0.8, label="")
    for a in A
        lo, hi = BOOT[a.name].b_ci
        plot!(p, [a.g - 1, a.g - 1], [lo, hi]; color=SIZE_COLORS[a.N], linewidth=0.8, alpha=0.35, label="")
        scatter!(p, [a.g - 1], [a.b]; color=SIZE_COLORS[a.N], markersize=5, markeralpha=0.45, markerstrokewidth=0, label="")
    end
    for N in 2:8
        plot!(p, [NaN], [NaN]; color=SIZE_COLORS[N], linewidth=5, label=string(N) * " neurons")
    end
    annotate!(p, 0.05, 1.5e-3, text("pitchfork →", 7, INK_SECONDARY, :left))
    p
end

# ── cell 1a9cd3e0 ────────────────────────────────────────────────────────────
"""Figure 4: gain and damping (A), the latch threshold (B), damping against the threshold (C) and by brain size (D), lettered and saved to `figures/`."""
figure_latch = let
    plt = plot(panel_gain_damping, panel_latch_surface, panel_damping_margin, panel_margin_sizes; layout=grid(2, 2),
        size=(1300, 1100), left_margin=6Plots.mm, bottom_margin=5Plots.mm)
    save_figure(panel_letters!(plt), "Categorization4")
end

# Export every name defined here, so the analysis scripts and the notebook see them all.
eval(Expr(:export, filter(n -> !startswith(string(n), '#') && n ∉ (:eval, :include, nameof(@__MODULE__)),
    names(@__MODULE__; all=true))...))

end
