# The Categorization figures' shared pieces: the fits (`ALL_FITS`, data/cache/), the rest states and action modes, agent
# A, the colours, `panel_letters!` and a few geometric helpers; the circuits, their simulator and the edge law come from
# the package (src/categorization/).
# The code of notebooks/Categorization.jl, which loads these modules and explains each panel; run there or by
# pipeline/analyze_categorization.jl (Categorization2–4) and pipeline/fit_categorization.jl (Categorization1).
module CategorizationCommon

using Markdown
using MathematicalSystems
import HybridSystems, LazySets
using OrdinaryDiffEqTsit5
using Plots
using LinearAlgebra
import JLD2
import KernelDensity, StatsBase, Roots, Random
# The package (src/BehaviorClasses.jl), loaded by whoever loads these modules: the pipeline (`using BehaviorClasses`) or
# the notebook (which includes its source). The two-node circuits, their simulator, the edge law, the relief deficit and
# the rest states are its own (src/categorization/), shared with the pipeline.
using ..BehaviorClasses.BehaviorTools, ..BehaviorClasses.AgentModels
import ..BehaviorClasses: CIRCUITS, Circuit, DT, LAW_DT, MASS, R_OBJ, SETTLE_STARTS, SIDE, STEPS, SUBSTEPS, TAU_RANGE, VIEW_HALF, VIEW_IN, VIEW_OUT, action_modes, bias_of, circle_loss, circuit, deficit_hold, deficit_path, dσ, geometry, held_forces, hold, in_view, is_latched, is_stable, is_stable_rest, latch_threshold, latched_pairs, law_path, line_geometry, mean_input, mirror_sums!, near_r, newton_rests, nnodes, node_mirror, node_out, outer_points, path_R2, push, relief_deficit, rest_states, run_circuit, run_deficit_law, run_edge_law, settle, stable_forces
figure_style!()

# ── cell 0cea0c15 ────────────────────────────────────────────────────────────
begin
    # Statistics is not in this notebook's environment: the few summaries it needs.
    mean(f, xs) = sum(f, xs) / length(xs)
    mean(xs) = sum(xs) / length(xs)
    median(xs) = (s = sort(collect(xs)); n = length(s); isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2)
    """A markdown table from a header and rows of strings."""
    md_table(header, rows) = Markdown.parse(join(["| " * join(header, " | ") * " |",
            "|" * join(fill(":--", length(header)), "|") * "|",
            ("| " * join(r, " | ") * " |" for r in rows)...], "\n"))
    fmt(x; d=2) = isnan(x) ? "–" : d == 0 ? string(round(Int, x)) : string(round(x; digits=d))
    """The range of `xs` as text, "10.55–13.27" (one value when both ends round alike): one interpolation in md text (two joined by a dash read as inline maths)."""
    span(xs; d=2) = (a = fmt(minimum(xs); d); b = fmt(maximum(xs); d); a == b ? a : a * "–" * b)
    """The index in `CIRCUITS` of the circuit with short name `s` ("A", "#16", "f3", …)."""
    circuit_index(s) = findfirst(c -> c.short == s, CIRCUITS)
    """A circle of radius `r` about (`x`, `y`) as a shape."""
    disc(x, y, r; n=60) = Shape(x .+ r .* cos.(range(0, 2π; length=n)), y .+ r .* sin.(range(0, 2π; length=n)))
end

# ── cell c172882d ────────────────────────────────────────────────────────────
begin

    """The smallest time to contact the damping and the inertia read (one step of 0.1): τ^−1 stays finite at contact."""
    const TAU_FLOOR = 0.1

    """
        law_cues(Δx, shape, h, v)

    What the laws read from the two outermost points (`outer_points`), the agent moving at `v`: the time to contact
    τ = ((y_L + y_R)/2 − 15)/3 (at least `TAU_FLOOR`), the bearing offset φ = −(ψ_L + ψ_R)/2 (degrees, ψ a point's
    bearing from vertical, + to the right; φ has Δx's sign) and the optic flow ω, the mean rate of change of −ψ at the
    two points (degrees per time unit), (v y − 3 x)/(x² + y²) for a point at (x, y) falling at 3 past an agent moving
    at v. `nothing` when nothing is in view.
    """
    function law_cues(Δx, shape, h, v; p=VISUAL)
        pts = outer_points(Δx, shape, h; p)
        pts === nothing && return nothing
        xL, yL, xR, yR = pts
        τ = max(((yL + yR) / 2 - p.body / 2) / p.fall_speed, TAU_FLOOR)
        φ = -(atand(xL / yL) + atand(xR / yR)) / 2
        ω = rad2deg(((v * yL - p.fall_speed * xL) / (xL^2 + yL^2) + (v * yR - p.fall_speed * xR) / (xR^2 + yR^2)) / 2)
        (; τ, φ, ω)
    end
end

# ── cell 2ae963e6 ────────────────────────────────────────────────────────────
begin
    """The heights of the circle's centre at the 868 recorded positions of a trial (275 down to 14.9)."""
    const TRIAL_H = collect(VISUAL.start_height .- VISUAL.dt * VISUAL.fall_speed .* (0:STEPS-1))

    """The time to contact τ = (h − 15)/3: when the circle's centre would reach the agent's rim."""
    time_to_contact(h) = (h - VISUAL.body / 2) / VISUAL.fall_speed
    """The angle of one edge of the circle from the agent's midline: the outer edge |φ| + β (e = 1), the inner |φ| − β (e = −1)."""
    edge_angle(Δx, h, e) = (g = geometry(Δx, h); abs(g[1]) + e * g[2])
    """Where the edge `e` sits on the angle α₀ at height h: |Δx| = h tan α₀ − e · 15/cos α₀ (a straight line in (Δx, h))."""
    locus_dx(α0, e, h) = h * tand(α0) - e * R_OBJ / cosd(α0)
    """How fast an agent on the line moves inward: 3 tan α₀."""
    drift_speed(α0) = VISUAL.fall_speed * tand(α0)
    """Where the outer edge's line reaches the centre: h = 15/sin α₀."""
    centring_height(α0) = R_OBJ / sind(α0)
end

# ── cell 3b5594dc ────────────────────────────────────────────────────────────
begin
    """The angle of one end of the line from the agent's midline: the outer end |φ| + β (e = 1), the inner |φ| − β (e = −1)."""
    line_end_angle(Δx, h, e) = (g = line_geometry(Δx, h); abs(g[1]) + e * g[2])
end

# ── cell e9d5c789 ────────────────────────────────────────────────────────────
begin
    const LAW_SUB = round(Int, VISUAL.dt / LAW_DT)

end

# ── cell b10c8ed5 ────────────────────────────────────────────────────────────
"""The fits for all 41 circuits (`pipeline/fit_categorization.jl`, step fit_all; two caches merged), keyed by the circuit's short name: its family and action mode, the circle fits for both edges (`circle[e]`), the edge kept `e`, the circle law on lines (`gamma0`) and the deficit fit (`deficit`)."""
ALL_FITS = let a = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "cache", "categorization_edge_law_all.jld2"), "data"),
    f2 = joinpath(@__DIR__, "..", "..", "data", "cache", "categorization_edge_law_all_2.jld2")
    isfile(f2) ? merge(a, JLD2.load(f2, "data")) : a
end

# ── cell 3d73a07d ────────────────────────────────────────────────────────────
"""
The symmetric rest of each circuit of `CIRCUITS` with nothing in view (§1): its `name`, bias `θ`, self and cross weights
`Ws` and `Wc`, the even and odd weights `S` and `D`, the net input `u` = u* (the one solution of u = θ + S σ(u)), the
output `o` = σ(u*), and the odd and even gains `odd` = D σ′(u*) and `even` = S σ′(u*).
"""
REST = rest_states(CIRCUITS)

# ── cell a884803e ────────────────────────────────────────────────────────────
"""
The action mode of each circuit (§2): its symmetric rest (`REST`), whether it is `latched`, and through what `mechanism`:
`:pitchfork` when its symmetric rest is unstable (odd gain above 1), `:fold` when it latches although its symmetric rest
is stable, else `:symmetric`. Also its latch threshold `dcrit` = D_crit(u*, S) and its `margin` D − D_crit.
"""
ACTION_MODES = action_modes(REST)

# ── cell 80b8559c ────────────────────────────────────────────────────────────
begin
    """The agent's paths (black) and its edge law's (red), in every panel. Red also marks where the law's push starts (§2)."""
    const AGENT_COLOR = INK
    const LAW_COLOR = "#c0122c"

    """The action modes' colours, as in `CategorizationStrategies.jl`: blue latched, orange symmetric."""
    const MODE_COLORS = (latched=COLORS.blue, symmetric=COLORS.orange)
    mode_color(k) = ACTION_MODES[k].latched ? MODE_COLORS.latched : MODE_COLORS.symmetric

    """
    Pushed left (pink) and right (green), ColorBrewer's PiYG ends as in `ActiveCategorizationTwoNodeAgents.jl`: the force
    map (§3) and the left and right node (§1), since o_L pushes left and o_R right.
    """
    const HELD_COLORS = (left="#c51b7d", left_light="#e9a3c9", none="#dedcd6", right_light="#a1d76a", right="#4d9221")
    const HELD_SCALE = cgrad([HELD_COLORS.left, HELD_COLORS.left_light, HELD_COLORS.none, HELD_COLORS.right_light,
        HELD_COLORS.right], [0, 0.25, 0.5, 0.75, 1])

    """The two-node agents' sign rows in the order of their ranges' angles (§3): −++ (0–5°), −+− (10–15°), +−− (beyond 15°), and their colours (`ActiveCategorizationTwoNodeSearch.jl`)."""
    const BAND_ORDER = ["−++", "−+−", "+−−"]
    """Every sign row of the agents of all sizes (3C, 3D), each next to the two-node row whose range it shares: −−+ holds like −++, ++− like −+−, and −−− has no range."""
    const ROW_ORDER = ["−++", "−−+", "−+−", "++−", "+−−", "−−−"]
    const ROW_COLORS = Dict("−+−" => "#6250d6", "+−−" => "#1baf7a", "−++" => "#eda100", "−−+" => "#9b5523",
        "++−" => "#3f9fd6", "−−−" => "#8a3a7a")   # the six pass the dataviz validator (all pairs, light)
    """Spaced signs for labels: "−+−" → "− + −"."""
    spaced(row) = join(collect(row), " ")

    """The families' names."""
    const FAMILY_NAME = Dict("middle" => "middle bringer", "first" => "middle first", "edge" => "edge tracker")
    const FAMILY_PLURAL = Dict("middle" => "middle bringers", "first" => "middle-first circuits", "edge" => "edge trackers")
    """The families' colours in §4: those of the sign rows that give them (Figure 3), − + − middle bringers, − + + middle first, + − − edge trackers."""
    const FAMILY_COLORS = Dict("middle" => ROW_COLORS["−+−"], "first" => ROW_COLORS["−++"], "edge" => ROW_COLORS["+−−"])
    """Brain sizes 2 … 8 in 2D and 4D: one warm grey, light (two nodes) to dark (eight); an ordinal ramp that passes the dataviz validator."""
    const SIZE_COLORS = Dict(zip(2:8, ["#b4b1a8", "#9b978d", "#828074", "#6a675d", "#524f47", "#3b3933", "#24221f"]))
    """The action modes' markers in 4B and 4C: a diamond latched, a cross symmetric (4A draws every agent as a circle)."""
    const MODE_MARKERS = (latched=:diamond, symmetric=:xcross)
end

# ── cell b49395e1 ────────────────────────────────────────────────────────────
begin
    """The agent of §1 and of panel 3B: circuit A, a latched middle bringer."""
    const AGENT = "A"
    """Its index in `CIRCUITS`."""
    const AGENT_K = circuit_index(AGENT)
    """Its circuit (`circuit`) and its edge law with the deficit term, (α0, e, V, ε, b, Γ, Dstar) (`ALL_FITS`)."""
    AGENT_CIRCUIT = circuit(CIRCUITS[AGENT_K].q)
    AGENT_LAW = ALL_FITS[AGENT].deficit.L
end

# ── cell c742e58a ────────────────────────────────────────────────────────────
"""
    panel_letters!(plt; size = 14, inset = (1.5, 1.0))

Letter the panels of the figure `plt` A, B, C, … in the order of its subplots (row by row), each in
bold at the top left of its panel's box, `inset` millimetres in from the corner (right, down). The
letters sit on one transparent subplot laid over the whole figure, so that they are placed alike on
2D and 3D panels.
"""
function panel_letters!(plt; size=14, inset=(1.5, 1.0))
    panels = copy(plt.subplots)
    Plots.prepare_output(plt)   # lay the figure out, to find where its panels are
    w, h = Plots.width(plt.layout.bbox), Plots.height(plt.layout.bbox)
    over = length(panels) + 1
    plot!(plt; inset=bbox(0, 0, 1, 1), subplot=over, framestyle=:none, background_color_subplot=:transparent,
        background_color_inside=:transparent, xlims=(0, 1), ylims=(0, 1), legend=false, grid=false, ticks=false)
    for (k, sp) in enumerate(panels)
        x = (Plots.left(sp.bbox) + inset[1] * Plots.mm) / w
        y = 1 - (Plots.top(sp.bbox) + inset[2] * Plots.mm) / h
        annotate!(plt, x, y, text(string('A' + k - 1), size, INK, :left, :top, "helvetica bold"); subplot=over)
    end
    plt
end

# Export every name defined here, so the analysis scripts and the notebook see them all.
eval(Expr(:export, filter(n -> !startswith(string(n), '#') && n ∉ (:eval, :include, nameof(@__MODULE__)),
    names(@__MODULE__; all=true, imported=true))...))

end
