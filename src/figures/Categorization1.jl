# Figure Categorization1: agent A and its edge law (A–D). Loading this module draws and saves it.
# The code of notebooks/Categorization.jl, which loads these modules and explains each panel; run there or by
# pipeline/analyze_categorization.jl (Categorization2–4) and pipeline/fit_categorization.jl (Categorization1).
module Categorization1

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

# ── cell b3f4a307 ────────────────────────────────────────────────────────────
"""The starts of the paths in 1B and 1C: d = ±10, ±20, …, ±50."""
MODEL_STARTS = [-50.0:10.0:-10.0; 10.0:10.0:50.0]

# ── cell 86e6be7a ────────────────────────────────────────────────────────────
"""
Agent A's paths and its edge law's from `MODEL_STARTS`, for each shape: Δx at the 868 recorded positions (heights
`TRIAL_H`), the agent's from `run_circuit` and the law's from `deficit_path`.
"""
MODEL_PATHS = Dict(s => (agent=[run_circuit(AGENT_CIRCUIT, s, d) for d in MODEL_STARTS],
    law=[deficit_path(AGENT_LAW, d; shape=s) for d in MODEL_STARTS]) for s in (:circle, :line))

# ── cell b1412817 ────────────────────────────────────────────────────────────
"""
Where the paths of 1B and 1C end, for each shape and mover: the range of |Δx| at the end (a circle is caught, and a line
missed, when |Δx| < 30), and `mirror`, the largest |Δx(−d) + Δx(d)| over the trial (0 for a mirror-symmetric mover).
"""
MODEL_ENDS = [(; shape=s, mover=m, ends=extrema(abs(x[end]) for x in MODEL_PATHS[s][m]),
    mirror=maximum(maximum(abs.(MODEL_PATHS[s][m][i] .+ MODEL_PATHS[s][m][end+1-i])) for i in eachindex(MODEL_STARTS)))
              for s in (:circle, :line) for m in (:agent, :law)]

# ── cell b46f445e ────────────────────────────────────────────────────────────
begin
    """The start of the node traces in 1D, d = 20."""
    NODE_START = 20.0

    """Agent A's trial from `NODE_START` under each shape (`run_circuit`, recorded): Δx and the node outputs o_L, o_R at the 868 positions."""
    NODE_TRIALS = Dict(map((:circle, :line)) do s
        c = AGENT_CIRCUIT
        r = run_circuit(c, s, NODE_START; record=true)
        s => (; r.x, oL=[node_out(c, r.Y[:, k], c.L) for k in 1:STEPS], oR=[node_out(c, r.Y[:, k], c.R) for k in 1:STEPS])
    end)
end

# ── cell a6c2dfa8 ────────────────────────────────────────────────────────────
begin
    unit_vector(a) = a ./ hypot(a...)
    """Points along the cubic Bézier (p0, c1, c2, p1)."""
    bezier(p0, c1, c2, p1; n=24) = [(1 - s)^3 .* p0 .+ 3(1 - s)^2 * s .* c1 .+ 3(1 - s) * s^2 .* c2 .+ s^3 .* p1
                                    for s in range(0, 1; length=n)]

    """
        draw_edge!(p, pts, w, width; head, color)

    An edge along the points `pts`: excitation (w > 0, black) ends in a filled arrowhead, inhibition (w < 0, grey) in a
    bar across the edge, as in `ActiveCategorizationV2.jl`'s network. `width` is the line width in points and `head` the
    arrowhead's length in data units.
    """
    function draw_edge!(p, pts, w, width; head, color=w > 0 ? INK : INK_MUTED)
        tip = pts[end]
        dir = unit_vector(tip .- pts[end-1])
        back = w > 0 ? 0.8head : 0.0
        line = [filter(q -> hypot((tip .- q)...) > back, pts[1:end-1]); [tip .- back .* dir]]
        plot!(p, first.(line), last.(line); color, linewidth=width, label="")
        nrm = (-dir[2], dir[1])
        if w > 0
            b, hw = tip .- head .* dir, 0.45head
            plot!(p, Shape([tip[1], b[1] + hw * nrm[1], b[1] - hw * nrm[1]], [tip[2], b[2] + hw * nrm[2], b[2] - hw * nrm[2]]);
                color, linewidth=0, label="")
        else
            hw = 0.55head
            plot!(p, [tip[1] + hw * nrm[1], tip[1] - hw * nrm[1]], [tip[2] + hw * nrm[2], tip[2] - hw * nrm[2]];
                color, linewidth=max(2.0, width), label="")
        end
        p
    end

    """
        agent_scene!(p, ax, h; shape = :circle, top)

    The task on `p`, to scale: the object `shape` with its centre at height `h` above the object's position 0, and the
    agent at `ax` (Δx) as a disc of its true radius (15, the circle's) with its field of view and its seven rays (dark up
    to where they hit), all cut off at the frame's top `top` (the rays reach 220 from the rim, well beyond it). Each ray
    is drawn from where it leaves the body, at its sensor (`brain!`), along its true angle.
    """
    function agent_scene!(p, ax, h; shape=:circle, top)
        half, reach, body = VISUAL.visual_angle / 2, VISUAL.body / 2 + VISUAL.max_ray, VISUAL.body / 2
        upto(r, α) = min(r, top / cos(α))                                # a radius cut off at the frame's top
        a = range(half, -half; length=31)
        plot!(p, Shape([ax .+ body .* sin.(reverse(a)); ax .+ upto.(reach, a) .* sin.(a)], [body .* cos.(reverse(a)); upto.(reach, a) .* cos.(a)]);
            color=INK_MUTED, alpha=0.1, linewidth=0, label="")         # the field of view, from the rim up
        for α in ray_angles()
            L = ray_length(α, ax, shape, h)
            far = upto(body + L, α)                                     # from the centre to where the ray ends
            hit = L < VISUAL.max_ray - 1e-6
            plot!(p, [ax + body * sin(α), ax + far * sin(α)], [body * cos(α), far * cos(α)]; color=hit ? INK_SECONDARY : INK_MUTED,
                linewidth=hit ? 1.4 : 0.7, alpha=hit ? 1.0 : 0.6, label="")
            hit && scatter!(p, [ax + far * sin(α)], [far * cos(α)]; color=INK, markersize=2.5, markerstrokewidth=0, label="")
        end
        if shape == :circle
            plot!(p, disc(0.0, h, VISUAL.size / 2; n=120); color="#ecebe6", linecolor=INK_SECONDARY, linewidth=1.5, label="")
        else
            plot!(p, [-VISUAL.size / 2, VISUAL.size / 2], [h, h]; color=INK_SECONDARY, linewidth=3, label="")
        end
        draw_edge!(p, [(-21.0, h + 9), (-21.0, h - 7)], 1.0, 1.2; head=2.2, color=INK_SECONDARY)
        annotate!(p, -23, h + 1, text("falls at 3", 8, INK_SECONDARY, :right))
        plot!(p, disc(ax, 0.0, body; n=120); color="#f4f3ef", linecolor=AGENT_COLOR, linewidth=1.5, label="")
        annotate!(p, ax - top * tan(half) - 1, top - 3, text("7 rays over 30°, 220 long", 8, INK_SECONDARY, :right))
        p
    end

    """
        brain!(p, c; ax, body = VISUAL.body / 2)

    The network of the two-node circuit `c` inside the agent's body (radius `body`, centred at (`ax`, 0)): the seven
    rays' sensors (on the rim where each ray leaves the body, 5° apart; the end ones numbered 1 and 7), the left and right
    neuron (L, R, outlined in the colour of the way they push), every weight as an edge (`draw_edge!`, width ∝ |w|: U
    from the rays, W_s as loops, W_c between the neurons), and each neuron's output leaving the body, below its loop, as
    the push it gives. Below the body: the body's equation and the key. Every length is in units of the body's radius.
    """
    function brain!(p, c; ax, body=VISUAL.body / 2)
        s, u = 0.263body, body / 100                                    # network units (rays one apart) and 1/100 of the body's radius
        T(q) = (ax + s * q[1], s * (q[2] + 1.08))                       # the network's box centred in the body
        rays = [(body * sin(α) / s, body * cos(α) / s - 1.08) for α in ray_angles()]   # the sensors, on the rim
        nodes = [(-1.5, -2.4), (1.5, -2.4)]
        rr, rn = 0.13, 0.92                                             # sensors small enough to sit 5° apart; the motor neurons
        wmax = maximum(abs, [c.W[:]; c.U[:]])
        width(w) = 0.5 + 3.0 * abs(w) / wmax
        head = 0.18s
        for i in 1:2, j in 1:7                                          # rays → neurons, U[i, j]
            v = unit_vector(nodes[i] .- rays[j])
            draw_edge!(p, [T(rays[j] .+ rr .* v), T(nodes[i] .- (rn + 0.03) .* v)], c.U[i, j], width(c.U[i, j]); head)
        end
        for (i, k, bow) in ((1, 2, 0.5), (2, 1, -0.5))                # neuron i → neuron k, W[k, i], bowed apart
            a, b = nodes[i], nodes[k]
            m = (a .+ b) ./ 2 .+ (0.0, bow)
            p0, p1 = a .+ rn .* unit_vector(m .- a), b .- (rn + 0.03) .* unit_vector(b .- m)
            draw_edge!(p, T.(bezier(p0, p0 .+ (m .- p0) .* (2 / 3), p1 .+ (m .- p1) .* (2 / 3), p1)), c.W[k, i],
                width(c.W[k, i]); head)
        end
        for (i, th) in ((1, π), (2, 0.0))                               # self loops, on the outer side
            at(a, d) = nodes[i] .+ d .* (cos(a), sin(a))
            pts = bezier(at(th + 0.35, rn), at(th + 0.6, rn + 1.1), at(th - 0.6, rn + 1.1), at(th - 0.35, rn + 0.03))
            draw_edge!(p, T.(pts), c.W[i, i], width(c.W[i, i]); head)
        end
        for q in rays
            plot!(p, disc(T(q)..., s * rr); color=:white, linecolor=INK_SECONDARY, linewidth=1, label="")
        end
        for (j, side) in ((1, -1), (7, 1))                              # the end sensors' numbers, just outside the rim
            α = ray_angles()[j] + side * deg2rad(7)
            annotate!(p, ax + 1.07body * sin(α), 1.07body * cos(α), text(string(j), 6, INK_SECONDARY, :center))
        end
        for (q, name, col, side, push) in zip(nodes, ("L", "R"), (HELD_COLORS.left, HELD_COLORS.right), (-1, 1),
            ("pushes left", "pushes right"))
            x, y = T(q .+ (0.0, -rn))
            low = T(q .+ (0.0, -rn - 0.5))[2]                           # below the self loop
            rim = ax + side * sqrt(body^2 - low^2)
            plot!(p, [x, x, rim], [y, low, low]; color=col, linewidth=2.4, label="")   # the output, out of the body
            plot!(p, disc(T(q)..., s * rn); color=:white, linecolor=col, linewidth=2.4, label="")
            annotate!(p, T(q)..., text(name, 11, INK, :center))
            draw_edge!(p, [(rim + side * 4u, low), (rim + side * 34u, low)], 1.0, 1.4; head=8u, color=AGENT_COLOR)
            annotate!(p, rim + side * 4u, low - 13u, text(push, 8, INK_SECONDARY, side < 0 ? :right : :left))
        end
        annotate!(p, ax, -body - 16u, text("agent: v̇ = 5 (R − L) − v", 8, INK_SECONDARY, :center))
        y = -body - 40u                                                 # the key
        draw_edge!(p, [(ax - 100u, y), (ax - 72u, y)], 1.0, 2.0; head)
        annotate!(p, ax - 67u, y, text("excites", 8, INK, :left))
        draw_edge!(p, [(ax + 2u, y), (ax + 30u, y)], -1.0, 2.0; head)
        annotate!(p, ax + 35u, y, text("inhibits", 8, INK, :left))
        p
    end
end

# ── cell 5ba10f6a ────────────────────────────────────────────────────────────
"""1A: agent A late in a circle trial, to scale, with its brain inside it."""
panel_agent = let d = 20.0, h = 50.0, top = 70.0
    i = round(Int, (VISUAL.start_height - h) / (VISUAL.dt * VISUAL.fall_speed)) + 1
    ax = MODEL_PATHS[:circle].agent[findfirst(==(d), MODEL_STARTS)][i]
    p = plot(; title="Agent Example", aspect_ratio=:equal, xlims=(ax - 50, ax + 50), ylims=(-24, top), framestyle=:none,
        grid=false, legend=false)
    agent_scene!(p, ax, TRIAL_H[i]; top)
    brain!(p, AGENT_CIRCUIT; ax)
end

# ── cell 2d304703 ────────────────────────────────────────────────────────────
"""
    paths_panel(shape; title, xlims = (-100, 100))

1B and 1C: agent A's paths (black) and its control law's, the edge law (red) under `shape` from `MODEL_STARTS`, Δx against the object's
height, with the catch lines |Δx| = 30 (dotted).
"""
function paths_panel(shape; title, xlims=(-100, 100))
    P = MODEL_PATHS[shape]
    p = plot(; title, xlabel="Δx, agent to object", ylabel="height of the object, h", xlims, ylims=(0, 285),
        legend=:topleft)
    vline!(p, [-30, 30]; color=INK_MUTED, linestyle=:dot, linewidth=0.8, label="")
    for (n, (a, l)) in enumerate(zip(P.agent, P.law))
        plot!(p, a, TRIAL_H; color=AGENT_COLOR, linewidth=1.1, label=n == 1 ? "Agent Example" : "")
        plot!(p, l, TRIAL_H; color=LAW_COLOR, linewidth=1.1, alpha=0.85, label=n == 1 ? "control law" : "")
    end
    p
end

# ── cell 346c4770 ────────────────────────────────────────────────────────────
"""1B: agent A and its edge law under circles."""
panel_circle_paths = paths_panel(:circle; title="circles")

# ── cell 795c7eb7 ────────────────────────────────────────────────────────────
"""1C: agent A and its edge law under lines."""
panel_line_paths = paths_panel(:line; title="lines")

# ── cell 570fe780 ────────────────────────────────────────────────────────────
"""1D: agent A's node outputs over a circle trial (top) and a line trial (bottom) from d = 20."""
panel_nodes = let gap = 0.4
    p = plot(; title="brain of Agent Example", xflip=true, xlims=(15, 275), ylims=(-0.08, 2 + gap + 0.5),
        xlabel="height of the object, h", ylabel="neuron output",
        yticks=([0, 0.5, 1, 1 + gap, 1.5 + gap, 2 + gap], ["0", "", "1", "0", "", "1"]), legend=:topright)
    for (row, s) in enumerate((:line, :circle))
        off = (row - 1) * (1 + gap)
        T = NODE_TRIALS[s]
        hline!(p, [off, off + 1]; color=INK_MUTED, linewidth=0.5, alpha=0.5, label="")
        plot!(p, TRIAL_H, T.oL .+ off; color=HELD_COLORS.left, linewidth=1.4, label=row == 1 ? "left neuron" : "")
        plot!(p, TRIAL_H, T.oR .+ off; color=HELD_COLORS.right, linewidth=1.4, label=row == 1 ? "right neuron" : "")
        annotate!(p, 272, off + 1.12, text(string(s), 9, INK_SECONDARY, :left))
    end
    p
end

# ── cell fed87b83 ────────────────────────────────────────────────────────────
"""Figure 1: agent A and its edge law (A–D), lettered and saved to `figures/`."""
figure_models = let
    plt = plot(panel_agent, panel_circle_paths, panel_line_paths, panel_nodes; layout=grid(2, 2; heights=[0.56, 0.44]), size=(1300, 1050),
        left_margin=6Plots.mm, bottom_margin=5Plots.mm)
    save_figure(panel_letters!(plt), "Categorization1")
end

# Export every name defined here, so the analysis scripts and the notebook see them all.
eval(Expr(:export, filter(n -> !startswith(string(n), '#') && n ∉ (:eval, :include, nameof(@__MODULE__)),
    names(@__MODULE__; all=true))...))

end
