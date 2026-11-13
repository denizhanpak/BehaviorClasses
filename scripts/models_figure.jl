# The paper's figure of the three models, drawn side by side, each seen from above and facing up towards its task:
# (A) Braitenberg's vehicle (Type B, crossed, with Hotton's forward neuron) and its stimulus, (B) the one-legged walker
# (Beer's, a cockroach with one leg and a single neuron) and (C) a 7-node categorization agent (data/inputs/nnode_agents.jld2, N 7 seed 5, a latched middle
# bringer) under a falling circle, with its brain inside its body. Saves Figures/Models.pdf and .png.
# Run: julia --project scripts/models_figure.jl [draft.png]
using BehaviorClasses, Plots
using BehaviorClasses.BehaviorTools
const NNode = BehaviorClasses.NM      # the n-node agents' model (src/categorization/nnode_model.jl)

figure_style!()

const AGENT = (N=7, seed=5)
const NODE_COLORS = (left="#c51b7d", right="#4d9221")   # CategorizationFigures' left and right neuron
const MODEL_INK = MODEL_COLORS.braitenberg               # the sensorimotor models' orange (vehicle and Walker)
const BODY_FILL = "#f4f3ef"
const OUT = isempty(ARGS) ? joinpath(@__DIR__, "..", "figures", "Models.png") : ARGS[1]   # or a draft path

# ── Drawing helpers (CategorizationFigures.jl's) ───────────────────────────────────────────────────────────────────────

unit_vector(a) = a ./ hypot(a...)
bezier(p0, c1, c2, p1; n=24) = [(1 - s)^3 .* p0 .+ 3(1 - s)^2 * s .* c1 .+ 3(1 - s) * s^2 .* c2 .+ s^3 .* p1
                                for s in range(0, 1; length=n)]
disc(x, y, r; n=60) = Shape(x .+ r .* cos.(range(0, 2π; length=n)), y .+ r .* sin.(range(0, 2π; length=n)))
arc(c, r, a, b; n=40) = [c .+ r .* (cos(t), sin(t)) for t in range(a, b; length=n)]
line!(p, pts; kw...) = plot!(p, first.(pts), last.(pts); label="", kw...)

"""
    draw_edge!(p, pts, w, width; head, color)

An edge along the points `pts`: excitation (w > 0, black) ends in a filled arrowhead, inhibition (w < 0, grey) in a bar
across the edge. `width` is the line width in points and `head` the arrowhead's length in data units.
"""
function draw_edge!(p, pts, w, width; head, color=w > 0 ? INK : INK_MUTED, alpha=1.0, linestyle=:solid)
    tip = pts[end]
    dir = unit_vector(tip .- pts[end-1])
    back = w > 0 ? 0.8head : 0.0
    line = [filter(q -> hypot((tip .- q)...) > back, pts[1:end-1]); [tip .- back .* dir]]
    plot!(p, first.(line), last.(line); color, alpha, linewidth=width, linestyle, label="")
    nrm = (-dir[2], dir[1])
    if w > 0
        b, hw = tip .- head .* dir, 0.45head
        plot!(p, Shape([tip[1], b[1] + hw * nrm[1], b[1] - hw * nrm[1]], [tip[2], b[2] + hw * nrm[2], b[2] - hw * nrm[2]]);
            color, alpha, linewidth=0, label="")
    else
        hw = 0.55head
        plot!(p, [tip[1] + hw * nrm[1], tip[1] - hw * nrm[1]], [tip[2] + hw * nrm[2], tip[2] - hw * nrm[2]];
            color, alpha, linewidth=max(2.0, width), label="")
    end
    p
end

"""A neuron: a white disc of radius `r` at `q`, outlined in `color`, with `label` inside."""
function neuron!(p, q, r; label="", color=INK_SECONDARY, linewidth=1.6, fontsize=9)
    plot!(p, disc(q..., r); color=:white, linecolor=color, linewidth, label="")
    isempty(label) || annotate!(p, q..., text(label, fontsize, INK, :center))
    p
end

"""A self-loop on the neuron at `q` (radius `r`), on the side at angle `th`, reaching `reach` beyond its rim."""
function loop_edge!(p, q, r, th, w, width; head, reach, spread=0.35, kw...)
    at(a, d) = q .+ d .* (cos(a), sin(a))
    pts = bezier(at(th + spread, r), at(th + spread + 0.25, r + reach), at(th - spread - 0.25, r + reach), at(th - spread, r + 0.03r))
    draw_edge!(p, pts, w, width; head, kw...)
end

"""An edge from the neuron at `a` (radius `ra`) to the one at `b` (radius `rb`), bowed by `bow` (× their distance) to its left."""
function chord_edge!(p, a, ra, b, rb, w, width; head, bow=0.0, kw...)
    n = unit_vector((-(b[2] - a[2]), b[1] - a[1]))
    m = (a .+ b) ./ 2 .+ bow * hypot((b .- a)...) .* n
    p0, p1 = a .+ ra .* unit_vector(m .- a), b .- (rb + 0.03rb) .* unit_vector(b .- m)
    draw_edge!(p, bezier(p0, p0 .+ (m .- p0) .* (2 / 3), p1 .+ (m .- p1) .* (2 / 3), p1), w, width; head, kw...)
end

"""A dimension line from `a` to `b` with end ticks and `label` at `at`."""
function dimension!(p, a, b, label, at; tick=0.12)
    n = tick .* unit_vector((-(b[2] - a[2]), b[1] - a[1]))
    line!(p, [a, b]; color=INK_SECONDARY, linewidth=0.8)
    for q in (a, b)
        line!(p, [q .- n, q .+ n]; color=INK_SECONDARY, linewidth=0.8)
    end
    annotate!(p, at..., text(label, 9, INK, :center))
end

blank(title, xlims, ylims) = plot(; title, aspect_ratio=:equal, xlims, ylims, framestyle=:none, grid=false, legend=false)

# ── A. Braitenberg's vehicle ───────────────────────────────────────────────────────────────────────────────────────────

"""
The vehicle (not to scale) at the origin with heading φ, the stimulus ahead and to its right with its field fading
outward. Sensors sₗ, sᵣ at the front corners, crossed synapses fₗ (sₗ → mᵣ) and fᵣ (sᵣ → mₗ), and the forward neuron
V, which drives both wheels (d apart) alike and moves the vehicle at the constant speed V. ψ is the heading error.
"""
function vehicle_panel()
    φ, goal, k = deg2rad(100), (1.6, 7.1), 1.2
    fwd, rgt = (cos(φ), sin(φ)), (sin(φ), -cos(φ))
    W(x, y) = k .* (x .* rgt .+ y .* fwd)                               # vehicle frame (right, forward) → world
    p = blank("Braitenberg vehicle", (-3.2, 4.4), (-3.2, 9.6))
    for r in range(2.3, 0.15; length=12)                                # the stimulus's field, brightest at the source
        plot!(p, disc(goal..., r); color="#f2c14e", alpha=0.09, linewidth=0, label="")
    end
    scatter!(p, [goal[1]], [goal[2]]; marker=:star5, markersize=9, color=INK, markerstrokewidth=0, label="")
    annotate!(p, goal[1] + 0.4, goal[2] + 0.45, text("stimulus", 8, INK_SECONDARY, :left))

    ψ0 = atan(goal[2], goal[1])                                         # heading φ, and the heading error ψ to the goal
    line!(p, [(0.0, 0.0), (3.2, 0.0)]; color=INK_MUTED, linewidth=0.8, linestyle=:dash)
    line!(p, [(0.0, 0.0), goal]; color=INK_MUTED, linewidth=0.8, linestyle=:dash)
    draw_edge!(p, [(0.0, 0.0), 4.3 .* fwd], 1.0, 1.0; head=0.3, color=INK_SECONDARY)
    line!(p, arc((0.0, 0.0), 2.75, 0.0, φ); color=INK_SECONDARY, linewidth=0.8)
    annotate!(p, (3.05 .* (cos(0.32φ), sin(0.32φ)))..., text("φ", 10, INK, :center))
    line!(p, arc((0.0, 0.0), 3.6, ψ0, φ); color=INK_SECONDARY, linewidth=0.8)
    annotate!(p, (3.95 .* (cos((ψ0 + φ) / 2), sin((ψ0 + φ) / 2)))..., text("ψ", 10, INK, :center))

    chassis = [W(-1.05, -1.45), W(1.05, -1.45), W(1.05, 1.2), W(-1.05, 1.2)]
    plot!(p, Shape(first.(chassis), last.(chassis)); color=BODY_FILL, linecolor=INK, linewidth=1.5, label="")
    rn, head = 0.3k, 0.22
    sl, sr, ml, mr, fw = W(-0.72, 1.2), W(0.72, 1.2), W(-0.75, -0.95), W(0.75, -0.95), W(0.0, -0.45)
    for s in (-1, 1)                                                    # the wheels, outside the chassis at the back,
        w = [W(s * 1.08, -1.4), W(s * 1.4, -1.4), W(s * 1.4, -0.5), W(s * 1.08, -0.5)]   # each against its motor
        plot!(p, Shape(first.(w), last.(w)); color=INK_SECONDARY, linewidth=0, label="")
    end
    chord_edge!(p, sl, rn, mr, rn, 1.0, 2.0; head)
    chord_edge!(p, sr, rn, ml, rn, 1.0, 2.0; head)
    chord_edge!(p, fw, 0.24k, ml, rn, 1.0, 1.2; head)
    chord_edge!(p, fw, 0.24k, mr, rn, 1.0, 1.2; head)
    neuron!(p, sl, rn; label="sₗ"); neuron!(p, sr, rn; label="sᵣ")
    neuron!(p, ml, rn; label="mₗ", color=MODEL_INK); neuron!(p, mr, rn; label="mᵣ", color=MODEL_INK)
    neuron!(p, fw, 0.24k; label="V", fontsize=8)
    annotate!(p, W(-0.68, 0.45)..., text("fₗ", 9, INK, :center))
    annotate!(p, W(0.68, 0.45)..., text("fᵣ", 9, INK, :center))
    dimension!(p, W(-1.24, -1.75), W(1.24, -1.75), "d", W(0.0, -2.1))
    p
end

# ── B. The one-legged walker ──────────────────────────────────────────────────────────────────────────────────────────────────────

"""
The one-legged walker from above, a cockroach walking up the page: one leg of length L = 15 on a joint at the body's right side, at angle φ
from the perpendicular (forward > 0), within the joint limits ±30°, drawn in stance (foot down, filled) and, dashed,
in swing (foot up, open). Its one neuron, with self-weight w, reads the leg angle through the sensor gain S_w and drives
the foot and the leg with its output o. The arrows span the fold Walker's swing (w 10, θ −5, τ 0.1: amplitude 0.281
about −0.012 rad).
"""
function walker_panel()
    L, joint = 15.0, (7.0, 0.0)
    φfront, φback = -0.012 + 0.281, -0.012 - 0.281
    φstance, φswing = 0.2, -0.2
    leg(φ, r=L) = joint .+ r .* (cos(φ), sin(φ))
    p = blank("one-legged walker", (-16.0, 34.0), (-21.0, 30.0))

    wedge = [joint; arc(joint, L + 1.5, -π / 6, π / 6)]                # the joint limits
    plot!(p, Shape(first.(wedge), last.(wedge)); color=INK_MUTED, alpha=0.08, linewidth=0, label="")
    for a in (-π / 6, π / 6)
        line!(p, [joint, leg(a, L + 1.5)]; color=INK_MUTED, linewidth=0.6, linestyle=:dot)
    end
    annotate!(p, leg(0.42, 11.6)..., text("joint limits", 7, INK_SECONDARY, :center))
    line!(p, [joint, (joint[1] + L + 1.5, 0.0)]; color=INK_MUTED, linewidth=0.8, linestyle=:dash)
    line!(p, arc(joint, 8.5, 0.0, φstance); color=INK_SECONDARY, linewidth=0.8)
    annotate!(p, leg(φstance / 2, 9.8)..., text("φ", 10, INK, :center))

    draw_edge!(p, arc(joint, L + 3.0, φback, φfront), 1.0, 1.1; head=1.3, color=INK_SECONDARY)   # swing: leg forward
    draw_edge!(p, arc(joint, L + 6.0, φfront, φback), 1.0, 1.1; head=1.3, color=INK_SECONDARY)   # stance: leg back
    annotate!(p, leg(φfront, L + 3.0)[1], leg(φfront, L + 3.0)[2] + 2.4, text("swing, foot up", 8, INK_SECONDARY, :center))
    annotate!(p, leg(φback, L + 6.0)[1], leg(φback, L + 6.0)[2] - 2.4, text("stance, foot down", 8, INK_SECONDARY, :center))

    shape!(pts) = plot!(p, Shape(first.(pts), last.(pts)); color=BODY_FILL, linecolor=INK, linewidth=1.5, label="")
    for s in (-1, 1)                                                    # the cockroach: long antennae sweeping out
        line!(p, bezier((s * 1.2, 19.4), (s * 3.0, 27.0), (s * 11.0, 29.0), (s * 14.5, 22.0)); color=INK, linewidth=1.0)
        line!(p, [(s * 1.2, -15.0), (s * 2.8, -18.6)]; color=INK, linewidth=1.4)   # and back, cerci at the rear,
    end
    shape!([(3.0 * cos(t), 17.4 + 2.5 * sin(t)) for t in range(0, 2π; length=80)])   # the head with its eyes,
    for s in (-1, 1)
        eye = [(s * 2.1 + 0.65 * cos(t), 17.9 + 1.0 * sin(t)) for t in range(0, 2π; length=40)]
        plot!(p, Shape(first.(eye), last.(eye)); color=INK_SECONDARY, linewidth=0, label="")
    end
    e = 2 / 2.6                                                         # a broad, flat oval body (a superellipse)
    shape!([(7.0 * sign(cos(t)) * abs(cos(t))^e, 15.5 * sign(sin(t)) * abs(sin(t))^e) for t in range(0, 2π; length=160)])
    shape!([(6.2 * cos(t), 12.3 + (t < π ? 4.2 : 1.2) * sin(t)) for t in range(0, 2π; length=80)])   # and the pronotum
    line!(p, [joint, leg(φswing)]; color=INK_MUTED, linewidth=2.2, linestyle=:dash)
    plot!(p, disc(leg(φswing)..., 0.9); color=:white, linecolor=INK_MUTED, linewidth=1.4, label="")
    line!(p, [joint, leg(φstance)]; color=INK, linewidth=3)
    plot!(p, disc(leg(φstance)..., 0.9); color=INK, linewidth=0, label="")

    rn, head = 2.3, 1.0
    cell = (0.0, 5.0)
    loop_edge!(p, cell, rn, π, 1.0, 2.0; head, reach=2.6)
    annotate!(p, cell[1] - rn - 3.6, cell[2], text("w", 10, INK, :center))
    chord_edge!(p, joint, 0.9, cell, rn, 1.0, 1.6; head, bow=0.18)     # the leg angle in (bowed below) ...
    chord_edge!(p, cell, rn, joint, 0.9, 1.0, 2.4; head, bow=0.18, color=MODEL_INK)   # ... the output out (above)
    neuron!(p, cell, rn; label="o", color=MODEL_INK, fontsize=11)
    plot!(p, disc(joint..., 0.9); color=:white, linecolor=INK_SECONDARY, linewidth=1.4, label="")
    annotate!(p, 1.7, 0.1, text("\$S_w\$ φ", 9, INK, :center))

    draw_edge!(p, [(-10.5, -9.0), (-10.5, 9.0)], 1.0, 1.4; head=1.6, color=INK_SECONDARY)
    annotate!(p, -12.5, 0.0, text("walks forward in stance", 8, INK_SECONDARY, :center; rotation=90))
    p
end

# ── C. The 7-node categorization agent ─────────────────────────────────────────────────────────────────────────────────

const VIS = NNode.VISUAL

"""The agent's circuit and its Δx when the circle, falling from 275 in the trial from d = 20, is at height `h`."""
function agent_at(h; d=20.0)
    e = only(filter(e -> e.N == AGENT.N && e.seed == AGENT.seed, nnode_agents()))
    c = NNode.ncircuit(e.q, AGENT.N)
    rec = zeros(NNode.nsteps(NNode.DT))
    NNode.final_dx(c, false, d; record=rec)
    c, rec[round(Int, (VIS.start_height - h) / (NNode.DT * VIS.fall_speed))]
end

"""
The task to scale: the object at height `h` above the agent's centre, the circle it catches (diameter 30) and, drawn
over it, the line it would avoid (30 wide), both see-through; the agent's field of view and its seven rays (dark up to
where they hit the circle), the field fading out below the frame's top `top`.
"""
function scene!(p, ax, h; top)
    half, reach, body, R = VIS.visual_angle / 2, VIS.body / 2 + VIS.max_ray, VIS.body / 2, VIS.size / 2
    upto(r, α, y=top) = min(r, y / cos(α))
    a = range(half, -half; length=31)
    for y in range(top - 20, top; length=10)                            # ten stacked wedges: 0.1 in all, fading at the top
        plot!(p, Shape([ax .+ body .* sin.(reverse(a)); ax .+ upto.(reach, a, y) .* sin.(a)], [body .* cos.(reverse(a)); upto.(reach, a, y) .* cos.(a)]);
            color=INK_MUTED, alpha=1 - 0.9^(1 / 10), linewidth=0, label="")
    end
    plot!(p, disc(0.0, h, R; n=120); color="#d9d7cf", fillalpha=0.5, linecolor=INK_SECONDARY, linealpha=0.75, linewidth=1.5, label="")
    line!(p, [(-R, h), (R, h)]; color=INK, linewidth=4.5, alpha=0.35)
    annotate!(p, -R * cos(π / 4) - 1.4, h + R * sin(π / 4) + 1.4, text("circle: catch", 8, INK_SECONDARY, :right))
    annotate!(p, -R - 1.4, h, text("line: avoid", 8, INK_SECONDARY, :right))
    for r in NNode.RAYS
        α = atan(r.ox, r.sy)
        L = NNode.ray_len(r, ax, false, h)
        far = upto(body + L, α)
        hit = L < VIS.max_ray - 1e-6
        line!(p, [(ax + body * sin(α), body * cos(α)), (ax + far * sin(α), far * cos(α))]; color=hit ? INK_SECONDARY : INK_MUTED,
            linewidth=hit ? 1.4 : 0.7, alpha=hit ? 1.0 : 0.6)
        hit && scatter!(p, [ax + far * sin(α)], [far * cos(α)]; color=INK, markersize=2.5, markerstrokewidth=0, label="")
    end
    draw_edge!(p, [(R + 5.0, h + 8), (R + 5.0, h - 8)], 1.0, 1.2; head=2.0, color=INK_SECONDARY)
    annotate!(p, R + 7.0, h, text("falls at 3", 8, INK_SECONDARY, :center; rotation=90))
    plot!(p, disc(ax, 0.0, body; n=160); color=BODY_FILL, linecolor=INK, linewidth=1.5, label="")
    y = 30.0                                                            # beside the ray fan, between body and object
    annotate!(p, ax + body * sin(half) + (y - body * cos(half)) * tan(half) + 1.2, y, text("7 rays\nover 30°", 8, INK_SECONDARY, :left))
    p
end

"""
The 7-node brain inside the body (radius 15, centred at (`ax`, 0)): the seven ray sensors on the rim where the rays
leave the body, and the nodes on a half ellipse in mirror order, L₁ L₂ L₃ C R₃ R₂ R₁ (so no edge between two nodes
passes through a third). Every ray reaches every node and every node every node, itself included; widths ∝ |weight|.
L₁ and R₁ drive the body (v̇ = 5 (o_R₁ − o_L₁) − v): their outputs leave the body as the pushes left and right.
"""
function brain7!(p, c::NNode.NCircuit{3,true}; ax)
    body = VIS.body / 2
    names = ["L₁", "L₂", "L₃", "C", "R₃", "R₂", "R₁"]
    centre = (ax, 3.0)
    pos = [centre .+ (10.8 * cos(t), 10.6 * sin(t)) for t in (π + k * π / 6 for k in 0:6)]
    sensors = [(ax + body * sin(α), body * cos(α)) for α in NNode.ray_angles()]
    rr, rn, head = 0.45, 1.35, 0.8
    # weights as a 7 × 7 matrix W[to, from] and a 7 × 7 ray matrix U[to, ray], in the drawing's node order
    side = [(:L, 1), (:L, 2), (:L, 3), (:C, 0), (:R, 3), (:R, 2), (:R, 1)]
    function weight(to, from)
        (st, i), (sf, k) = side[to], side[from]
        st == :C && sf == :C && return c.WCC
        st == :C && return c.WCi[k]
        sf == :C && return c.WiC[i]
        st == sf ? c.Ws[i][k] : c.Wc[i][k]
    end
    function ray_weight(to, j)
        s, i = side[to]
        s == :C ? c.UC[min(j, 8 - j)] : s == :L ? c.U[i][j] : c.U[i][8-j]
    end
    W = [weight(t, f) for t in 1:7, f in 1:7]
    U = [ray_weight(t, j) for t in 1:7, j in 1:7]
    wmax = maximum(abs, [W[:]; U[:]])
    width(w) = 0.2 + 2.0 * abs(w) / wmax
    for t in 1:7, j in 1:7                                              # rays → nodes, faint so the circuit reads
        v = unit_vector(pos[t] .- sensors[j])
        draw_edge!(p, [sensors[j] .+ rr .* v, pos[t] .- (rn + 0.04) .* v], U[t, j], width(U[t, j]); head, alpha=0.45)
    end
    for t in 1:7, f in 1:7                                              # node → node, each direction bowed to its side
        t == f && continue
        chord_edge!(p, pos[f], rn, pos[t], rn, W[t, f], width(W[t, f]); head, bow=0.1)
    end
    for k in 1:7                                                        # self loops, away from the ellipse's centre
        th = k == 1 ? 3π / 4 : k == 7 ? π / 4 : atan(pos[k][2] - centre[2], pos[k][1] - centre[1])
        loop_edge!(p, pos[k], rn, th, W[k, k], width(W[k, k]); head, reach=2.4)
    end
    for q in sensors
        plot!(p, disc(q..., rr); color=:white, linecolor=INK_SECONDARY, linewidth=1, label="")
    end
    for (j, s) in ((1, -1), (7, 1))
        α = NNode.ray_angles()[j] + s * deg2rad(7)
        annotate!(p, ax + 1.08body * sin(α), 1.08body * cos(α), text(string(j), 6, INK_SECONDARY, :center))
    end
    for (k, s, col, push) in ((1, -1, NODE_COLORS.left, "pushes\nleft"), (7, 1, NODE_COLORS.right, "pushes\nright"))
        x, y = pos[k]
        rim = ax + s * sqrt(body^2 - y^2)
        line!(p, [(x + s * rn, y), (rim, y)]; color=col, linewidth=2.4)
        draw_edge!(p, [(rim + s * 0.6, y), (rim + s * 5.6, y)], 1.0, 1.4; head=1.2, color=INK)
        annotate!(p, rim + s * 4.4, y - 3.0, text(push, 8, INK_SECONDARY, :center))
    end
    for k in 1:7
        col = k == 1 ? NODE_COLORS.left : k == 7 ? NODE_COLORS.right : INK_SECONDARY
        neuron!(p, pos[k], rn; label=names[k], color=col, linewidth=k in (1, 7) ? 2.2 : 1.4, fontsize=8)
    end
    y = -body - 2.8                                                     # the key
    draw_edge!(p, [(ax - 14.5, y), (ax - 10.5, y)], 1.0, 2.0; head)
    annotate!(p, ax - 9.8, y, text("excites", 8, INK, :left))
    draw_edge!(p, [(ax + 0.8, y), (ax + 4.8, y)], -1.0, 2.0; head)
    annotate!(p, ax + 5.6, y, text("inhibits", 8, INK, :left))
    p
end

function agent_panel(; h=56.0, top=73.0)                           # h ≤ 57: all seven rays hit the circle
    c, ax = agent_at(h)
    println("agent N", AGENT.N, " s", AGENT.seed, ": Δx = ", round(ax; digits=2), " at h = ", h)
    p = blank("active categorization agent, 7 neurons", (ax - 25, ax + 25), (-19.5, top))
    scene!(p, ax, h; top)
    brain7!(p, c; ax)
end

# ── The figure ─────────────────────────────────────────────────────────────────────────────────────────────────────────

"""CategorizationFigures.jl's `panel_letters!`: bold A, B, C … at the top left of each panel."""
function panel_letters!(plt; size=14, inset=(1.5, 1.0))
    panels = copy(plt.subplots)
    Plots.prepare_output(plt)
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

plt = plot(vehicle_panel(), walker_panel(), agent_panel(); layout=grid(1, 3; widths=[0.25, 0.40, 0.35]), size=(1300, 730))
panel_letters!(plt)
save_figure(plt, first(splitext(basename(OUT))); dir=dirname(OUT))   # Models.png and Models.pdf, like the notebooks' figures
println("saved ", OUT)
