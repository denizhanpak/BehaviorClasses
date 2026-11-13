# The view's closed-form outermost points (te_work/spring_relay/core.jl) and the edge law of CategorizationEdgeLaw.jl
# (te_work/edge_law/law.jl; user, 2026-10-21): under a falling circle the agent holds one edge of the circle at a fixed
# visual angle α₀; the law pushes toward that hold and damps the velocity. Purely geometric: it reads the object's
# centre bearing φ and angular half-width β, never the rays. Lines: the same law on the line's image.

"""Half-angle of the field of view (15°), and the nearest and farthest points in view from the agent's centre (15, 235)."""
const VIEW_HALF = VISUAL.visual_angle / 2
const VIEW_IN, VIEW_OUT = VISUAL.body / 2, VISUAL.body / 2 + VISUAL.max_ray

"""The distance from the agent's centre to the near side of a circle of radius `R` at distance `d`, `δ` off its centre's bearing."""
near_r(d, R, δ) = d * cos(δ) - sqrt(max(R^2 - (d * sin(δ))^2, 0.0))

"""
    outer_points(Δx, shape, h)

The two outermost points of the object `shape` (`:circle` or `:line`, width 30) in view, the object's centre at height
`h` and the agent at `Δx` from it: of the outline points that face the agent, lie within 15° of vertical and 15 to 235
from its centre (the points `field_cues` samples), the one with the smallest bearing and the one with the largest, as
(x_L, y_L, x_R, y_R) from the agent's centre; `nothing` when none is in view. In closed form: a circle's visible part is
the near arc between its tangent points, a line's the segment, each cut by the wedge's edges and the two ranges.
"""
function outer_points(Δx, shape, h; p=VISUAL)
    cx, cy = -Δx, h
    if shape == :line
        cy > 0 && cy <= VIEW_OUT || return nothing
        a, w, reach = p.size / 2, cy * tan(VIEW_HALF), sqrt(VIEW_OUT^2 - cy^2)
        lo, hi = max(cx - a, -w, -reach), min(cx + a, w, reach)
        if cy < VIEW_IN
            r_in = sqrt(VIEW_IN^2 - cy^2)
            abs(lo) < r_in && (lo = r_in)
            abs(hi) < r_in && (hi = -r_in)
        end
        lo <= hi || return nothing
        return (lo, cy, hi, cy)
    end
    R = p.size / 2
    d2 = cx^2 + cy^2
    d = sqrt(d2)
    (d > R && d - R <= VIEW_OUT) || return nothing
    ψc, αt = atan(cx, cy), asin(R / d)
    lo, hi = ψc - αt, ψc + αt
    if d2 - R^2 > VIEW_OUT^2
        δ = acos((d2 + VIEW_OUT^2 - R^2) / (2d * VIEW_OUT))
        lo, hi = max(lo, ψc - δ), min(hi, ψc + δ)
    end
    lo, hi = max(lo, -VIEW_HALF), min(hi, VIEW_HALF)
    if d - R < VIEW_IN
        d2 - R^2 > VIEW_IN^2 || return nothing
        δ = acos((d2 + VIEW_IN^2 - R^2) / (2d * VIEW_IN))
        ψc - δ < lo < ψc + δ && (lo = ψc + δ)
        ψc - δ < hi < ψc + δ && (hi = ψc - δ)
    end
    lo <= hi || return nothing
    rL, rR = near_r(d, R, lo - ψc), near_r(d, R, hi - ψc)
    (rL * sin(lo), rL * cos(lo), rR * sin(hi), rR * cos(hi))
end

# ---------------------------------------------------------------------------------------------------------------------
# The edge law (law.jl)

const R_OBJ = VISUAL.size / 2
const LAW_DT = 0.01
const SUB = round(Int, VISUAL.dt / LAW_DT)
"""The object's height at each of the 868 recorded positions."""
const H = VISUAL.start_height .- VISUAL.dt * VISUAL.fall_speed .* (0:STEPS-1)
"""The six circuits of the edge law's first steps, one per family × action mode."""
const PICKS = ["A", "D", "K", "J", "F", "P"]
"""The edge each circuit holds (step 1, edges): +1 the outer edge |φ| + β (hold H = α₀ − β), −1 the inner |φ| − β (H = α₀ + β)."""
const EDGE = Dict("A" => 1, "D" => 1, "K" => 1, "J" => 1, "F" => -1, "P" => -1)
"""The index in `CIRCUITS` of the circuit with letter `l`."""
index_of(l) = findfirst(c -> c.letter == l, CIRCUITS)

"""The circle's centre bearing φ (degrees, with Δx's sign: > 0 when the circle is on the agent's left) and angular half-width β (degrees), seen from the agent's centre, Δx = agent − object, centre at height h."""
function geometry(Δx, h)
    r = hypot(Δx, h)
    atand(Δx / h), r > R_OBJ ? asind(R_OBJ / r) : 90.0
end
"""The line's image, seen the same way: its two ends at bearings atan((Δx ± 15)/h); φ is their mean and β half their spread."""
function line_geometry(Δx, h)
    a, c = atand((Δx + R_OBJ) / h), atand((Δx - R_OBJ) / h)
    (a + c) / 2, (a - c) / 2
end

"""The bearing at which the law holds the circle's centre: α₀ − β for the outer edge (e = 1), α₀ + β for the inner (e = −1)."""
hold(L, β) = L.α0 - L.e * β

"""The push F = V tanh(φ (H|H| − φ²)/ε): away from the circle inside the hold ±H, toward it outside; toward the centre once H < 0."""
function push(L, φ, β)
    Hb = hold(L, β)
    L.V * tanh(φ * (Hb * abs(Hb) - φ^2) / L.ε)
end

"""Whether any of the object is in view (within ±15° and 15–235 from the agent's centre): geometry, not rays."""
in_view(Δx, h, shape=:circle) = outer_points(Δx, shape, h) !== nothing

"""
    run_edge_law(L, d; shape = :circle, dt = LAW_DT, steps)

Δx over a trial of the edge law `L` = (α0, e, V, ε, b) from rest at offset `d` under a circle (or a line): read the
geometry, the object falls, the velocity takes an exact step of v̇ = F − b v for the push held over it, Δx takes a step
(stopping at the world's edges). Out of view F = 0 and the velocity decays at rate b.
"""
function run_edge_law(L, d; shape=:circle, dt=LAW_DT, steps=round(Int, (STEPS - 1) * VISUAL.dt / dt) + 1)
    p = VISUAL
    ax, oy, v = Float64(d), p.start_height, 0.0
    decay = exp(-L.b * dt)
    xs = zeros(steps)
    xs[1] = ax
    for k in 1:steps-1
        F = 0.0
        if in_view(ax, oy, shape)
            φ, β = shape == :circle ? geometry(ax, oy) : line_geometry(ax, oy)
            F = push(L, φ, β)
        end
        oy -= dt * p.fall_speed
        v = v * decay + F * (1 - decay) / L.b
        ax += dt * v
        if abs(ax) > p.width / 2
            ax, v = clamp(ax, -p.width / 2, p.width / 2), 0.0
        end
        xs[k+1] = ax
    end
    xs
end
"""The law's path at the circuit's 868 recorded positions (every 0.1)."""
law_path(L, d; shape=:circle, dt=LAW_DT) = run_edge_law(L, d; shape, dt)[1:round(Int, VISUAL.dt / dt):end]

"""The law from z = (α₀, log V, log ε, log b) and the edge e: α₀ 0.1–45°, V 10⁻³–10⁴, ε 10⁻³–10⁸ (degrees³), b 10⁻³–10²."""
edge_params(z, e) = (α0=clamp(z[1], 0.1, 45.0), e, V=exp(clamp(z[2], log(1e-3), log(1e4))),
    ε=exp(clamp(z[3], log(1e-3), log(1e8))), b=exp(clamp(z[4], log(1e-3), log(1e2))))

"""A circuit's circle paths from d = 1 … 50 (the other side is their mirror image)."""
circle_paths(k) = [run_circuit(circuit(CIRCUITS[k].q), :circle, Float64(d)) for d in 1:50]
"""The same for a line."""
line_paths(k) = [run_circuit(circuit(CIRCUITS[k].q), :line, Float64(d)) for d in 1:50]

"""The fit loss under circles (the spring/relay fit loss restricted to circles): mean squared Δx difference plus λ times the squared distance past 25 at the end."""
function circle_loss(run, paths; λ=10.0)
    mean(1:50) do d
        x = run(d)
        mean(abs2, x .- paths[d]) + λ * max(abs(x[end]) - 25, 0.0)^2
    end
end
"""The fit loss under lines: mean squared Δx difference plus λ times the squared shortfall from 35 at the end."""
function line_loss(run, paths; λ=10.0)
    mean(1:50) do d
        x = run(d)
        mean(abs2, x .- paths[d]) + λ * max(35 - abs(x[end]), 0.0)^2
    end
end
"""Path R² against the law that never moves (Δx = d throughout)."""
function path_R2(run, paths)
    err = mean(d -> mean(abs2, run(d) .- paths[d]), 1:50)
    null = mean(d -> mean(abs2, d .- paths[d]), 1:50)
    1 - err / null
end
# The same two for a law `L` (its constants, a NamedTuple), as the figures call them: the law run by `law_path`.
circle_loss(L::NamedTuple, paths; λ=10.0) = circle_loss(d -> law_path(L, d), paths; λ)
path_R2(L::NamedTuple, paths) = path_R2(d -> law_path(L, d), paths)

# ---------------------------------------------------------------------------------------------------------------------
# The relief deficit (deficit.jl; the same as CategorizationEdgeLaw.jl's cell): the edge law's hold gains a term in the
# relief the object lacks compared with a circle of the same width. ρ, the object's relative relief: the depth at the
# two edges of its image beyond that of a flat surface facing the agent through the point its centre ray hits, as a
# share of that point's depth. A circle's ρ is tan β exactly (any radius, distance and bearing); a flat line's is ≈ 0.
# The deficit D = tan β − ρ is 0 on a circle and ≈ tan β = 15/h on a line, so it grows only while a line falls.
# Hold H = α₀ − eβ + Γ max(D − D*, 0).

"""The relief deficit D = tan β − ρ of the object at Δx (agent − object) and height h, read from its image."""
function relief_deficit(Δx, h, shape)
    if shape == :circle
        r = hypot(Δx, h)
        r <= R_OBJ && return 0.0
        sβ = R_OBJ / r
        cβ = sqrt(1 - sβ^2)
        d0 = r - R_OBJ                      # the centre ray hits the circle's nearest point; the edges are tangent at r cos β
        return sβ / cβ - (r * cβ - d0 / cβ) / d0
    end
    a, c = atan((Δx + R_OBJ) / h), atan((Δx - R_OBJ) / h)
    φ, β = (a + c) / 2, (a - c) / 2
    d0 = h / cos(φ)                         # the centre ray, at the ends' mean bearing, hits the line here
    tan(β) - ((h / cos(a) + h / cos(c)) / 2 - d0 / cos(β)) / d0
end

"""The hold bearing with the deficit term: α₀ − eβ + Γ max(D − D*, 0) (degrees)."""
deficit_hold(L, β, D) = L.α0 - L.e * β + L.Γ * max(D - L.Dstar, 0.0)

"""`run_edge_law` with the deficit term in the hold; Γ = 0 gives the edge law itself."""
function run_deficit_law(L, d; shape=:circle, dt=LAW_DT, steps=round(Int, (STEPS - 1) * VISUAL.dt / dt) + 1)
    p = VISUAL
    ax, oy, v = Float64(d), p.start_height, 0.0
    decay = exp(-L.b * dt)
    xs = zeros(steps)
    xs[1] = ax
    for k in 1:steps-1
        F = 0.0
        if in_view(ax, oy, shape)
            φ, β = shape == :circle ? geometry(ax, oy) : line_geometry(ax, oy)
            Hb = deficit_hold(L, β, relief_deficit(ax, oy, shape))
            F = L.V * tanh(φ * (Hb * abs(Hb) - φ^2) / L.ε)
        end
        oy -= dt * p.fall_speed
        v = v * decay + F * (1 - decay) / L.b
        ax += dt * v
        if abs(ax) > p.width / 2
            ax, v = clamp(ax, -p.width / 2, p.width / 2), 0.0
        end
        xs[k+1] = ax
    end
    xs
end
"""The deficit law's path at the circuit's 868 recorded positions."""
deficit_path(L, d; shape=:circle, dt=LAW_DT) = run_deficit_law(L, d; shape, dt)[1:round(Int, VISUAL.dt / dt):end]

"""The circle law `L0` with the deficit term from z = (log Γ, D*): Γ 1–10⁵ degrees, D* 0–0.4."""
deficit_params(z, L0) = merge(L0, (Γ=exp(clamp(z[1], 0.0, log(1e5))), Dstar=clamp(z[2], 0.0, 0.4)))

const DEFICIT_GRID = (logΓ=range(0.0, log(1e5); length=41), Dstar=0.0:0.01:0.35)
const DEFICIT_SCALE = [1.0, 0.03]

"""
    fit_deficit(L0, lpaths, cpaths)

Γ and D* fitted to a circuit's line paths with the circle law `L0` frozen: the line loss over a grid of (log Γ, D*),
then CMA-ES from the grid's three best cells at least 0.05 apart in D*. Returns the fit, its scores on lines and
circles, and the grid.
"""
function fit_deficit(L0, lpaths, cpaths)
    on_lines(L) = d -> deficit_path(L, d; shape=:line)
    loss(z) = line_loss(on_lines(deficit_params(z, L0)), lpaths)
    cells = vec(collect(Iterators.product(DEFICIT_GRID.logΓ, DEFICIT_GRID.Dstar)))
    G = fetch.([Threads.@spawn loss([g, D]) for (g, D) in cells])
    order = sortperm(G)
    starts = Int[]
    for i in order
        all(j -> abs(cells[i][2] - cells[j][2]) >= 0.05, starts) && push!(starts, i)
        length(starts) == 3 && break
    end
    runs = map(enumerate(starts)) do (s, i)
        r = cmaes(loss, collect(cells[i]), DEFICIT_SCALE; σ0=0.5, λ=16, maxgen=200, rng=MersenneTwister(s), stall=40)
        (; start=collect(cells[i]), grid_f=G[i], r.p, r.f, r.gens)
    end
    best = runs[argmin([q.f for q in runs])]
    L = deficit_params(best.p, L0)
    lines = (loss=best.f, R2=path_R2(on_lines(L), lpaths), ends=[on_lines(L)(d)[end] for d in 1:50])
    circle_shift = maximum(d -> maximum(abs, deficit_path(L, d) .- law_path(L0, d)), 1:50)
    on_circles = d -> deficit_path(L, d)
    circles = (loss=circle_loss(on_circles, cpaths), R2=path_R2(on_circles, cpaths), shift=circle_shift)
    (; z=best.p, L, lines, circles, grid=reshape(G, length(DEFICIT_GRID.logΓ), length(DEFICIT_GRID.Dstar)),
        runs=[(start=q.start, grid_f=q.grid_f, f=q.f, gens=q.gens) for q in runs])
end

# ---------------------------------------------------------------------------------------------------------------------
# The two-stage fit of one agent's paths (all_fit.jl and nnode_fit.jl, which share it verbatim)

"""The CMA-ES scales of the circle law's free vector: α₀ (degrees), log V, log ε, log b."""
const EDGE_SCALE = [2.0, 0.5, 1.0, 0.5]

"""
    edge_starts(six)

The four CMA-ES starts per edge (all_fit.jl): the six fits of that edge (outer A D K J; inner F P, and A K's constants
with e = −1), from the six circuits' fits `six` (the edge-law fit stage's output).
"""
edge_starts(six) = Dict(1 => [("A", six["A"].z), ("D", six["D"].z), ("K", six["K"].z), ("J", six["J"].z)],
    -1 => [("F", six["F"].z), ("P", six["P"].z), ("A", six["A"].z), ("K", six["K"].z)])

"""
    fit_edge_and_deficit(cpaths, lpaths, starts)

The two stages of all_fit.jl for one agent's circle paths `cpaths` and line paths `lpaths` (d = 1 … 50):
1. circles: the four constants (α₀, V, ε, b) for each edge e = ±1, CMA-ES from the four `starts` of that edge (seeds 1–4);
   the edge with the lower circle loss is kept;
2. lines: Γ and D* with that circle law frozen (`fit_deficit`).
Returns (e, circle, gamma0, deficit) as the scripts store them.
"""
function fit_edge_and_deficit(cpaths, lpaths, starts)
    on_circles(L) = d -> law_path(L, d)
    circle = Dict{Int,Any}()
    for e in (1, -1)
        loss(z) = circle_loss(on_circles(edge_params(z, e)), cpaths)
        runs = map(enumerate(starts[e])) do (i, (from, z0))
            r = cmaes(loss, copy(z0), EDGE_SCALE; σ0=0.5, λ=24, maxgen=300, rng=MersenneTwister(i), stall=60)
            (; from, r.p, r.f, r.gens)
        end
        best = runs[argmin([q.f for q in runs])]
        L = edge_params(best.p, e)
        circle[e] = (; z=best.p, L, loss=best.f, R2=path_R2(on_circles(L), cpaths), ends=[on_circles(L)(d)[end] for d in 1:50],
            starts=[(from=q.from, f=q.f, gens=q.gens) for q in runs])
    end
    e = circle[1].loss <= circle[-1].loss ? 1 : -1
    L0 = merge(circle[e].L, (Γ=0.0, Dstar=0.0))
    off = (loss=line_loss(d -> deficit_path(L0, d; shape=:line), lpaths), R2=path_R2(d -> deficit_path(L0, d; shape=:line), lpaths))
    deficit = fit_deficit(L0, lpaths, cpaths)
    (; e, circle, gamma0=off, deficit)
end
