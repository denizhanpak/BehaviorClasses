# The §7 views of CategorizationStrategies.jl (te_work/relief/s7_cells.jl: views, view_lengths!, run_view), verbatim
# except that `run_view` takes the Euler step as a keyword (`dt`, default `DT`) where the scripts reassigned the
# notebook's globals DT and SUBSTEPS (the category ensembles at dt 0.001).

"""
A view of the falling object (§7): which rays reach it (`hits`, the circle's or the line's), the height its rays are
read at (`depth`, the circle's nearest hit or the line's height) and whether each ray also reads the circle's relief
(`relief`). The circle is (circle, circle, relief) and the line (line, line, no relief); the other six views mix the
two, one kind of information at a time.
"""
const VIEWS = [(hits=hh, depth=dd, relief=rr) for rr in (true, false) for dd in (:circle, :line) for hh in (:circle, :line)]

"""The circle as it is (§7)."""
const CIRCLE_VIEW = (hits=:circle, depth=:circle, relief=true)

"""The line as it is (§7)."""
const LINE_VIEW = (hits=:line, depth=:line, relief=false)

"""The flat circle (§7): the circle's hits and nearest hit, without its relief."""
const FLAT_VIEW = (hits=:circle, depth=:circle, relief=false)

"""A short label for a view: the circle's information it keeps ("edges", "depth", "relief"), or "line" for none."""
function view_label(v)
    kept = [name for (name, on) in (("edges", v.hits == :circle), ("depth", v.depth == :circle), ("relief", v.relief)) if on]
    isempty(kept) ? "line" : length(kept) == 3 ? "circle" : join(kept, " + ")
end

"""
    view_lengths!(L, Lf, ax, oy, view; p = VISUAL)

The seven ray lengths `L` that the agent at offset `ax` reads from the object at height `oy` in `view` (`VIEWS`), and
the flat lengths `Lf`, which the same rays would read from a flat surface at the height they are read at. The circle's
own image gives its nearest hit's height ``Y_{\\mathrm{n}} = \\min_j (15 + L_j)\\cos\\alpha_j`` over the rays that hit
it, and each ray's relief ``\\delta_j = L_j - (Y_{\\mathrm{n}}/\\cos\\alpha_j - 15) \\ge 0``, the length it reads
beyond that flat surface. A ray among the view's hits reads ``Y/\\cos\\alpha_j - 15``, plus ``\\delta_j`` with relief,
where ``Y`` is ``Y_{\\mathrm{n}}`` (the circle's depth) or ``h`` (the line's). A ray that misses, or would read beyond
220, reads 220. The circle and the line themselves are read with their exact `ray_length`s.
"""
function view_lengths!(L, Lf, ax, oy, view; p=VISUAL)
    angles = ray_angles(p)
    if view == LINE_VIEW
        for j in 1:7
            L[j] = Lf[j] = ray_length(angles[j], ax, :line, oy; p)
        end
        return L, Lf
    end
    Yn = Inf
    for j in 1:7
        L[j] = ray_length(angles[j], ax, :circle, oy; p)
        L[j] < p.max_ray && (Yn = min(Yn, (p.body / 2 + L[j]) * cos(angles[j])))
    end
    Y = view.depth == :circle && isfinite(Yn) ? Yn : oy
    for j in 1:7
        δ = L[j] < p.max_ray ? L[j] - (Yn / cos(angles[j]) - p.body / 2) : 0.0
        hit = view.hits == :circle ? L[j] < p.max_ray : ray_length(angles[j], ax, :line, oy; p) < p.max_ray
        flat = Y / cos(angles[j]) - p.body / 2
        Lf[j] = hit ? min(p.max_ray, flat) : p.max_ray
        view == CIRCLE_VIEW && continue
        L[j] = hit ? min(p.max_ray, flat + (view.relief ? δ : 0.0)) : p.max_ray
    end
    L, Lf
end

"""
    run_view(c, d, view; outside = CIRCLE_VIEW, window = (-Inf, Inf), mass = MASS, steps = STEPS, record = false, dt = DT)

`run_circuit` with the object read in `view` (`VIEWS`, `view_lengths!`) while its height is inside `window`, and in
`outside` elsewhere, in the same order of steps (round(0.1/`dt`) Euler steps of `dt` between recorded positions). With
`record` it also returns the inputs `I` and the flat inputs `If` (7 × steps, at each position; ray j's relief is
``r_j = I^{\\mathrm{flat}}_j - I_j``), the node states `Y` (N × steps) and the velocity `v`, as `run_circuit` does.
"""
function run_view(c::Circuit, d, view; outside=CIRCLE_VIEW, window=(-Inf, Inf), mass=MASS, steps=STEPS, record=false, dt=DT)
    substeps = round(Int, VISUAL.dt / dt)
    N, p = nnodes(c), VISUAL
    y, o, I, dy = zeros(N), zeros(N), zeros(7), zeros(N)
    L, Lf = zeros(7), zeros(7)
    decay = exp(-dt / mass)
    ax, oy, v = Float64(d), p.start_height, 0.0
    xs = zeros(steps)
    xs[1] = ax
    Is, Ifs, Y, V = record ? (zeros(7, steps), zeros(7, steps), zeros(N, steps), zeros(steps)) : (nothing, nothing, nothing, nothing)
    for k in 1:steps
        for s in 1:substeps
            view_lengths!(L, Lf, ax, oy, window[1] <= oy <= window[2] ? view : outside; p)
            for j in 1:7
                I[j] = p.input_gain * (p.max_ray - L[j]) / p.max_ray
            end
            record && s == 1 && (Is[:, k] .= I; Ifs[:, k] .= p.input_gain .* (p.max_ray .- Lf) ./ p.max_ray)
            k == steps && break
            for i in 1:N
                o[i] = node_out(c, y, i)
            end
            mirror_sums!(dy, c, o, I)
            y .+= dt .* (dy .- y) ./ c.τ
            oy -= dt * p.fall_speed
            F = p.velocity_gain * (node_out(c, y, c.R) - node_out(c, y, c.L))
            v = F + (v - F) * decay
            ax += dt * v
            if abs(ax) > p.width / 2
                ax, v = clamp(ax, -p.width / 2, p.width / 2), 0.0
            end
        end
        k == steps && break
        xs[k+1] = ax
        record && (Y[:, k+1] .= y; V[k+1] = v)
    end
    record ? (; x=xs, v=V, I=Is, If=Ifs, Y) : xs
end
