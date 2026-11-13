"""
    BehaviorTools

Model-independent pieces shared by the notebooks in this directory (the figures of Chapter 2,
"Behavior classes"). The models themselves are MathematicalSystems.jl / HybridSystems.jl
systems; this module only runs them and draws them.

- `trajectory`: a numerical run of an `InitialValueProblem` (`IVP`) of a continuous system
  (by `vector_field`, with OrdinaryDiffEq's Tsit5), of a discrete system (by `successor`), or
  of a `HybridSystem` whose guards are LazySets `HalfSpace`s (each transition fires when the
  state enters its guard; the state is then sent through the transition's reset map).
- The figure style (`figure_style!`, `MODEL_COLORS`) and `save_figure`.
- `wrap_angle`.
- Limit cycles and phase: `shoot_cycle` (the shooting method on a Poincaré return map),
  `asymptotic_phase`, `phase_map` and `isochron_lines!`.

Model pieces shared by two or more notebooks are in `AgentModels.jl`. Include both in the
notebook's first cell:

    begin
        include(joinpath(@__DIR__, "BehaviorTools.jl"))
        include(joinpath(@__DIR__, "AgentModels.jl"))
        using .BehaviorTools, .AgentModels
    end
"""
module BehaviorTools

using LinearAlgebra
using MathematicalSystems
using HybridSystems
using LazySets: HalfSpace
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqTsit5: SciMLBase
using Plots

export trajectory, mode_at, section_callback
export COLORS, MODEL_COLORS, INK, INK_SECONDARY, INK_MUTED, figure_style!, save_figure
export wrap_angle
export shoot_cycle, asymptotic_phase, phase_grid, phase_map, isochron_lines!

# ── Running systems ─────────────────────────────────────────────────────────────────────

const SOLVER_TOLERANCES = (reltol=1e-9, abstol=1e-11)

"""Value of the half-space `H` = {x : a⋅x ≤ b} at `x`: a⋅x − b (negative inside)."""
halfspace_value(H::HalfSpace, x) = dot(H.a, x) - H.b

"""
    section_callback(H, on_entry)

A callback calling `on_entry(integrator)` each time the run enters the half-space `H`, i.e.
crosses its boundary (a Poincaré section) in the direction of `H`.
"""
section_callback(H::HalfSpace, on_entry) = ContinuousCallback((x, t, i) -> halfspace_value(H, x), nothing, on_entry)

"""
    trajectory(IVP(system, x0), tspan; callback = nothing, kwargs...)

Run a continuous system from `x0` over `tspan` with Tsit5 (tolerances 1e-9 / 1e-11 unless
given) and return the ODE solution. `kwargs` go to `solve` (e.g. `saveat`).
"""
function trajectory(prob::InitialValueProblem{<:AbstractContinuousSystem}, tspan; callback=nothing, kwargs...)
    sys = system(prob)
    f(x, p, t) = vector_field(sys, x; check_constraints=false)
    solve(ODEProblem(f, initial_state(prob), tspan), Tsit5(); callback, merge(SOLVER_TOLERANCES, values(kwargs))...)
end

"""
    trajectory(IVP(system, x0), nsteps; stop = x -> false)

Iterate a discrete system `nsteps` times from `x0` (or until `stop(x)`), returning the states.
"""
function trajectory(prob::InitialValueProblem{<:AbstractDiscreteSystem}, nsteps::Integer; stop=x -> false)
    sys = system(prob)
    xs = [initial_state(prob)]
    for _ in 1:nsteps
        stop(xs[end]) && break
        push!(xs, successor(sys, xs[end]))
    end
    xs
end

"""
    trajectory(IVP(hs, (q0, x0)), tspan; on_transition = (integrator, t) -> nothing, interp_points = 10, kwargs...)

Run the hybrid system `hs` from state `x0` in mode `q0`. Mode q follows `vector_field(mode(hs, q), x)`
until the state enters the guard (a `HalfSpace`) of one of q's outgoing transitions t; the state
is then mapped by `resetmap(hs, t)` and the mode becomes `target(hs, t)`, after which
`on_transition(integrator, t)` is called (it may e.g. record the time or `terminate!`). The guards
are watched in every mode, but only those of the current mode's transitions act.
`interp_points` is the number of points per step at which the guards are checked for a crossing
(OrdinaryDiffEq's default, 10); fewer is faster.

The state may be a `Vector` or an immutable static vector (then the modes' vector fields and
the reset maps return static vectors too, and the run allocates almost nothing). All guards
are one `VectorContinuousCallback`, and the problem is fully specialized, so the callback is
compiled for the system rather than called through OrdinaryDiffEq's type-erased wrapper. This
matters for runs from many threads (e.g. a `phase_grid` of runs). For full specialization to
pay off, the modes should share one concrete type (e.g. one closure with a mode flag).

Returns `(; solution, jumps, q0)`: the ODE solution (resets applied in place, so it jumps), and
the `(time, new mode)` of each transition (`mode_at` reads the mode at a time).
"""
function trajectory(prob::InitialValueProblem{<:HybridSystem}, tspan; on_transition=(integ, t) -> nothing, interp_points=10, kwargs...)
    hs = system(prob)
    q0, x0 = initial_state(prob)
    jumps = Tuple{Float64,Int}[]
    ts = collect(transitions(hs))
    guards = [guard(hs, t) for t in ts]
    guard_values!(out, x, t, integ) = foreach(k -> out[k] = halfspace_value(guards[k], x), eachindex(guards))
    function jump!(integ, crossings)
        for (k, t) in enumerate(ts)
            # crossing −1: the value went from positive to negative, i.e. the state entered the guard
            (crossings[k] == -1 && integ.p[] == source(hs, t)) || continue
            x = apply(resetmap(hs, t), integ.u)
            ismutable(integ.u) ? (integ.u .= x) : (integ.u = x)
            integ.p[] = target(hs, t)
            push!(jumps, (integ.t, integ.p[]))
            on_transition(integ, t)
            return
        end
    end
    f(x, q, t) = vector_field(mode(hs, q[]), x; check_constraints=false)
    u0 = ismutable(x0) ? collect(float.(x0)) : float.(x0)
    problem = ODEProblem{false,SciMLBase.FullSpecialize}(f, u0, tspan, Ref(q0))
    solution = solve(problem, Tsit5(); callback=VectorContinuousCallback(guard_values!, jump!, length(ts); interp_points),
        merge(SOLVER_TOLERANCES, values(kwargs))...)
    (; solution, jumps, q0)
end

"""Mode of a hybrid `trajectory` result `run` at time `t`."""
function mode_at(run, t)
    k = findlast(j -> j[1] <= t, run.jumps)
    k === nothing ? run.q0 : run.jumps[k][2]
end

# ── Figure style ────────────────────────────────────────────────────────────────────────

"""
Categorical colours (the first three slots of the validated reference palette; on white the
worst all-pairs CVD ΔE is 9.2, so up to three series may share a plot). Aqua is below 3:1
contrast on white, so plots using it carry a legend and direct labels.
"""
const COLORS = (blue="#2a78d6", orange="#eb6834", aqua="#1baf7a")

"""
Colour of each model, fixed across notebooks: blue for the action–perception laws (Warren,
Bingham), orange for the sensorimotor models (Braitenberg's vehicle, the reflexive walker),
aqua for the central pattern generator walker; circles blue and lines orange.
"""
const MODEL_COLORS = (warren=COLORS.blue, bingham=COLORS.blue, braitenberg=COLORS.orange,
    rpg=COLORS.orange, cpg=COLORS.aqua, circle=COLORS.blue, line=COLORS.orange)

const INK = "#0b0b0b"
const INK_SECONDARY = "#52514e"
const INK_MUTED = "#898781"
const GRID = "#e1e0d9"
const AXIS = "#c3c2b7"

"""Set the Plots defaults every figure here uses: white surface, recessive grid and axes, thin lines."""
function figure_style!()
    default(; background_color=:white, framestyle=:axes, grid=true, gridcolor=GRID, gridalpha=1.0,
        gridlinewidth=0.5, foreground_color_axis=AXIS, foreground_color_border=AXIS,
        tickfontcolor=INK_SECONDARY, guidefontcolor=INK_SECONDARY, titlefontcolor=INK,
        legendfontcolor=INK_SECONDARY, foreground_color_legend=nothing, background_color_legend=nothing,
        tickfontsize=8, guidefontsize=9, titlefontsize=10, legendfontsize=8, linewidth=2, dpi=200)
end

"""Save `plt` as `figures/<name>.png` and `.pdf` at the top of the repository; returns `plt`."""
function save_figure(plt, name; dir=joinpath(@__DIR__, "..", "figures"))
    mkpath(dir)
    savefig(plt, joinpath(dir, name * ".png"))
    savefig(plt, joinpath(dir, name * ".pdf"))
    plt
end

# ── Angles ──────────────────────────────────────────────────────────────────────────────

"""The angle `a` wrapped to [−π, π)."""
wrap_angle(a) = mod(a + π, 2π) - π

# ── Limit cycles and phase ──────────────────────────────────────────────────────────────

"""
    shoot_cycle(step, s0; tol = 1e-10, maxiter = 30, h = 1e-7)

The shooting method for a periodic orbit through a Poincaré section. `step(s)` runs the system
from the section point with free coordinates `s` to its next return to the section and returns
`(s′, T)`. Newton's method solves s′(s) = s, with the return map's Jacobian J by forward
differences (step `h` relative). Returns `(; s, period, residual, iterations, multipliers)`;
`multipliers` are J's eigenvalues at the cycle, its nontrivial Floquet multipliers (all of
modulus < 1 for a stable cycle).
"""
function shoot_cycle(step, s0; tol=1e-10, maxiter=30, h=1e-7)
    s = collect(float.(s0))
    for iteration in 1:maxiter
        s1, period = step(s)
        J = zeros(length(s1), length(s))
        for j in eachindex(s)
            δ = h * max(1, abs(s[j]))
            e = copy(s)
            e[j] += δ
            J[:, j] = (step(e)[1] .- s1) ./ δ
        end
        r = s1 .- s
        if norm(r) <= tol * max(1, norm(s))
            return (; s, period, residual=norm(r), iterations=iteration, multipliers=eigvals(J))
        end
        s = s .- (J - I) \ r
    end
    error("shoot_cycle: no convergence in $maxiter iterations")
end

"""
    asymptotic_phase(times, period; tol = 1e-3)

Asymptotic phase θ ∈ [−π, π) of a point, from the times `times` at which the run from it entered
the Poincaré section that defines phase 0. Once the run has converged onto the cycle the entries
are a period apart, and the point's phase is −2π t_last / T (mod 2π). `NaN` when the last two
entries are not a period apart to within `tol`·T (not converged, or another attractor).
"""
function asymptotic_phase(times, period; tol=1e-3)
    length(times) < 2 && return NaN
    abs(times[end] - times[end-1] - period) > tol * period && return NaN
    wrap_angle(-2π * times[end] / period)
end

"""
    phase_grid(phase_of, xs, ys)

The matrix Θ[i, j] = `phase_of(xs[j], ys[i])` (e.g. an `asymptotic_phase`, or any other quantity
of a run from each grid point, of any type), computed on all threads. Each thread takes the next
cell as it frees up, so cells of very different cost are shared out evenly.
"""
function phase_grid(phase_of, xs, ys)
    T = Base.promote_op(phase_of, eltype(xs), eltype(ys))
    Θ = Matrix{isconcretetype(T) ? T : Any}(undef, length(ys), length(xs))
    Threads.@threads :greedy for c in CartesianIndices(Θ)
        Θ[c] = phase_of(xs[c[2]], ys[c[1]])
    end
    Θ
end

"""
    phase_map(xs, ys, Θ; kwargs...)

Heatmap of the asymptotic phase `Θ[i, j]` at `(xs[j], ys[i])` on the cyclic map `:twilight`
(phase 0 dark, ±π light). NaN cells (no phase: e.g. an unstable equilibrium) are left blank.
"""
phase_map(xs, ys, Θ; kwargs...) = heatmap(xs, ys, Θ; color=:twilight, clims=(-π, π),
    colorbar_title="asymptotic phase (rad)", kwargs...)

"""
    isochron_lines!(plt, xs, ys, Θ; phases = −3π/4:π/4:π, kwargs...)

Draw the isochrons {θ = φ} of the phase grid `Θ` (as in `phase_map`) for each φ in `phases`:
the zero contour of sin(Θ − φ) where cos(Θ − φ) > 0, which avoids the cut at ±π.
"""
function isochron_lines!(plt, xs, ys, Θ; phases=range(-3π / 4, π; step=π / 4), kwargs...)
    for φ in phases
        F = [cos(θ - φ) > 0 ? sin(θ - φ) : NaN for θ in Θ]
        contour!(plt, xs, ys, F; levels=[0.0], color=INK, linewidth=0.8, colorbar_entry=false, kwargs...)
    end
    plt
end

end
