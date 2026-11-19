### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ c8e259bd-ee05-5a01-b3f3-041f90e88eba
begin
    # Packages used by the shared modules, named here so Pluto installs them.
    using MathematicalSystems, HybridSystems
    using LazySets: HalfSpace
    using OrdinaryDiffEqTsit5
    using StaticArrays
    using Plots
    using LinearAlgebra

    include(joinpath(@__DIR__, "..", "src", "BehaviorTools.jl"))
    include(joinpath(@__DIR__, "..", "src", "AgentModels.jl"))
    using .BehaviorTools, .AgentModels
    figure_style!()
end

# ╔═╡ 6a668412-35fe-5490-a116-5d17e58f2463
md"""
# Oscillation, version 3: the AP Oscillator and the Walker

Figures for the limb-oscillation example in the chapter's Results (`chap3.tex`, "Limb Oscillation"). Two models of a rhythmically moving limb, each with a stable limit cycle:

- **The AP Oscillator**, Bingham's action–perception (AP) law, driven by the limb's perceived phase, as a MathematicalSystems.jl `BlackBoxContinuousSystem`;
- **The Walker**, Beer's one-legged walker driven by one neuron that senses the leg angle (the reflexive pattern generator, RPG), a sensorimotor model, as a HybridSystems.jl `HybridSystem` with a swing and a stance mode. Its neuron has a fold (self-weight ``w > 4``) that sets the leg's amplitude (`WalkerPeriodAmplitude.jl`).

This notebook follows `OscillationV2.jl` ("V2" below), with two changes. The Walker is the walker of `WalkerPeriodAmplitude.jl` ("WPA" below; its "fold walker"), with the gains of Beer (2010), in place of V2's slow-neuron walker. Figure 2 shows what sets each model's period and amplitude, in place of V2's perceived-phase readouts.

Figures, saved to `Figures/` as PNG and PDF:

1. `Oscillator1`: the flow and the limit cycle of each model, the AP Oscillator's in its plane ``(x, v)`` and the Walker's in its three-dimensional state space ``(\phi, \omega, o)`` (top), and the asymptotic phase and the isochrons around each cycle (bottom).
2. `Oscillator2`: each model's limb over time, with its period and amplitude; amplitude against each model's amplitude parameter; the Walker's period over its neuron's time constant and its body's gains; and period against each model's period parameter.
3. `Oscillator3`: which phases a noisy reading of the Walker's leg angle, its neuron or both leaves open (top), and the transfer entropy between the Walker's leg and its neuron over the cycle (bottom).
4. `Oscillator4`: what the Walker's leg angle and its neuron state tell about the limit-cycle phase a lag later, split into what either tells, what one alone tells and what only the two together tell (top), and the small parts enlarged (bottom).
5. `Oscillator5`: the integrated information decomposition of the Walker's neuron state and leg angle over the cycle, the sixteen parts of its lattice.

Each section computes what it shows and then draws it panel by panel, each `panel_…` cell below the text that describes it. The `figure_…` cells at the bottom of the notebook lay the panels out into the figures, letter the panels A, B, … and save them; the text above each panel begins with its figure and letter (1A, 1B, …). The settings of a panel worth changing (starting states, ranges, the camera) are named at the top of its cell.

Blue is the AP Oscillator and orange the Walker throughout (in figure 3, dark brown is the Walker's neuron as a source of transfer entropy). The phase colour scale is the isochrons' (dark at ``\theta = 0``, light at ``\pm\pi``) wherever it appears.

| symbol | meaning |
|:--|:--|
| ``x``, ``v`` | the AP Oscillator's limb position (rad) and velocity (rad/s) |
| ``\phi_{\mathrm{p}}`` | the AP Oscillator's perceived phase, the four-quadrant angle of the point ``(x, v/\sqrt{k})`` |
| ``\phi``, ``\omega`` | the Walker's leg angle (rad, forward positive) and angular velocity (rad per unit time) |
| ``y``, ``o`` | the Walker's neuron state and output, ``o = \sigma(y + \beta)`` |
| ``d`` | in stance, the foot's distance from the body's axis, ``L\cos\phi`` at the step |
| ``\theta`` | the phase of a state on a cycle, or the asymptotic phase of a state off it (rad) |
| ``T`` | the period of a cycle |
| ``A`` | the amplitude: half the range of the limb's position (the AP Oscillator, ``x``) or of the leg angle (the Walker, ``\phi``) over a cycle |
| ``k_{\mathrm{t}}`` | the Walker's tempo: both body gains ``\times k_{\mathrm{t}}^2`` and the neuron's time constant ``\div k_{\mathrm{t}}`` (§2) |
| ``g`` | a factor on both of the Walker's body gains, ``\alpha_{\max}`` and ``a_{\max}`` (§2) |
"""

# ╔═╡ 9fadde5b-0de4-5f35-be20-f56a9652312e
md"""
## The AP Oscillator

```math
\dot x = v, \qquad \dot v = -b v - k x + c \sin\phi_{\mathrm{p}}, \qquad
\phi_{\mathrm{p}} = \operatorname{atan2}\!\bigl(v/\sqrt{k},\, x\bigr),
```

with stiffness ``k = 2`` s⁻², damping ``b = 0.3`` s⁻¹ and drive strength ``c = 1`` rad/s² (`BinghamModel.nb`). The perceived phase ``\phi_{\mathrm{p}}`` is the four-quadrant angle of the point ``(x, v/\sqrt k)``, so the drive ``c\sin\phi_{\mathrm{p}} = c\,\frac{v/\sqrt k}{\sqrt{x^2 + v^2/k}}`` pushes in the direction of the velocity. On the cycle it exactly cancels the damping: the cycle is the circle of radius ``R = c/(b\sqrt k)`` in ``(x, v/\sqrt k)``, run once in the period ``T = 2\pi/\sqrt k``, and the origin is an unstable equilibrium (§2 derives both).

The chapter writes the drive as ``c\sin(\tan^{-1}\frac{v}{x\sqrt k})``. With the one-argument arctangent this is ``\operatorname{sign}(x)\,c\sin\phi_{\mathrm{p}}``, which drains energy for ``x < 0``, and every run then decays to rest; so does its rewriting with ``\tanh`` (`OscillatoryBehavior.jl`, "Bingham's oscillator"). The four-quadrant angle is Bingham's.
"""

# ╔═╡ 37559631-8146-574d-937f-87937916fa28
"""The AP Oscillator: stiffness `k`, damping `b`, drive strength `c` (`BinghamModel.nb`)."""
BINGHAM = (k=2.0, b=0.3, c=1.0)

# ╔═╡ 9293fc3c-70a1-551f-818c-5c3351bf9d58
"""The AP Oscillator's perceived phase φₚ = atan2(v/√k, x) of the state (x, v)."""
perceived_phase(p, x, v) = atan(v / sqrt(p.k), x)

# ╔═╡ dbfe1397-bd52-565d-828b-3ccf6987d497
"""The AP Oscillator ẍ = −b ẋ − k x + c sin φₚ as a system of state (x, v)."""
bingham(p) = BlackBoxContinuousSystem(u -> [u[2], p.c * sin(perceived_phase(p, u...)) - p.b * u[2] - p.k * u[1]], 2)

# ╔═╡ 936359bc-1138-5c1e-9ea8-c07b766f0d0b
md"""
## The Walker

Beer's one-legged walker (Beer 2010, §2 and §4; `FitnessSpace.jl`), as in WPA §1. The leg has angle ``\phi`` (from the perpendicular to the body, forward positive) and angular velocity ``\omega``. The neuron has state ``y`` and output ``o = \sigma(y + \beta)``, with ``\sigma(u) = 1/(1 + e^{-u})``. It senses the leg angle,
```math
\tau \dot y = -y + w\,o + S_W\,\phi,
```
and the foot is down while ``o > \tfrac{1}{2}``.

- **Swing** (foot up): ``\dot\omega = \alpha_{\max}(1 - 2o)``, which is positive, so the leg is always pushed forward. At a stop it stays put while pushed against it. The body stands still.
- **Stance** (foot down): ``\dot\omega = \frac{a_{\max}(1 - 2o)}{d}\cos^2\phi - 2\omega^2\tan\phi`` while ``\phi_{\min} \le \phi \le \phi_{\max}``, and ``-2\omega^2\tan\phi`` beyond, where the body coasts. The leg is pushed back and the body forward, at ``-d\,\omega/\cos^2\phi``.

The transitions are five reset maps, each a transition of the hybrid automaton with a half-space guard, which fires when the state enters it:

1. **the step** (the foot put down), swing → stance, when ``o`` rises through ½: ``\omega \mapsto 0``, ``d \mapsto L\cos\phi``;
2. **the raise** (the foot lifted), stance → swing, when ``o`` falls through ½: ``\phi`` is clipped to ``[\phi_{\min}, \phi_{\max}]``, and ``\omega`` is kept unless it was;
3. and 4. **the swing stops** at ``\phi_{\max}`` and ``\phi_{\min}``: ``\omega \mapsto 0``;
5. **the stance stop**, the foot ``x_{\min}`` behind the joint: ``\omega \mapsto 0``.

The Walker reaches none of the stops. Its state is ``(o, \phi, \omega)``, three-dimensional, as in the chapter; beside it the hybrid state keeps ``d``, fixed at each step by the leg angle, and the positions of the body and the foot, which only the stance stop reads.

**The fold.** Hold the leg still at angle ``\phi``: the neuron settles where ``\dot y = 0``, on its steady-state curve ``\phi = (\sigma^{-1}(o) - \beta - w\,o)/S_W``. For ``w > 4`` this curve folds: between the leg angles of its two edges it has three outputs, and the middle one is unstable. A fast neuron stays on the lower branch while swing pushes the leg forward, until that branch ends at the fold's front edge; it then jumps up (the step), and stance pushes the leg back along the upper branch to the fold's back edge, where it jumps down (the raise). The cycle is a relaxation oscillation around the fold (WPA §2).

| parameter | value | what it sets |
|:--|:--|:--|
| ``L`` | 15 | leg length |
| ``\alpha_{\max}`` | 1/40 | the leg's angular acceleration in swing at full command |
| ``a_{\max}`` | 1/20 | the push on the body in stance at full command |
| ``\phi_{\max} = -\phi_{\min}`` | ``\pi/6`` | the swing stops |
| ``x_{\min}`` | −20 | the stance stop: the foot at most 20 behind the joint |
| ``S_W`` | ``30/\pi`` | the sensor gain on the leg angle (Beer's ``5\phi/\phi_{\max}``) |
| ``w`` | 10 | the neuron's self-weight: the fold's width |
| ``\beta`` | −5 | the neuron's bias (the chapter's ``\theta``; here ``\theta`` is the phase): ``\beta = -w/2`` centres the fold on ``\phi = 0`` |
| ``\tau`` | 0.1 | the neuron's time constant: a fast neuron |

The gains are Beer's (2010), as in `FitnessSpace.jl` and WPA; V2's walker (from `WalkerDefinition5.nb`) had them the other way round.
"""

# ╔═╡ 5e24b94c-5a42-56a6-bb43-d36508289c42
"""Leg, body and sensor of Beer (2010), §2: length, swing and stance gains, swing stops, stance stop x_min and sensor gain S_W."""
LEG = (L=15.0, αmax=1 / 40, amax=1 / 20, φmin=-π / 6, φmax=π / 6, xmin=-20.0, sw=30 / π)

# ╔═╡ af440e5c-b7ff-550e-9119-deeb3d4c7299
begin
    # modes of the Walker
    SWING, STANCE = 1, 2
    # transitions: the step, the raise, the swing stops at φmax and φmin, the stance stop at x_min
    STEP, RAISE, FRONT_STOP, BACK_STOP, FOOT_STOP = 1, 2, 3, 4, 5
end

# ╔═╡ c6f5c4cb-cc55-57d6-a777-ab4cade819e2
"""
    rpg_walker(w, β, τ; leg = LEG)

Beer's single-neuron walker with self-weight `w`, bias `β` and time constant `τ` (WPA, from
`FitnessSpace.jl`), as a `HybridSystem` with modes `SWING` and `STANCE` and the five transitions
above. State (static vector): neuron state y, leg angle φ, angular velocity ω, stance distance d,
body and foot positions. Both modes are one closure, so `trajectory` is fully specialized.
"""
function rpg_walker(w, β, τ; leg=LEG)
    neuron = ctrnn_system(ctrnn_parameters(SA[float(τ)], SA[float(β)], SMatrix{1,1}(float(w)); g=SA[1.0]))
    ẏ(y, φ) = vector_field(neuron, SA[y], SA[leg.sw * φ])[1]
    function swing(x)
        y, φ, ω = x[1], x[2], x[3]
        α = leg.αmax * (1 - 2σ(y + β))
        held = (φ >= leg.φmax && α > 0) || (φ <= leg.φmin && α < 0)
        SA[ẏ(y, φ), ω, held ? 0.0 : α, 0.0, 0.0, 0.0]
    end
    function stance(x)
        y, φ, ω, d = x[1], x[2], x[3], x[4]
        a = leg.φmin <= φ <= leg.φmax ? leg.amax * (1 - 2σ(y + β)) : 0.0
        SA[ẏ(y, φ), ω, a * cos(φ)^2 / d - 2ω^2 * tan(φ), 0.0, -d * ω / cos(φ)^2, 0.0]
    end
    function reset(k, x)
        y, φ, ω, d, xb, xf = x
        if k == STEP
            φ = clamp(φ, leg.φmin, leg.φmax)
            return SA[-β, φ, 0.0, leg.L * cos(φ), xb, xb + leg.L * sin(φ)]
        elseif k == RAISE
            φc = clamp(φ, leg.φmin, leg.φmax)
            return SA[-β, φc, φc == φ ? ω : 0.0, d, xb, xf]
        elseif k == FRONT_STOP
            return SA[y, leg.φmax, 0.0, d, xb, xf]
        elseif k == BACK_STOP
            return SA[y, leg.φmin, 0.0, d, xb, xf]
        end
        SA[y, φ, 0.0, d, xb, xf]                                     # FOOT_STOP
    end
    e(i) = [j == i ? 1.0 : 0.0 for j in 1:6]
    guards = Dict(STEP => HalfSpace(-e(1), β),                       # o ≥ ½
        RAISE => HalfSpace(e(1), -β),                                # o ≤ ½
        FRONT_STOP => HalfSpace(-e(2), -leg.φmax),                   # φ ≥ φmax
        BACK_STOP => HalfSpace(e(2), leg.φmin),                      # φ ≤ φmin
        FOOT_STOP => HalfSpace(e(6) - e(5), leg.xmin))               # foot − body ≤ x_min
    automaton = GraphAutomaton(2)
    for (k, from, to) in ((STEP, SWING, STANCE), (RAISE, STANCE, SWING), (FRONT_STOP, SWING, SWING),
        (BACK_STOP, SWING, SWING), (FOOT_STOP, STANCE, STANCE))
        add_transition!(automaton, from, to, k)
    end
    field(q) = x -> q == SWING ? swing(x) : stance(x)
    modes = [ConstrainedBlackBoxContinuousSystem(field(SWING), 6, guards[RAISE]),
        ConstrainedBlackBoxContinuousSystem(field(STANCE), 6, guards[STEP])]
    resets = [ConstrainedBlackBoxMap(6, 6, x -> reset(k, x), guards[k]) for k in 1:5]
    HybridSystem(automaton, modes, resets, fill(AutonomousSwitching(), 5))
end

# ╔═╡ bb049417-5b2e-5874-ab69-a3032f748d60
begin
    """The body `leg` with both gains × g."""
    gains(leg, g) = merge(leg, (; αmax=leg.αmax * g, amax=leg.amax * g))
    """
        walker(w, β, τ; k = 1, g = 1, leg = LEG)

    The Walker of self-weight `w`, bias `β` and time constant `τ`, with both body gains × `g`, run at
    tempo `k`: `rpg_walker` with time constant τ/k and both gains × g k² (§2).
    """
    walker(w, β, τ; k=1.0, g=1.0, leg=LEG) = rpg_walker(w, β, τ / k; leg=gains(leg, g * k^2))
end

# ╔═╡ 34cba43a-1415-589a-8890-3db840d56045
"""The Walker: self-weight w = 10, bias β = −w/2 (the fold centred on φ = 0), time constant τ = 0.1 (WPA)."""
FOLD = (w=10.0, β=-5.0, τ=0.1)

# ╔═╡ 4bdf9cb7-ec83-5b85-9a7c-92f932075ead
"""The Walker as a hybrid system. State: y, φ, ω, d, body position, foot position."""
fold_walker = walker(FOLD...)

# ╔═╡ 9e3c912e-1399-59f1-9583-a2fa6f67ec27
"""The Walker's neuron output o in walker state `u`."""
neuron_output(u) = σ(u[1] + FOLD.β)

# ╔═╡ 39dffbd4-ce74-51c4-b66f-c54427af51ca
md"""
## The cycles, by shooting

Each cycle is found by the shooting method (`shoot_cycle`): Newton's method on the return map to a Poincaré section, started from a long run's last crossing. The AP Oscillator's section is the turning point ``v = 0`` of maximal ``x``. The Walker's is the step: just after it ``o = \tfrac{1}{2}``, ``\omega = 0`` and ``d = L\cos\phi``, so a point on the section is fixed by the leg angle alone. The multipliers are the return map's eigenvalues; a modulus below 1 means the cycle attracts.

The phase ``\theta`` of a point on a cycle is ``2\pi t/T`` for the time ``t`` since the section, taken in ``[-\pi, \pi)`` (``\theta = 0`` at the section) in §1 and in ``[0, 2\pi)`` from the step in §3, so that the Walker's stance and swing are each one stretch there.
"""

# ╔═╡ bb2a3a73-893c-5b03-8a2e-3767cc6642ab
begin
    """
    Walker state with neuron output `o` at leg angle φ, the leg at rest (ω = 0) and the foot, if
    down, at distance `d` from the body's axis, by default just placed (d = L cos φ). With o = ½ in
    stance and the default d, it is the state just after a step at φ.
    """
    walker_state(φ, o; d=LEG.L * cos(φ)) = SA[logit(o) - FOLD.β, φ, 0.0, d, 0.0, d * tan(φ)]

    """Times of the steps of the Walker run from state `x` in mode `q` (`SWING` or `STANCE`) for `tmax`, and the run."""
    function step_times(q, x, tmax; kwargs...)
        times = Float64[]
        run = trajectory(IVP(fold_walker, (q, x)), (0.0, tmax); interp_points=3,
            on_transition=(i, t) -> symbol(fold_walker, t) == STEP && push!(times, i.t), kwargs...)
        times, run
    end

    """The Walker's return map on its step section: [leg angle at the step] ↦ ([leg angle at the next step], return time)."""
    function walker_return(s)
        T = Ref(NaN)
        run = trajectory(IVP(fold_walker, (STANCE, walker_state(s[1], 0.5))), (0.0, 1000.0); interp_points=3,
            on_transition=(i, t) -> symbol(fold_walker, t) == STEP && (T[] = i.t; terminate!(i)))
        ([run.solution.u[end][2]], T[])
    end

    """The AP Oscillator's section: entering v ≤ 0, at the turning point of maximal x."""
    BINGHAM_SECTION = HalfSpace([0.0, 1.0], 0.0)

    """The AP Oscillator's return map on its section: [x at the turning point] ↦ ([x at the next one], return time)."""
    function bingham_return(s)
        T = Ref(NaN)
        sol = trajectory(IVP(bingham(BINGHAM), [s[1], 0.0]), (0.0, 100.0);
            callback=section_callback(BINGHAM_SECTION, i -> i.t > 1e-6 && (T[] = i.t; terminate!(i))))
        ([sol.u[end][1]], T[])
    end
end

# ╔═╡ 1ab1462a-11ad-52c7-ae74-db848691f38f
"""Leg angle at the last step of a 2000-long run of the Walker from Beer's reset: neuron state 0, the leg forward at φmax and still, the foot up."""
settled_step_angle = let last = Ref(NaN)
    start = SA[0.0, LEG.φmax, 0.0, LEG.L * cos(LEG.φmax), 0.0, LEG.L * sin(LEG.φmax)]
    trajectory(IVP(fold_walker, (SWING, start)), (0.0, 2000.0); interp_points=3, save_everystep=false,
        on_transition=(i, t) -> symbol(fold_walker, t) == STEP && (last[] = i.u[2]))
    last[]
end

# ╔═╡ 48361514-d110-5f81-8109-0eefc9342f4c
cycles = (bingham=shoot_cycle(bingham_return, [2.0]), rpg=shoot_cycle(walker_return, [settled_step_angle]))

# ╔═╡ d5154c7d-e8ff-58c6-9e4f-350f0734ad2e
"""One period of each cycle from its section point, saved at 801 evenly spaced times (and, for the Walker, at every transition)."""
on_cycle = let n = 801
    Tb, Tr = cycles.bingham.period, cycles.rpg.period
    (bingham=trajectory(IVP(bingham(BINGHAM), [cycles.bingham.s[1], 0.0]), (0.0, Tb); saveat=range(0, Tb; length=n)),
        rpg=trajectory(IVP(fold_walker, (STANCE, walker_state(cycles.rpg.s[1], 0.5))), (0.0, Tr); saveat=range(0, Tr; length=n)))
end

# ╔═╡ 45731442-548e-5dc9-abbe-9415046875c4
let s = on_cycle.rpg.solution, R = BINGHAM.c / (BINGHAM.b * sqrt(BINGHAM.k))
    leg = extrema(u[2] for u in s.u)
    [(model=:bingham, period=round(cycles.bingham.period; digits=4), expected=round(2π / sqrt(BINGHAM.k); digits=4),
            multipliers=round.(abs.(cycles.bingham.multipliers); sigdigits=3), residual=cycles.bingham.residual,
            amplitude=round(cycles.bingham.s[1]; digits=4), expected_amplitude=round(R; digits=4)),
        (model=:fold_walker, period=round(cycles.rpg.period; digits=4), multipliers=round.(abs.(cycles.rpg.multipliers); sigdigits=3),
            residual=cycles.rpg.residual, leg_angles=round.(leg; digits=3), transitions=length(on_cycle.rpg.jumps),
            stance_share=round(on_cycle.rpg.jumps[1][1] / cycles.rpg.period; digits=3),
            walking_speed=round((s.u[end][5] - s.u[1][5]) / cycles.rpg.period; digits=4))]
end

# ╔═╡ 7d639d49-7f6d-5f58-ab45-45b423a11ca9
md"""
## 1. Limit cycles and isochrons

Each model's flow onto its limit cycle, and the asymptotic phase that the cycle gives the states around it. Each panel below comes after the text that describes it; figure 1, at the bottom of the notebook, puts the four together, the flows on top and the isochrons below.

**Asymptotic phase and isochrons.** Every state of a grid is given the phase ``\theta \in [-\pi, \pi)`` of the point of the cycle it converges to, its asymptotic phase: it is run until its remaining distance to the cycle has shrunk by about ``10^{-4}`` (judged from the multipliers), and the phase is read off its last section crossing (`asymptotic_phase`). The isochrons, the sets of states of equal asymptotic phase, are drawn every ``\pi/4``. Blank cells have no phase.
"""

# ╔═╡ 8f5fc848-e32b-5e85-a5c9-b072115b8c14
"""Coordinates of a run's saved states, one vector per function in `coordinates`, with NaN between the two states saved at each jump so that lines break there."""
function broken_path(sol, coordinates...)
    paths = map(_ -> Float64[], coordinates)
    for k in eachindex(sol.t)
        k > 1 && sol.t[k] == sol.t[k-1] && foreach(p -> push!(p, NaN), paths)
        foreach((p, f) -> push!(p, f(sol.u[k])), paths, coordinates)
    end
    paths
end

# ╔═╡ 99601001-0dd8-5301-b63f-e54624b8e240
"""
Periods to run before reading off a phase: enough for the slowest multiplier to shrink a distance by
10⁻⁴, and at least 5 (the Walker's multiplier is tiny, but a state far from its cycle takes a
period to come near it).
"""
settling_periods(c) = clamp(ceil(Int, log(1e-4) / log(maximum(abs, c.multipliers))) + 1, 5, 40)

# ╔═╡ 3fe70423-3ede-5f2a-b638-7daaf4e8d177
"""
Asymptotic phase on a grid of each model's states: the AP Oscillator's in (x, v), the Walker's in (φ, o)
with ω = 0 and the cycle's foot distance d = L cos φ★ (in swing if o < ½, else in stance). NaN
where a state has no phase.
"""
isochrons = let tol = (reltol=1e-7, abstol=1e-9)
    xs, vs = range(-4, 4; length=120), range(-5, 5; length=120)   # even counts: no point at the origin
    Tb, nb = cycles.bingham.period, settling_periods(cycles.bingham)
    function bingham_phase(x, v)
        times = Float64[]
        trajectory(IVP(bingham(BINGHAM), [x, v]), (0.0, nb * Tb); callback=section_callback(BINGHAM_SECTION, i -> push!(times, i.t)), tol...)
        asymptotic_phase(times, Tb)
    end
    φs, os = range(LEG.φmin, LEG.φmax; length=121), range(1 / 240, 239 / 240; length=120)   # cells tiling [0, 1], none centred on the guard o = ½
    Tr, nr = cycles.rpg.period, settling_periods(cycles.rpg)
    d★ = LEG.L * cos(cycles.rpg.s[1])   # the cycle's foot distance, set at its step
    function rpg_phase(φ, o)
        times, _ = step_times(o < 0.5 ? SWING : STANCE, walker_state(φ, o; d=d★), nr * Tr; save_everystep=false, tol...)
        asymptotic_phase(times, Tr)
    end
    (bingham=(x=xs, y=vs, Θ=phase_grid(bingham_phase, xs, vs)), rpg=(x=φs, y=os, Θ=phase_grid(rpg_phase, φs, os)))
end

# ╔═╡ a026ce9f-663d-5380-bacf-44a812d7da5f
[(model=name, points=length(g.Θ), unresolved=count(isnan, g.Θ)) for (name, g) in pairs(isochrons)]

# ╔═╡ 1d325bda-3929-53dc-823c-a1dab00b723e
begin
    """Phases marked on both cycles in §1: θ = −π, −3π/4, …, 3π/4."""
    MARKER_PHASES = range(-π, π; length=9)[1:end-1]

    """Scatter style of the phase markers: each coloured by its phase on the isochrons' colour scale."""
    PHASE_MARKERS = (marker_z=collect(MARKER_PHASES), color=:twilight, clims=(-π, π), markersize=6, markerstrokecolor=INK,
        markerstrokewidth=1, colorbar=false)

    """States of the cycle `c` (from `shoot_cycle`; `sol` is a run of one period from its section point) at the phases `θ`."""
    cycle_at(sol, c, θ) = sol.(mod.(θ, 2π) ./ 2π .* c.period)
end

# ╔═╡ 241b08d5-3a45-534c-8e2c-07cd8fffdc1f
"""
    flow_arrows(sys, xs, vs; shaft = 0.6, head = 0.22)

Direction field of the planar system `sys` on the grid `xs × vs` (ranges): at each grid point but
the origin, an arrow along the flow of the same length in grid steps (`shaft`, with an open head of
side `head`), so that all look alike. All arrows form one series `(x, v)`, NaN between pieces.
"""
function flow_arrows(sys, xs, vs; shaft=0.6, head=0.22)
    h = (step(xs), step(vs))
    turn(a, β) = (cos(β) * a[1] - sin(β) * a[2], sin(β) * a[1] + cos(β) * a[2])
    ax, av = Float64[], Float64[]
    for x in xs, v in vs
        hypot(x / h[1], v / h[2]) > 0.5 || continue   # the origin: no direction
        f = vector_field(sys, [x, v])
        a = (f[1] / h[1], f[2] / h[2]) ./ hypot(f[1] / h[1], f[2] / h[2])   # direction, in grid steps
        tail, tip = (x, v) .- shaft / 2 .* a .* h, (x, v) .+ shaft / 2 .* a .* h
        left, right = (tip .- head .* turn(a, β) .* h for β in (0.45, -0.45))
        append!(ax, [tail[1], tip[1], NaN, left[1], tip[1], right[1], NaN])
        append!(av, [tail[2], tip[2], NaN, left[2], tip[2], right[2], NaN])
    end
    ax, av
end

# ╔═╡ 1adc49de-20b1-5e7b-9294-2715ee344557
"""
    arrowheads_3d!(p, path; lims, camera, spacing, head = 0.035, avoid = nothing, clear = 0.0, kwargs...)

Open arrowheads along `path` (three coordinate vectors, NaN where it breaks) in the 3D plot `p`
with axis limits `lims` and `camera` (azimuth, elevation in degrees, as Plots takes them): one
every `spacing` of the path's length on screen, the first half a spacing in and again after each
break, each drawn in the plane of the screen with sides of length `head`. Lengths are in the plot
box scaled to a unit cube. A head that would come within `clear` (on screen) of a point of `avoid`
(three coordinate vectors) is left out. `kwargs` style the heads.
"""
function arrowheads_3d!(p, path; lims, camera, spacing, head=0.035, avoid=nothing, clear=0.0, kwargs...)
    lo, span = [float(l[1]) for l in lims], [float(l[2] - l[1]) for l in lims]
    in_box(path) = [([x...] .- lo) ./ span for x in zip(path...)]
    az, el = camera
    toward_camera = [cosd(el) * sind(az), -cosd(el) * cosd(az), sind(el)]   # GR's view (az − 90, 90 − el)
    on_screen(a) = a - (a ⋅ toward_camera) * toward_camera
    q = in_box(path)
    marks = isnothing(avoid) ? [] : filter(b -> !any(isnan, b), in_box(avoid))
    hx, hy, hz = Float64[], Float64[], Float64[]
    len, next = 0.0, spacing / 2
    for i in 2:length(q)
        a = on_screen(q[i] - q[i-1])
        any(isnan, a) && ((len, next) = (0.0, spacing / 2); continue)   # a break: start afresh
        len += norm(a)
        while len >= next
            tip = q[i] - (len - next) / norm(a) * (q[i] - q[i-1])
            d, w = normalize(a), normalize(cross(toward_camera, a))
            next += spacing
            any(b -> min(norm(on_screen(tip - b)), norm(on_screen(tip - head * d - b))) < clear, marks) && continue
            for c in (tip - head * (cos(0.45) * d + sin(0.45) * w), tip, tip - head * (cos(0.45) * d - sin(0.45) * w), fill(NaN, 3))
                x = lo .+ c .* span
                push!(hx, x[1]); push!(hy, x[2]); push!(hz, x[3])
            end
        end
    end
    plot!(p, hx, hy, hz; kwargs...)
end

# ╔═╡ 8dd71357-3bee-5ec5-a0d1-e32de585c54b
"""
    isochron_map(g, curve; kwargs...)

The asymptotic phase on the grid `g` (an entry of `isochrons`) with its isochrons every π/4 and the
cycle `curve` (coordinate vectors) in black edged white. `kwargs` go to `phase_map`: title, labels,
limits, colorbar.
"""
function isochron_map(g, curve; kwargs...)
    p = phase_map(g.x, g.y, g.Θ; colorbar=false, colorbar_title="asymptotic phase θ (rad)", xlims=extrema(g.x),
        ylims=extrema(g.y), legend=false, kwargs...)
    isochron_lines!(p, g.x, g.y, g.Θ)
    plot!(p, curve...; color=:white, linewidth=4.5)
    plot!(p, curve...; color=INK, linewidth=2.5)
end

# ╔═╡ eced8e23-18d8-5c59-bec2-67b6a04e2569
md"""
**1A. The AP Oscillator's flow and limit cycle.** The AP Oscillator in its plane ``(x, v)``, with arrows of equal length along the flow. Runs from two starting states (grey) converge onto the cycle (blue) that shooting found, the ellipse ``x^2 + v^2/k = R^2``, around the unstable equilibrium at the origin (open circle). The markers on the cycle are the phases ``\theta = -\pi, -3\pi/4, \dots, 3\pi/4``, on the colour scale of the isochron maps below.
"""

# ╔═╡ aebfa6c1-94c4-5147-bc72-85a6ff778d3c
"""The AP Oscillator in its plane (x, v): arrows along the flow, runs onto the cycle, the cycle with its phase markers, and the unstable origin."""
panel_bingham_flow = let sys = bingham(BINGHAM)
    starts, duration = ([0.6, 0.0], [3.9, -3.0]), 13.0
    p = plot(; title="AP Oscillator", xlabel="x (rad)", ylabel="v (rad/s)", xlims=(-4, 4), ylims=(-5, 5), legend=false)
    plot!(p, flow_arrows(sys, range(-3.6, 3.6; length=13), range(-4.5, 4.5; length=13))...; color=INK_MUTED, linewidth=0.9)
    for x0 in starts
        run = trajectory(IVP(sys, x0), (0.0, duration); saveat=0.01)
        plot!(p, first.(run.u), last.(run.u); color=INK_SECONDARY, linewidth=1)
    end
    plot!(p, first.(on_cycle.bingham.u), last.(on_cycle.bingham.u); color=MODEL_COLORS.bingham, linewidth=3)
    scatter!(p, [0.0], [0.0]; color=:white, markerstrokecolor=INK, markersize=5)
    marks = cycle_at(on_cycle.bingham, cycles.bingham, MARKER_PHASES)
    scatter!(p, first.(marks), last.(marks); PHASE_MARKERS...)
end

# ╔═╡ f0c94cc4-bbfb-52aa-9eab-ec019cf4bc3f
md"""
**1B. The Walker's flow and limit cycle.** The Walker in its state space ``(\phi, \omega, o)``. Runs from three starting states (grey) converge onto the cycle (orange) that shooting found, with arrowheads along both in the direction of travel, and the cycle carries the same phase markers as the AP Oscillator's. The shaded plane ``o = \tfrac{1}{2}`` is the foot's threshold: the Walker is in stance above it and in swing below it, puts its foot down (the step) when it crosses it upwards and lifts it (the raise) when it crosses it downwards. The fast neuron (``\tau = 0.1``) keeps the cycle near ``o = 0`` in swing and ``o = 1`` in stance, and carries it across the plane almost at once, at a nearly constant leg angle: at the fold's front edge (the step) and at its back edge (the raise). At the step ``\omega`` is reset to 0, and the runs break there (dotted on the cycle). The raise keeps ``\omega``: the leg is still moving back as the foot lifts. Starting states are at rest (``\omega = 0``), in swing below the plane and in stance above it.
"""

# ╔═╡ 6081c9f2-fc31-5c71-aad4-9fc3bea06da2
"""
The Walker in its state space (φ, ω, o): the plane o = ½ of the foot's threshold, runs onto
the cycle, and the cycle with its dotted step reset and phase markers, with arrowheads in the
direction of travel.
"""
panel_walker_flow = let s = on_cycle.rpg.solution, T = cycles.rpg.period
    starts = ((-0.45, 0.05), (0.45, 0.6), (-0.1, 0.95))   # (φ, o) at rest: in swing if o < ½, else in stance
    ωlims, camera = (-0.08, 0.2), (55, 20)
    lims = ((LEG.φmin, LEG.φmax), ωlims, (0.0, 1.0))
    coordinates = (u -> u[2], u -> u[3], neuron_output)   # (φ, ω, o) of a walker state
    p = plot(; title="Walker", xlabel="leg angle φ (rad)" * " "^10,   # padded: clear of the 0.4 tick
        ylabel="ω (rad/unit time)", zlabel="neuron output o", xlims=lims[1], ylims=lims[2], zlims=lims[3],
        xticks=-0.4:0.4:0.4, yticks=-0.05:0.1:0.15, camera, legend=false)
    mesh3d!(p, [LEG.φmin, LEG.φmax, LEG.φmax, LEG.φmin], [ωlims[1], ωlims[1], ωlims[2], ωlims[2]], fill(0.5, 4);
        connections=[(1, 2, 3, 4)], color=INK_MUTED, fillalpha=0.15, linealpha=0)
    cycle = broken_path(s, coordinates...)
    marks = map(f -> f.(cycle_at(s, cycles.rpg, MARKER_PHASES)), coordinates)
    for (φ, o) in starts
        _, run = step_times(o < 0.5 ? SWING : STANCE, walker_state(φ, o), 1.5T; saveat=0.05)
        path = broken_path(run.solution, coordinates...)
        plot!(p, path...; color=INK_SECONDARY, linewidth=1)
        arrowheads_3d!(p, path; lims, camera, spacing=0.35, avoid=cycle, clear=0.025, color=INK_SECONDARY, linewidth=1)
    end
    plot!(p, cycle...; color=MODEL_COLORS.rpg, linewidth=3)
    arrowheads_3d!(p, cycle; lims, camera, spacing=0.3, head=0.05, avoid=marks, clear=0.04, color=MODEL_COLORS.rpg,
        linewidth=3)
    k = findfirst(k -> k > 1 && s.t[k] == s.t[k-1] && s.u[k][3] != s.u[k-1][3], eachindex(s.t))   # the step: ω reset to 0
    plot!(p, map(f -> f.(s.u[k-1:k]), coordinates)...; color=MODEL_COLORS.rpg, linestyle=:dot, linewidth=2)
    scatter!(p, marks...; PHASE_MARKERS...)
end

# ╔═╡ 706ecbf6-52f0-5dba-9236-b2cb496b359f
md"""
**1C. The AP Oscillator's isochrons.** The asymptotic phase of the AP Oscillator in ``(x, v)``, the plane its phase lives in, with its cycle (black). Each isochron passes through the marker of its phase on the cycle in 1A. The origin has no phase.
"""

# ╔═╡ 6e363438-2f18-5ec5-af13-60394bd7b7b2
"""The AP Oscillator's asymptotic phase and isochrons in (x, v), with its cycle."""
panel_bingham_isochrons = isochron_map(isochrons.bingham, (first.(on_cycle.bingham.u), last.(on_cycle.bingham.u));
    title="AP Oscillator", xlabel="x (rad)", ylabel="v (rad/s)")

# ╔═╡ c04e9129-06ac-581a-b075-5ad0af9ddf47
md"""
**1D. The Walker's isochrons.** The Walker's asymptotic phase on the slice ``\omega = 0`` of its state space, in ``(\phi, o)`` over the whole range of ``o``: states with ``o < \tfrac{1}{2}`` start in swing, the others in stance. In stance the phase also depends on the foot's distance ``d``, which the step sets and the plotted coordinates do not show. Every state here has the cycle's, ``d = L\cos\phi_\star`` with ``\phi_\star`` the cycle's step angle, so the slice lies in the three-dimensional space ``(\phi, \omega, o)`` of 1B that holds the whole cycle (with the foot just placed instead, ``d = L\cos\phi``, the stance phases would move by about 0.04 rad in mid-stance and by up to 0.2 rad ahead of the step angle). Ahead of the step angle this leaves a seam at ``o = \tfrac{1}{2}``: a swing state there steps at once and puts its foot down at its own angle, ``d = L\cos\phi``, while the stance state just above it keeps the cycle's ``d``. Its cycle is drawn projected onto the slice (it lies on the slice only at the step, where ``\omega`` is reset, and where the leg turns around at the back early in swing). The phase jumps along the diagonal, the middle (unstable) branch of the neuron's steady-state curve: from above it the neuron rises to its upper branch and the foot stays down or goes down at once, from below it the neuron falls to its lower branch and the foot stays up or comes up at once, whichever mode the state starts in. Elsewhere the isochrons run nearly vertically: the fast neuron reaches its branch within a small fraction of the period, so the leg angle and the branch set a state's phase.

The map is checked in `isochron_check/`. Along runs from across the map (2.5 million saved states) the mode always agrees with ``o``, swing below ``\tfrac{1}{2}`` and stance above, so the map's split of the starting states is the hybrid system's own. States of the cycle get back their own phase (to ``2\cdot 10^{-6}``). The cycle's two points on this slice, the step and the leg's backward turn early in swing, get their cycle phases. At tolerance ``10^{-11}`` with twice the settling, the map changes by at most ``3\cdot 10^{-6}`` rad. Reachability bounds on the first step (ReachabilityAnalysis.jl, Taylor-model flowpipes of the same automaton) over 117 whole cells and 96 points contain the map's phase everywhere off the jump. In the two cells on the jump the bounds missed the states that fall, as a BigFloat run of the plain equations shows.
"""

# ╔═╡ 73bc3e0c-725b-5cb7-8e48-d9f507a299c3
"""The Walker's asymptotic phase and isochrons on the slice ω = 0, in (φ, o), with its cycle projected onto it."""
panel_walker_isochrons = let s = on_cycle.rpg.solution
    isochron_map(isochrons.rpg, ([u[2] for u in s.u], neuron_output.(s.u)); title="Walker, ω = 0",
        xlabel="leg angle φ (rad)", ylabel="neuron output o", colorbar=true,
        ylims=(0, 1.025), yticks=0:0.2:1)   # headroom: the cycle comes within 0.001 of o = 1
end

# ╔═╡ 61033d71-7a1f-5f2a-b9e1-ca2c7a0a2fdb
md"""
## 2. Period and amplitude

What sets each model's period and amplitude, and whether the two can be set separately (WPA works this out at length; this section gathers it into one figure).

**The AP Oscillator.** Write the state as ``(x, v/\sqrt{k}) = r\,(\cos\phi_{\mathrm{p}}, \sin\phi_{\mathrm{p}})``. Then
```math
\dot r = \sin^2\!\phi_{\mathrm{p}}\,\Bigl(\frac{c}{\sqrt{k}} - b\,r\Bigr), \qquad
\dot\phi_{\mathrm{p}} = -\sqrt{k} + \tfrac{1}{2}\sin 2\phi_{\mathrm{p}}\,\Bigl(\frac{c}{\sqrt{k}\,r} - b\Bigr).
```
Both brackets vanish on the circle ``r = R = c/(b\sqrt{k})``. The cycle is that circle, run at the constant rate ``\sqrt{k}``, so
```math
T = \frac{2\pi}{\sqrt{k}}, \qquad A = R = \frac{c}{b\sqrt{k}} .
```
The drive over the damping, ``c/b``, is a pure amplitude parameter: ``A`` is proportional to it, and ``T`` does not move. The stiffness ``k`` is the period parameter, and it moves the amplitude too, ``A \propto 1/\sqrt{k}``.

**The Walker.** The amplitude is set by the neuron's fold. The leg turns near the fold's two edges, whose leg angles lie ``2A_0`` apart, with
```math
A_0(w) = \frac{\sqrt{w(w-4)} - 4\ln\!\bigl(\tfrac{1}{2}(\sqrt{w} + \sqrt{w-4})\bigr)}{2S_W}
```
(Beer 1995; WPA §2), so the self-weight ``w`` is the amplitude parameter (here with ``\beta = -w/2``, which keeps the fold centred on ``\phi = 0``). The leg overshoots both edges, by its momentum at the raise and by the neuron's lag, so ``A`` lies above ``A_0``, the more so the wider the swing. Below ``w \approx 3.65`` the Walker comes to rest; between that and ``w = 4`` it keeps a small cycle without a fold, carried by the leg's momentum and the neuron's lag. The period grows with ``w`` too, since a wider swing takes longer.

The period is set by the tempo. Let ``(y(t), \phi(t), \omega(t))`` be a run of the Walker with time constant ``\tau`` and gains ``\alpha_{\max}``, ``a_{\max}``, and set ``\tilde y(t) = y(k_{\mathrm{t}} t)``, ``\tilde\phi(t) = \phi(k_{\mathrm{t}} t)``, ``\tilde\omega(t) = k_{\mathrm{t}}\,\omega(k_{\mathrm{t}} t)``. Each time derivative picks up a factor ``k_{\mathrm{t}}``, and the tilde run is a run of the Walker with time constant ``\tau/k_{\mathrm{t}}`` and gains ``k_{\mathrm{t}}^2\alpha_{\max}``, ``k_{\mathrm{t}}^2 a_{\max}`` (WPA §4 writes it out). The guards depend only on ``y``, ``\phi`` and the foot's place, and the resets set ``\omega`` to 0 or keep it, so the transitions fall at the same points of the run. The new walker has the same cycle in ``(\phi, o)``, and so the same amplitude, with period ``T/k_{\mathrm{t}}``, exactly.

The tempo moves two lower parameters together: the neuron's time constant ``\tau`` and a factor ``g`` on both body gains. On their own, each changes the Walker, not just its speed. The scaling above gives the whole map from one of its lines:
```math
T(\tau, g) = \frac{T\bigl(\tau\sqrt{g},\, 1\bigr)}{\sqrt{g}},
```
since the Walker ``(\tau, g)`` is the Walker ``(\tau\sqrt{g}, 1)`` run at tempo ``\sqrt{g}``. Along a line of constant ``\tau\sqrt{g}`` (the neuron's time constant measured in the body's time) the Walker is the same, run faster or slower: these are the tempo's lines. Across them, ``\tau\sqrt{g}`` sets how far the neuron lags behind the leg, which adds overshoot and so lengthens the swing, and once the leg reaches its stops, waiting time.

So the two models mirror each other. The AP Oscillator has a pure amplitude parameter (``c/b``) and a period parameter that also moves the amplitude (``k``). The Walker has a pure period parameter (its tempo ``k_{\mathrm{t}}``) and an amplitude parameter that also moves the period (``w``).
"""

# ╔═╡ 37549ce4-1ff3-518e-84ff-22fd5cbadabf
"""
    bingham_cycle(k, b, c; x0 = 0.5, n = 1200)

The AP Oscillator's cycle with stiffness `k`, damping `b` and drive `c`, from a run started at (x0, 0) and long
enough to settle (40/b plus ten periods): the period `T` between its last two maxima of x, the
amplitude `A` (half the difference between the last maximum and the minimum before it, both located
where v crosses 0), the predictions `T0` = 2π/√k and `R` = c/(b√k), and the last period saved at
`n` times from a maximum (`t`, `x`, `v`). From WPA.
"""
function bingham_cycle(k, b, c; x0=0.5, n=1200)
    tmax = 40 / b + 10 * 2π / sqrt(k)
    maxima, minima = Tuple{Float64,Float64}[], Tuple{Float64,Float64}[]
    turns = CallbackSet(section_callback(BINGHAM_SECTION, i -> push!(maxima, (i.t, i.u[1]))),
        section_callback(HalfSpace([0.0, -1.0], 0.0), i -> push!(minima, (i.t, i.u[1]))))    # v ≤ 0 entered: a maximum; v ≥ 0: a minimum
    sol = trajectory(IVP(bingham((; k, b, c)), [x0, 0.0]), (0.0, tmax); callback=turns)
    (t1, _), (t2, xmax) = maxima[end-1], maxima[end]
    xmin = last(minima[findlast(m -> m[1] < t2, minima)])
    ts = range(t1, t2; length=n)
    xs, vs = [sol(t)[1] for t in ts], [sol(t)[2] for t in ts]
    (; k, b, c, T=t2 - t1, A=(xmax - xmin) / 2, T0=2π / sqrt(k), R=c / (b * sqrt(k)), t=ts .- t1, x=xs, v=vs)
end

# ╔═╡ b07aeb1f-6c56-5a75-94c0-c337b1a08398
begin
    """Edges I_L(w), I_R(w), in total input, of the fold in a self-connected neuron's steady state (Beer 1995); NaN for w ≤ 4."""
    I_L(w) = w <= 4 ? NaN : 2log((sqrt(w) + sqrt(w - 4)) / 2) - (w + sqrt(w * (w - 4))) / 2
    I_R(w) = w <= 4 ? NaN : -2log((sqrt(w) + sqrt(w - 4)) / 2) - (w - sqrt(w * (w - 4))) / 2
    """The fold's half-width A₀ = (φ_IR − φ_IL)/2 in leg angle: it depends on w and S_W only."""
    fold_half_width(w; leg=LEG) = (I_R(w) - I_L(w)) / 2leg.sw
    """The fold's centre −(w/2 + β)/S_W in leg angle, which is also the equilibrium's leg angle (o = ½, ω = 0)."""
    fold_centre(w, β; leg=LEG) = -(w / 2 + β) / leg.sw
end

# ╔═╡ e66d216b-8832-5ea2-90bf-d6b5bf9bc054
"""
    walk(hs, (q, x), tmax; stop = events -> false, kwargs...)

Run the Walker `hs` from state `x` in mode `q` for at most `tmax`, recording every transition as
`(time, label, state just after)`, until `stop(events)`. Returns `(events, run)`.
"""
function walk(hs, (q, x), tmax; stop=ev -> false, kwargs...)
    events = Tuple{Float64,Int,SVector{6,Float64}}[]
    run = trajectory(IVP(hs, (q, x)), (0.0, tmax); interp_points=3, on_transition=function (i, t)
        push!(events, (i.t, symbol(hs, t), i.u))
        stop(events) && terminate!(i)
    end, kwargs...)
    events, run
end

# ╔═╡ bda21176-06a2-5550-aa43-78890fa2aaee
"""
    settle(w, β, τ; k = 1, g = 1)

The cycle of the Walker `walker(w, β, τ; k, g)`, from Beer's reset (neuron state 0, the leg forward
at φmax and still): run until two successive steps agree (angle to 1e-9, period to 1e-6 relative).
Returns `(; status, φstep, period)` with status `:cycle`, `:rest` (the steps converge onto the
equilibrium), `:stuck` (the Walker stops stepping) or `:running` (not settled after 300 steps);
`kwargs` go to the solver. From WPA.
"""
function settle(w, β, τ; k=1.0, g=1.0, maxsteps=300, kwargs...)
    hs = walker(w, β, τ; k, g)
    φ̄ = fold_centre(w, β)
    steps = Int[]
    status = Ref(:running)
    function stop(ev)
        ev[end][2] == STEP || return false
        push!(steps, length(ev))
        n = length(steps)
        abs(ev[end][3][2] - φ̄) < 1e-7 && (status[] = :rest; return true)
        if n >= 3
            a, b, c = ev[steps[n-2]], ev[steps[n-1]], ev[steps[n]]
            if abs(c[3][2] - b[3][2]) < 1e-9 && abs((c[1] - b[1]) - (b[1] - a[1])) < 1e-6 * (c[1] - b[1])
                status[] = :cycle
                return true
            end
        end
        n >= maxsteps
    end
    start = (σ(β) > 0.5 ? STANCE : SWING, SA[0.0, LEG.φmax, 0.0, LEG.L * cos(LEG.φmax), 0.0, LEG.L * sin(LEG.φmax)])
    events, run = walk(hs, start, (3000.0 + 400τ) / (k * sqrt(g)); stop, save_everystep=false, kwargs...)
    n = length(steps)
    none = (; status=:stuck, φstep=NaN, period=NaN)
    n < 2 && return none
    status[] == :rest && return (; none..., status=:rest)
    a, b = events[steps[n-1]], events[steps[n]]
    T = b[1] - a[1]
    status[] != :cycle && run.solution.t[end] - b[1] > 2T && return none
    (; status=status[], φstep=b[3][2], period=T)
end

# ╔═╡ 73b02df6-a852-522d-b792-68c456b44641
"""
    leg_cycle(w, β, τ; k = 1, g = 1, n = 1500)

One period of the cycle of `walker(w, β, τ; k, g)`, from a step: the period `T`, the step angle
`φstep`, the leg's forward and backward turning points `φhi` and `φlo`, the amplitude `A` and centre
`φc`, the stance share of the period, the leg's angle at the raise `φraise` (before any snap-back),
whether a stop is reached, and the run saved at `n` evenly spaced times (`t`, `φ`, `o`, and at each
transition before and after it) with its transitions `events`. `nothing` if the Walker has no
cycle. From WPA.
"""
function leg_cycle(w, β, τ; k=1.0, g=1.0, n=1500, kwargs...)
    s = settle(w, β, τ; k, g, kwargs...)
    s.status == :cycle || return nothing
    hs = walker(w, β, τ; k, g)
    start = (STANCE, SA[-β, s.φstep, 0.0, LEG.L * cos(s.φstep), 0.0, LEG.L * sin(s.φstep)])   # just after a step
    ev, run = walk(hs, start, 3s.period; stop=ev -> ev[end][2] == STEP, saveat=range(0, s.period; length=n), kwargs...)
    T = ev[end][1]
    sol = run.solution
    raise = ev[findfirst(e -> e[2] == RAISE, ev)]
    foot_angle = atan((raise[3][6] - raise[3][5]) / raise[3][4])   # the leg's angle before a snap-back at the raise
    φ = [u[2] for u in sol.u]
    φhi = max(maximum(φ), s.φstep)
    φlo = min(minimum(φ), foot_angle)
    (; w, β, τ, k, g, T, φstep=s.φstep, φhi, φlo, A=(φhi - φlo) / 2, φc=(φhi + φlo) / 2, stance=raise[1] / T,
        φraise=foot_angle, stops=any(e -> e[2] in (FRONT_STOP, BACK_STOP, FOOT_STOP), ev),
        t=sol.t, φ, o=[σ(u[1] + β) for u in sol.u], events=ev)
end

# ╔═╡ d13d93e1-7efe-5e79-85b4-907d08387557
"""Each model's default cycle, with its period and amplitude: the AP Oscillator's (k = 2, b = 0.3, c = 1) and the Walker's."""
defaults = (bingham=bingham_cycle(BINGHAM...), walker=leg_cycle(FOLD...))

# ╔═╡ e62dac68-adf7-555a-950a-1d6877168df0
"""
Amplitude against each model's amplitude parameter: the AP Oscillator's for drives c = q × 1 (b = 0.3, k = 2,
so c/b = q × 10/3), and the Walker's over its self-weight w with β = −w/2, τ = 0.1 (period and stops
too; NaN where it has no cycle).
"""
amplitude_scan = let qs = 0.25:0.25:2.0, ws = 0.5:0.25:20.0
    bingham = [bingham_cycle(BINGHAM.k, BINGHAM.b, q * BINGHAM.c) for q in qs]
    out = Vector{Any}(undef, length(ws))
    Threads.@threads :greedy for i in eachindex(ws)
        c = leg_cycle(ws[i], -ws[i] / 2, FOLD.τ; n=400)
        out[i] = c === nothing ? (A=NaN, T=NaN, stops=false) : (; c.A, c.T, c.stops)
    end
    (; qs, A_bingham=[m.A for m in bingham], ws, A=getindex.(out, :A), T=getindex.(out, :T), stops=getindex.(out, :stops))
end

# ╔═╡ 754e48c6-94fe-512a-9862-17f616636c30
"""
The Walker's period over its neuron's time constant τ and a factor g on both body gains
(81 × 81 points, log-spaced), with its amplitude and whether the cycle reaches a stop; NaN where it
has no cycle. Rows are g, columns τ.
"""
period_map = let τs = exp10.(range(-2, 0; length=81)), gs = exp10.(range(-1, 1; length=81))
    grid = [(τ, g) for g in gs, τ in τs]
    out = Matrix{Any}(undef, size(grid))
    Threads.@threads :greedy for i in eachindex(grid)
        τ, g = grid[i]
        c = leg_cycle(FOLD.w, FOLD.β, τ; g, n=200)
        out[i] = c === nothing ? (T=NaN, A=NaN, stops=false) : (; c.T, c.A, c.stops)
    end
    (; τs, gs, T=getindex.(out, :T), A=getindex.(out, :A), stops=getindex.(out, :stops))
end

# ╔═╡ d7c2a79d-829d-5d3f-949d-9f4e65da9f27
"""Period against each model's period parameter, at factors 1/4 to 4 of its default: the AP Oscillator's stiffness k (b, c as they are) and the Walker's tempo kₜ."""
tempo_scan = let factors = [0.25, 0.5, 1.0, 2.0, 4.0]
    (; factors, bingham=[bingham_cycle(f * BINGHAM.k, BINGHAM.b, BINGHAM.c).T for f in factors],
        walker=[leg_cycle(FOLD...; k=f, n=200).T for f in factors])
end

# ╔═╡ 26358daa-f01b-5690-8ca8-9a18d31c1256
let d = defaults, s = tempo_scan, m = period_map
    r(x) = round(x; sigdigits=6)
    # the map from its line g = 1: T(τ, g) = T(τ√g, 1)/√g, at three points off that line
    from_line = map(((0.05, 4.0), (0.2, 0.25), (0.02, 9.0))) do (τ, g)
        (; τ, g, T=r(leg_cycle(FOLD.w, FOLD.β, τ; g, n=200).T), from_line=r(leg_cycle(FOLD.w, FOLD.β, τ * sqrt(g); n=200).T / sqrt(g)))
    end
    (; bingham=(T=r(d.bingham.T), T_predicted=r(d.bingham.T0), A=r(d.bingham.A), A_predicted=r(d.bingham.R)),
        walker=(T=r(d.walker.T), A=r(d.walker.A), A₀=r(fold_half_width(FOLD.w)), stops=d.walker.stops),
        walker_T_times_kt=r.(s.walker .* s.factors), bingham_T_times_sqrt_k=r.(s.bingham .* sqrt.(s.factors .* BINGHAM.k)),
        from_line, no_cycle_below_w=amplitude_scan.ws[findlast(isnan, amplitude_scan.A)],
        map_period_range=r.(extrema(filter(!isnan, m.T))), map_unresolved=count(isnan, m.T), map_with_stops=count(m.stops))
end

# ╔═╡ b02a317c-d23c-5390-a8d0-0b390f63e9ad
"""
The Walker's period and whether it reaches a stop along the line g = 1 of the period map, at
121 time constants u from 10^−2.5 to 10^0.5. By T(τ, g) = T(τ√g, 1)/√g this line fixes the whole
map: its contour T = T* is the curve (τ, g) = (u T*/T(u), (T(u)/T*)²).
"""
map_line = let us = exp10.(range(-2.5, 0.5; length=121))
    out = Vector{Any}(undef, length(us))
    Threads.@threads :greedy for i in eachindex(us)
        c = leg_cycle(FOLD.w, FOLD.β, us[i]; n=200)
        out[i] = c === nothing ? (T=NaN, stops=false) : (; c.T, c.stops)
    end
    stops = getindex.(out, :stops)
    (; us, T=getindex.(out, :T), stops, u★=us[findfirst(stops)])   # u★: the first time constant on g = 1 whose cycle reaches a stop
end

# ╔═╡ c1eaabd2-863d-5f08-b176-96ce8a17d25a
begin
    """
        split_by(y, blocked)

    The series `y` in two parts, `(clear, blocked)`, each with NaN elsewhere: its values where `blocked` is
    false, and where it is true together with their neighbours, so that a line drawn for each part joins
    the other (used to draw a cycle's measures dotted where the cycle reaches a stop). From WPA.
    """
    function split_by(y, blocked)
        n = length(y)
        near = [blocked[i] || (i > 1 && blocked[i-1]) || (i < n && blocked[i+1]) for i in 1:n]
        (ifelse.(blocked, NaN, y), ifelse.(near, y, NaN))
    end

    """
        period_bracket!(p, t0, T, y; tick, label)

    A horizontal bracket over one period, from `t0` to `t0 + T` at height `y`, with end ticks of height
    `tick` and `label` above it.
    """
    function period_bracket!(p, t0, T, y; tick, label)
        plot!(p, [t0, t0 + T, NaN, t0, t0, NaN, t0 + T, t0 + T], [y, y, NaN, y - tick, y + tick, NaN, y - tick, y + tick];
            color=INK, lw=0.8)
        annotate!(p, t0 + T / 2, y + 1.6tick, text(label, 8, INK_SECONDARY, :bottom))
    end

    """
        amplitude_bracket!(p, t, lo, hi; tick, label)

    A vertical bracket at time `t` from the turning point `lo` to `hi` (twice the amplitude), with end
    ticks of width `tick` and `label` to its right.
    """
    function amplitude_bracket!(p, t, lo, hi; tick, label)
        plot!(p, [t, t, NaN, t - tick, t + tick, NaN, t - tick, t + tick], [lo, hi, NaN, hi, hi, NaN, lo, lo]; color=INK, lw=0.8)
        annotate!(p, t + 1.5tick, (lo + hi) / 2, text(label, 8, INK_SECONDARY, :left))
    end

    """
        phase_markers_in_time!(p, T, positions; periods)

    The phase markers of §1 (`MARKER_PHASES`, coloured as `PHASE_MARKERS`) on a series of `periods`
    periods `T` that starts at the section (θ = 0): at times θ T/2π (θ taken in [0, 2π)) in each
    period, and at the end, with `positions` the limb's position at each of `MARKER_PHASES`.
    """
    function phase_markers_in_time!(p, T, positions; periods)
        θ = [repeat(collect(MARKER_PHASES), periods); 0.0]
        t = [j * T + mod(a, 2π) / 2π * T for j in 0:periods-1 for a in MARKER_PHASES]
        scatter!(p, [t; periods * T], [repeat(positions, periods); positions[MARKER_PHASES.==0]];
            PHASE_MARKERS..., marker_z=θ)
    end

    """Ticks of a log axis at the values `v`, labelled as plain numbers."""
    log_ticks(v) = (v, [x < 1 ? string(x) : string(round(Int, x)) for x in v])

    """Colour scale of the Walker's period: its orange, light (short) to dark (long); one hue, lightness monotone (dataviz ordinal check)."""
    PERIOD_SHADE = cgrad(["#fbe3d3", "#f5b48f", MODEL_COLORS.rpg, "#a8421b", "#4a1a08"])
end

# ╔═╡ fcd02053-3e73-51ab-9ff3-1f5cf75a6214
md"""
**2A. Period and amplitude over time.** Each model's limb over two periods of its default cycle: the AP Oscillator's position ``x`` (left; ``k = 2``, ``b = 0.3``, ``c = 1``) and the Walker's leg angle ``\phi`` (right; ``w = 10``, ``\beta = -5``, ``\tau = 0.1``). The bracket on top spans one period ``T``, the one at the right twice the amplitude ``A``, between the turning points (black lines). The markers are the phases ``\theta = -\pi, -3\pi/4, \dots, 3\pi/4`` of figure 1 in each period, on the isochrons' colour scale: the points where the isochrons of those phases cross each cycle, the markers on the cycles in 1A and 1B. Time runs from each model's section, so the darkest marker (``\theta = 0``) is the AP Oscillator's forward turn and the Walker's step. The AP Oscillator's limb moves as a sine. The Walker's leg moves back slowly in stance (about two thirds of its period) and forward fast in swing, so six of its eight markers, the step's included, fall in stance; the open dots are its raises, and its time is in the model's own units.
"""

# ╔═╡ 93b47daa-8251-50fa-8dee-6ce33d207574
"""The AP Oscillator's limb position over two periods of its cycle, with its period and amplitude and the phase markers of §1."""
panel_series_bingham = let m = defaults.bingham, periods = 2, ylims = (-3.4, 4.1)
    t = reduce(vcat, [m.t .+ j * m.T for j in 0:periods-1])
    x = repeat(m.x, periods)
    hi, lo = maximum(m.x), minimum(m.x)
    span = (ylims[2] - ylims[1], (periods + 0.3) * m.T)
    p = plot(; title="AP Oscillator: T = $(round(m.T; digits=2)) s, A = $(round(m.A; digits=2)) rad",
        xlabel="time (s)", ylabel="limb position x (rad)", xlims=(0, span[2]), ylims, legend=false)
    hline!(p, [hi, lo]; color=INK, lw=0.7)
    plot!(p, t, x; color=MODEL_COLORS.bingham, lw=2)
    phase_markers_in_time!(p, m.T, first.(cycle_at(on_cycle.bingham, cycles.bingham, MARKER_PHASES)); periods)
    period_bracket!(p, 0.0, m.T, hi + 0.08span[1]; tick=0.025span[1], label="T")
    amplitude_bracket!(p, (periods + 0.08) * m.T, lo, hi; tick=0.012span[2], label="2A")
end

# ╔═╡ 67ed563c-6c01-5012-9534-8a2f721c149f
"""The Walker's leg angle over two periods of its cycle, with its period and amplitude, raises (open) and the phase markers of §1 (θ = 0: the step)."""
panel_series_walker = let c = defaults.walker, periods = 2, ylims = (-0.38, 0.47)
    t = reduce(vcat, [c.t .+ j * c.T for j in 0:periods-1])
    φ = repeat(c.φ, periods)
    span = (ylims[2] - ylims[1], (periods + 0.3) * c.T)
    p = plot(; title="Walker: T = $(round(c.T; digits=1)), A = $(round(c.A; digits=2)) rad",
        xlabel="time", ylabel="leg angle φ (rad)", xlims=(0, span[2]), ylims, legend=false)
    hline!(p, [c.φhi, c.φlo]; color=INK, lw=0.7)
    plot!(p, t, φ; color=MODEL_COLORS.rpg, lw=2)
    scatter!(p, [(j + c.stance) * c.T for j in 0:periods-1], fill(c.φraise, periods); color=:white, msc=INK, ms=4, msw=1)
    phase_markers_in_time!(p, c.T, [u[2] for u in cycle_at(on_cycle.rpg.solution, cycles.rpg, MARKER_PHASES)]; periods)
    period_bracket!(p, 0.0, c.T, c.φhi + 0.08span[1]; tick=0.025span[1], label="T")
    amplitude_bracket!(p, (periods + 0.08) * c.T, c.φlo, c.φhi; tick=0.012span[2], label="2A")
    annotate!(p, c.stance * c.T / 2, c.φlo - 0.045span[1], text("stance", 7, INK_SECONDARY))
    annotate!(p, c.T * (1 + c.stance) / 2, c.φlo - 0.045span[1], text("swing", 7, INK_SECONDARY))
    p
end

# ╔═╡ 0c0200bb-3329-57ed-8db5-02b31640425d
md"""
**2B. Amplitude against the amplitude parameter.** Each model's amplitude against its amplitude parameter, both divided by their default values, so that both curves pass through ``(1, 1)``: the AP Oscillator's drive over damping ``c/b`` (default ``10/3``, amplitude ``2.36`` rad; ``c`` varied, ``b`` and ``k`` held) and the Walker's self-weight ``w`` (default 10, amplitude ``0.28`` rad; ``\beta = -w/2``, ``\tau = 0.1``). The AP Oscillator's amplitude is proportional to its parameter, the line ``A = R = c/(b\sqrt{k})`` (dots: measured cycles). The Walker has a cycle only from ``w \approx 3.65`` on, a little before its neuron's fold appears at ``w = 4``. Its amplitude then grows with ``w``, following the fold's half-width ``A_0(w)`` (§2) plus an overshoot that grows with the swing, until the swing reaches the stops (dotted from ``w \approx 14.5``; grey dashed: a swing from stop to stop, ``A = \pi/6``). Above ``w \approx 17.6`` the stops hold the leg and the Walker stops stepping.
"""

# ╔═╡ df3fec1b-01cc-5c6a-af24-51449a86f5b9
"""Amplitude against the amplitude parameter, each relative to its default: the AP Oscillator's c/b and the Walker's w."""
panel_amplitude = let s = amplitude_scan, d = defaults, lim = 2.05
    xw, yw = collect(s.ws) ./ FOLD.w, s.A ./ d.walker.A
    p = plot(; xlabel="amplitude parameter ÷ default", ylabel="amplitude A ÷ default",
        xlims=(0, lim), ylims=(0, lim), legend=:bottomright)
    hline!(p, [LEG.φmax / d.walker.A]; color=INK_MUTED, ls=:dash, lw=0.8, label="")
    plot!(p, [0, lim], [0, lim]; color=MODEL_COLORS.bingham, lw=2, label="AP Oscillator: drive ÷ damping, c/b")
    scatter!(p, s.qs, s.A_bingham ./ d.bingham.A; color=MODEL_COLORS.bingham, ms=4, msw=0, label="")
    clear, blocked = split_by(yw, s.stops)
    plot!(p, xw, blocked; color=MODEL_COLORS.rpg, lw=1.2, ls=:dot, label="")
    plot!(p, xw, clear; color=MODEL_COLORS.rpg, lw=2, label="Walker: self-weight w")
    scatter!(p, [1.0], [1.0]; color=:white, msc=INK, ms=5, msw=1.2, label="")
    annotate!(p, 0.05, LEG.φmax / d.walker.A + 0.05, text("Walker's stops, π/6", 7, INK_MUTED, :left))
    annotate!(p, 1.04, 0.94, text("default", 7, INK_SECONDARY, :left))
    p
end

# ╔═╡ 60160519-4337-5c94-b6af-0a31a0b6d2a2
md"""
**2C. The Walker's period over its tempo parameters.** The Walker's period over the neuron's time constant ``\tau`` and a factor ``g`` on both body gains (log axes; colour from short, light, to long, dark). The black lines are contours of ``T``, drawn from the one line ``g = 1`` through ``T(\tau, g) = T(\tau\sqrt{g}, 1)/\sqrt{g}``; they lie on the directly computed map. The dashed line is the tempo's line through the default walker (``\tau\sqrt{g} = 0.1``), with dots at ``k_{\mathrm{t}}`` = 0.5, 1 and 2 (``\tau = 0.1/k_{\mathrm{t}}``, ``g = k_{\mathrm{t}}^2``): along it the Walker is the same, faster or slower. Mostly the body sets the period, ``T \propto 1/\sqrt{g}`` at fixed ``\tau``; a slower neuron lengthens it, by adding overshoot, and in the corner above the grey line (``\tau\sqrt{g}`` above $(round(map_line.u★; sigdigits=2))) the swing reaches the stops.
"""

# ╔═╡ ccab8dc2-5bc4-52a6-848a-c592b12a3753
"""The Walker's period over τ and g, with contours from the line g = 1, the tempo line through the default, and the stops' boundary."""
panel_period_map = let m = period_map, line = map_line, levels = (10, 15, 20, 30, 45, 70), kts = (0.5, 1.0, 2.0)
    xl, yl = (-2, 0), (-1, 1)
    p = heatmap(log10.(m.τs), log10.(m.gs), log10.(m.T); color=PERIOD_SHADE, colorbar=false, xlims=xl, ylims=yl,
        xticks=(log10.([0.01, 0.03, 0.1, 0.3, 1]), ["0.01", "0.03", "0.1", "0.3", "1"]),
        yticks=(log10.([0.1, 0.3, 1, 3, 10]), ["0.1", "0.3", "1", "3", "10"]),
        xlabel="neuron's time constant τ", ylabel="factor g on both body gains", title="Walker's period T",
        legend=false, right_margin=12Plots.mm)
    for T in levels
        τ, g = line.us .* T ./ line.T, (line.T ./ T) .^ 2
        plot!(p, log10.(τ), log10.(g); color=:white, lw=2.2)
        plot!(p, log10.(τ), log10.(g); color=INK, lw=0.9)
        i = argmin(abs.(log10.(τ)))                       # where it meets the right edge, τ = 1
        if yl[1] < log10(g[i]) < yl[2]
            annotate!(p, 0.03, log10(g[i]), text("T = $T", 7, INK_SECONDARY, :left))
        else                                              # it leaves through the top: label it there, inside, to its left
            j = findlast(<(yl[2]), log10.(g))
            annotate!(p, log10(τ[j]) - 0.04, yl[2] - 0.06, text("T = $T", 7, INK_SECONDARY, :right, :top))
        end
    end
    τs = exp10.(range(xl...; length=50))
    plot!(p, log10.(τs), log10.((line.u★ ./ τs) .^ 2); color=INK_MUTED, lw=1.5)   # the stops' boundary, τ√g = u★
    annotate!(p, -0.05, 0.9, text("stops reached", 7, INK, :right))
    plot!(p, log10.(τs), log10.((FOLD.τ ./ τs) .^ 2); color=INK, ls=:dash, lw=1.2)
    for kt in kts
        scatter!(p, [log10(FOLD.τ / kt)], [log10(kt^2)]; color=MODEL_COLORS.rpg, msc=INK, ms=5, msw=1)
        annotate!(p, log10(FOLD.τ / kt) + 0.05, log10(kt^2) + 0.05, text("kₜ = " * string(isinteger(kt) ? Int(kt) : kt), 7, INK, :left, :bottom))
    end
    p
end

# ╔═╡ e559ce9d-f0ab-5294-a5d9-fe209fe7e092
md"""
**2D. Period against the period parameter.** Each model's period against its period parameter, both divided by their default values, on log axes: the AP Oscillator's stiffness ``k`` (default 2 s⁻², period ``4.44`` s; ``b`` and ``c`` held) and the Walker's tempo ``k_{\mathrm{t}}`` (default 1, period ``26.2``). Lines: ``T = 2\pi/\sqrt{k}``, slope ``-\tfrac{1}{2}``, and ``T = T(1)/k_{\mathrm{t}}``, slope ``-1``; dots: measured cycles. The tempo is the dashed line of 2C seen along its length. The AP Oscillator's ``k`` moves the amplitude too, ``A \propto 1/\sqrt{k}``; the Walker's tempo does not.
"""

# ╔═╡ 8c9e8edd-d561-5405-97a7-af15a911ebcc
"""Period against the period parameter, each relative to its default, on log axes: the AP Oscillator's stiffness k and the Walker's tempo kₜ."""
panel_period = let s = tempo_scan, d = defaults, f = exp10.(range(log10(0.22), log10(4.5); length=60))
    ticks = log_ticks([0.25, 0.5, 1, 2, 4])
    p = plot(; xscale=:log10, yscale=:log10, xticks=ticks, yticks=ticks, xlims=(0.22, 4.5), ylims=(0.22, 4.5),
        xlabel="period parameter ÷ default", ylabel="period T ÷ default", legend=:topright)
    plot!(p, f, f .^ -0.5; color=MODEL_COLORS.bingham, lw=2, label="AP Oscillator: stiffness k, T = 2π/√k")
    scatter!(p, s.factors, s.bingham ./ d.bingham.T; color=MODEL_COLORS.bingham, ms=4, msw=0, label="")
    plot!(p, f, 1 ./ f; color=MODEL_COLORS.rpg, lw=2, label="Walker: tempo kₜ, T = T(1)/kₜ")
    plot!(p, [NaN], [NaN]; la=0, label="τ → τ/kₜ, both gains → kₜ² × gains")   # a legend row with no line: what the tempo does

    scatter!(p, s.factors, s.walker ./ d.walker.T; color=MODEL_COLORS.rpg, ms=4, msw=0, label="")
    p
end

# ╔═╡ 7c1d0b55-591f-50ed-b930-2f22ac3826a4
md"""
## 3. Phase information in the leg and the neuron

How much of the Walker's phase its leg angle and its neuron each pin down, and how phase information passes between the two. Figure 3, at the bottom of the notebook, puts the panels together: the posteriors on top and the transfer entropy below.

**Which phases a reading leaves open.** Take the Walker's phase ``\theta`` to be uniform over the cycle (a moment of the cycle drawn at random) and read a variable ``c`` of the Walker with Gaussian noise, ``r = c(\theta) + \varepsilon``, of standard deviation ``\sigma`` equal to 5 % of the variable's range over the cycle in each coordinate (`READING_NOISE`). Without noise a reading would pin down a point of the cycle; the noise sets the resolution. The posterior over the phase is then
```math
p(\hat\theta \mid r) \propto \exp\Bigl(-\sum_i \frac{\bigl(r_i - c_i(\hat\theta)\bigr)^2}{2\sigma_i^2}\Bigr),
```
taken over 512 equal phases ``\hat\theta``. Each panel shows, for the Walker at phase ``\theta`` (horizontal) and the reading it gives there without its noise, ``r = c(\theta)``, the posterior over the phase inferred from that reading (vertical), each column scaled to its peak: the shading shows which phases a reading leaves open, not how sharply it picks them out. Knowing the phase exactly would be the diagonal. The phase is counted from the step, ``\theta \in [0, 2\pi)``, so that stance (up to the raise, dashed) and swing are one stretch each. The readings are the leg angle ``\phi``, the neuron state ``y`` (not its output ``o``, which sits near 1 through stance and near 0 through swing and hides the state's movement there) and both together.
"""

# ╔═╡ 9638f418-856d-5232-8d31-6a62f14e0e49
"""Standard deviation of the noise on a reading, as a fraction of the reading's range over the cycle."""
READING_NOISE = 0.05

# ╔═╡ ebd07e1f-7590-59cd-a18f-d2bf9f074ad2
"""Standard deviation of the noise in each coordinate of `readings` (tuples): `level` × the coordinate's range over the cycle."""
function noise_scale(readings; level=READING_NOISE)
    ntuple(length(first(readings))) do i
        lo, hi = extrema(c[i] for c in readings)
        level * (hi - lo)
    end
end

# ╔═╡ b59a6d1d-e76c-5cfa-960e-fc7da0c48ffe
"""
    cycle_states(ts)

The Walker's states on its cycle at the times `ts` (sorted, in [0, T)) after the step, from a run that
saves exactly there, and the run. Reading them off `on_cycle` would interpolate between its saved
states, too coarsely for the neuron's fast jumps.
"""
function cycle_states(ts; tol=(reltol=1e-11, abstol=1e-13))
    run = trajectory(IVP(fold_walker, (STANCE, walker_state(cycles.rpg.s[1], 0.5))), (0.0, cycles.rpg.period); saveat=ts, tol...)
    [run.solution.u[findlast(==(t), run.solution.t)] for t in ts], run
end

# ╔═╡ 154a84dd-2220-5b80-8101-006c71936b8b
"""
Posterior maps of the Walker's phase at 512 equal phases from the step, θ ∈ [0, 2π): for each
phase θ, p(θ̂ | c(θ)) over the phases θ̂ given the noise-free reading c(θ), scaled to its peak (a
column), for readings of the leg angle φ, the neuron state y and both, with noise of `noise_scale`.
Also the phase of the raise.
"""
phase_posterior = let n = 512
    θ = range(0, 2π; length=n + 1)[1:end-1]
    states, _ = cycle_states(θ ./ 2π .* cycles.rpg.period)
    readings = (leg=[(u[2],) for u in states], neuron=[(u[1],) for u in states], both=[(u[2], u[1]) for u in states])
    # p(θ̂ | r) over the phases, scaled to its peak
    function scaled_posterior(c, σ, r)
        ℓ = [-sum(abs2, (r .- x) ./ σ) / 2 for x in c]
        exp.(ℓ .- maximum(ℓ))
    end
    maps = map(c -> (σ = noise_scale(c); reduce(hcat, [scaled_posterior(c, σ, r) for r in c])), readings)
    (; θ, maps..., raise=2π * on_cycle.rpg.jumps[1][1] / cycles.rpg.period)
end

# ╔═╡ a5d004ee-6090-5865-931e-166785781290
md"""
**How phase information passes between the leg and the neuron.** The transfer entropy from a source ``X`` to a target ``Y`` (Schreiber 2000),
```math
T_{X \to Y} = I\bigl(Y_{t+\Delta};\, X_t \bigm| Y_t\bigr),
```
is what ``X``'s present tells about ``Y``'s state a lag ``\Delta`` later that ``Y``'s own present does not. On the deterministic cycle every variable is a fixed function of the phase, so the transfer entropy would only re-describe how the readings above divide the phase between them. It needs dynamics that are noisy themselves, so that a fluctuation of the leg can be followed into the neuron and back.

The Walker here gets white noise on its neuron state ``y`` and on its leg's angular velocity ``\omega`` (`PROCESS_NOISE`, equal multiples of each variable's range over the cycle), in the small-noise limit. Its deviation from the cycle then follows the cycle's linearization (`cycle_linearization`: propagators over one time step by central differences of runs, resets included), the deviations are Gaussian, and the transfer entropies have closed forms (`gaussian_cmi`). They do not depend on the noise's overall size, only on the ratio of the two noises (ratios from ¼ to 4 change the magnitudes below, not the pattern, as for V2's walker in `OscillatoryBehavior.jl` §8). Walkers are seen a time ``s`` after their last step, at phase ``\theta = 2\pi s/T``; the noise also spreads them along the cycle, and the step angle (the leg angle at the step) has the spread of the noisy return map. With the leg's state ``\ell = (\phi, \omega, d)``,
```math
T_{\text{neuron} \to \text{leg}}(\theta) = I\bigl((\phi, \omega)_{t+\Delta};\, y_t \bigm| \ell_t\bigr), \qquad
T_{\text{leg} \to \text{neuron}}(\theta) = I\bigl(y_{t+\Delta};\, \ell_t \bigm| y_t\bigr),
```
here at lag ``\Delta \approx 1``, as in V2: about a 26th of the period, and ten of the fast neuron's time constants.
"""

# ╔═╡ 812ce438-74af-5b9d-b90b-95e3f1250149
"""
Process noise of the Walker: white noise on the neuron's state y and on the leg's angular
velocity ω, each as a multiple of its variable's range over the cycle per unit time. In the
small-noise limit used here only their ratio matters.
"""
PROCESS_NOISE = (neuron=1.0, leg=1.0)

# ╔═╡ 868b17cb-3c76-5349-adb9-e8c4aa91ef8d
"""
The Walker's cycle linearized at n = 512 times sⱼ = (j − ½)T/n after the step, half a time step off
every transition: the propagators Φⱼ over one time step of the deviation (y, φ, ω, d), by central
differences of runs (resets included), and the flow's tangent vⱼ. For the step section:
`g0`, a change in the step angle carried half a time step on, and `a`, the gradient of the next
step angle at the last grid time.
"""
cycle_linearization = let n = 512, h = 1e-6, tol = (reltol=1e-11, abstol=1e-13)
    T = cycles.rpg.period
    Δt = T / n
    s = ((1:n) .- 0.5) .* Δt
    X, ref = cycle_states(s)
    q = [mode_at(ref, t) for t in s]
    run_for(q, x, Δ) = trajectory(IVP(fold_walker, (q, x)), (0.0, Δ); save_everystep=false, tol...).solution.u[end][1:4]
    nudge(x, i, δ) = setindex(x, x[i] + δ, i)
    Φ = Vector{Matrix{Float64}}(undef, n)
    Threads.@threads for j in 1:n
        Φ[j] = reduce(hcat, [(run_for(q[j], nudge(X[j], i, h), Δt) .- run_for(q[j], nudge(X[j], i, -h), Δt)) ./ 2h for i in 1:4])
    end
    v = [vector_field(mode(fold_walker, q[j]), X[j]; check_constraints=false)[1:4] for j in 1:n]
    # the step section
    φ0 = cycles.rpg.s[1]
    g0 = (run_for(STANCE, walker_state(φ0 + h, 0.5), Δt / 2) .- run_for(STANCE, walker_state(φ0 - h, 0.5), Δt / 2)) ./ 2h
    function step_angle(x)
        φ = Ref(NaN)
        trajectory(IVP(fold_walker, (q[n], x)), (0.0, T); save_everystep=false, tol...,
            on_transition=(i, t) -> symbol(fold_walker, t) == STEP && (φ[] = i.u[2]; terminate!(i)))
        φ[]
    end
    a = [(step_angle(nudge(X[n], i, h)) - step_angle(nudge(X[n], i, -h))) / 2h for i in 1:4]
    ranges = (y=-(-)(extrema(x[1] for x in X)...), ω=-(-)(extrema(x[3] for x in X)...))
    (; n, Δt, θ=2π .* s ./ T, raise=2π * ref.jumps[1][1] / T, Φ, v, g0, a, ranges)
end

# ╔═╡ c233065a-b044-5ccf-9f77-78aa672147dd
let lin = cycle_linearization
    g = foldl((x, Φ) -> Φ * x, lin.Φ[1:end-1]; init=lin.g0)
    (return_multiplier=round(lin.a ⋅ g; digits=5), shooting_multiplier=round(cycles.rpg.multipliers[1]; digits=5),
        tangent_carried=maximum(j -> norm(lin.Φ[j] * lin.v[j] - lin.v[mod1(j + 1, lin.n)]) / norm(lin.v[mod1(j + 1, lin.n)]), 1:lin.n-1))
end

# ╔═╡ e4bde1f4-aa5c-5bac-bce1-25e356887177
"""
    walker_fluctuations(lin, noise)

The Walker's deviations (y, φ, ω, d) from its cycle under the white noise `noise` (as
`PROCESS_NOISE`) on y and ω, in the small-noise limit, for walkers seen a time sⱼ after their
last step: their covariance `C` at each grid time of `lin`, the noise covariance `Q` each time step
adds, and the return map's multiplier `μ`. The step angle has the spread of the noisy return
map's stationary state.
"""
function walker_fluctuations(lin, noise)
    D = Diagonal([(noise.neuron * lin.ranges.y)^2, 0.0, (noise.leg * lin.ranges.ω)^2, 0.0])
    Q = [(Φ * D * Φ' + D) * lin.Δt / 2 for Φ in lin.Φ]
    # the noise gathered since the step, and a change in the step angle, carried along the cycle
    C, g = [Matrix(D) * lin.Δt / 2], [lin.g0]
    for j in 1:lin.n-1
        push!(C, lin.Φ[j] * C[j] * lin.Φ[j]' + Q[j])
        push!(g, lin.Φ[j] * g[j])
    end
    μ = lin.a ⋅ g[end]
    spread = lin.a' * (C[end] + D * lin.Δt / 2) * lin.a / (1 - μ^2)   # stationary variance of the step angle
    (; C=[Cj + spread * gj * gj' for (Cj, gj) in zip(C, g)], Q, μ)
end

# ╔═╡ 957508eb-3c75-5007-bb5c-c619da65c58c
"""
    gaussian_cmi(S, A, B, C)

Conditional mutual information I(A; B | C) in bits of jointly Gaussian variables with covariance
`S`, the index vectors `A`, `B`, `C` picking them out, from the covariances of A given C and
given B and C (pseudo-inverses, so that exactly dependent conditions are allowed).
"""
function gaussian_cmi(S, A, B, C)
    residual(X) = isempty(X) ? S[A, A] : S[A, A] - S[A, X] * pinv(S[X, X]; rtol=1e-10) * S[X, A]
    (logdet(Symmetric(residual(C))) - logdet(Symmetric(residual([B; C])))) / 2log(2)
end

# ╔═╡ b1ec5ec6-febf-5240-a230-629e11d7e0ab
"""
    transfer_entropies(lin, fl, lag)

Transfer entropy in bits at each grid phase of `lin`, at a lag of `lag` time steps, for the
fluctuations `fl`: from the neuron to the leg, I(φ, ω at t + Δ; y at t | φ, ω, d at t), and from the
leg to the neuron, I(y at t + Δ; φ, ω, d at t | y at t).
"""
function transfer_entropies(lin, fl, lag)
    n = lin.n
    to_leg, to_neuron = zeros(n), zeros(n)
    Threads.@threads for j in 1:n
        P, R = Matrix(1.0I, 4, 4), zeros(4, 4)
        for i in mod1.(j .+ (0:lag-1), n)   # carry the deviation and the noise on by the lag
            P = lin.Φ[i] * P
            R = lin.Φ[i] * R * lin.Φ[i]' + fl.Q[i]
        end
        # y, φ, ω, d at t (1–4) and at t + Δ (5–8)
        W = [I zeros(4, 4); P I]
        S = W * [fl.C[j] zeros(4, 4); zeros(4, 4) R] * W'
        to_leg[j] = gaussian_cmi(S, [6, 7], [1], [2, 3, 4])
        to_neuron[j] = gaussian_cmi(S, [5], [2, 3, 4], [1])
    end
    (; neuron_to_leg=to_leg, leg_to_neuron=to_neuron)
end

# ╔═╡ 196cd759-e4ac-52d4-b9b6-a177b5febf99
"""Transfer entropies of the Walker under `PROCESS_NOISE` at the lag Δ ≈ 1 (a whole number of time steps)."""
transfer = let lin = cycle_linearization
    lag = round(Int, 1 / lin.Δt)
    (; transfer_entropies(lin, walker_fluctuations(lin, PROCESS_NOISE), lag)..., Δ=lag * lin.Δt)
end

# ╔═╡ f83d5699-3902-583c-a549-021d9865bfa2
let lin = cycle_linearization, te = transfer, s = on_cycle.rpg.solution
    stance = lin.θ .< lin.raise
    average(v, sel=trues(lin.n)) = round(sum(v[sel]) / count(sel); digits=3)
    peak(v, sel=trues(lin.n)) = (bits=round(maximum(v[sel]); sigdigits=2), θ=round(lin.θ[sel][argmax(v[sel])]; digits=2))
    mid_stance = 0.5 .< lin.θ .< lin.raise - 0.5
    near_raise, before_step = abs.(lin.θ .- lin.raise) .< 0.3, lin.θ .> 2π - 0.3
    turn = 2π * s.t[argmin([u[2] for u in s.u])] / cycles.rpg.period   # the leg's backward turn, early in swing
    [(direction="leg → neuron", mean=average(te.leg_to_neuron), stance=average(te.leg_to_neuron, stance),
            swing=average(te.leg_to_neuron, .!stance), peak=peak(te.leg_to_neuron), backward_turn=round(turn; digits=2)),
        (direction="neuron → leg", mean=average(te.neuron_to_leg), stance=average(te.neuron_to_leg, stance),
            swing=average(te.neuron_to_leg, .!stance), mid_stance=peak(te.neuron_to_leg, mid_stance),
            raise=(θ=round(lin.raise; digits=2), peak=peak(te.neuron_to_leg, near_raise)), before_step=peak(te.neuron_to_leg, before_step))]
end

# ╔═╡ 4888772a-0ed5-5d35-86df-85ae210d6577
begin
    """Dark brown of the Walker's swing and, in the transfer entropy, of its neuron as a source."""
    SWING_BROWN = "#9c3b16"

    """Colour scale of the posterior maps: the Walker's hue, white to dark."""
    POSTERIOR_SHADE = cgrad(["#ffffff", "#f5c3a8", MODEL_COLORS.rpg, SWING_BROWN, "#3d1607"])

    """
        posterior_map(p, name; title, ylabel = "", colorbar = false)

    Heat map of `p[name]` (an entry of `phase_posterior`): the phase inferred from the reading against
    the phase θ where the Walker is, both from the step, with the raise dashed.
    """
    function posterior_map(p, name; title, ylabel="", colorbar=false)
        ticks = ([0, π / 2, π, p.raise, 2π], ["0", "π/2", "π", "raise", "2π"])
        h = heatmap(p.θ, p.θ, p[name]; color=POSTERIOR_SHADE, clims=(0, 1), colorbar, colorbar_title="normalized posterior",
            title, xlabel="phase θ of walking cycle (rad)", ylabel, xticks=ticks, yticks=ticks, xlims=(0, 2π),
            ylims=(0, 2π), aspect_ratio=:equal, legend=false)
        hline!(h, [p.raise]; color=INK_MUTED, linestyle=:dash, linewidth=1)
        vline!(h, [p.raise]; color=INK_MUTED, linestyle=:dash, linewidth=1)
    end
end

# ╔═╡ 8666d1d5-050a-572d-b28f-8452d3b3a34e
md"""
**3A. Leg angle.** The posterior over the phase given a reading of the leg angle ``\phi``. Its second band lies in the off-diagonal blocks: the leg passes each angle once in stance and once in swing, so a reading mistakes stance for swing and swing for stance. Early in swing the leg also turns around at the back, passing the same angles going back and coming forward (the cross in the swing block).
"""

# ╔═╡ f11f3bdf-39e7-543f-ae07-0717d39beebb
"""Phases left open by a noisy reading of the Walker's leg angle."""
panel_posterior_leg = posterior_map(phase_posterior, :leg; title="Leg angle φ", ylabel="phase inferred from the reading (rad)")

# ╔═╡ fd344984-024c-5dfe-a0ee-fd981faa3c68
md"""
**3B. Neuron state.** Given a reading of the neuron state ``y``. The fast neuron is on its upper branch through stance and on its lower branch through swing, so its state tells stance from swing: the off-diagonal blocks are empty but for thin lines along their edges, the step and the raise, where ``y = -\beta`` at both. Within a mode ``y`` follows the leg angle along its branch, so the band runs along the diagonal. It is widest early in stance, where the leg has barely begun to move back, and it keeps the leg's cross at the backward turn early in swing.
"""

# ╔═╡ 563d5ab4-0baf-5bbb-9a47-347e0ce34551
"""Phases left open by a noisy reading of the Walker's neuron state."""
panel_posterior_neuron = posterior_map(phase_posterior, :neuron; title="Neuron state y")

# ╔═╡ a7eaae3b-f124-5e0f-b4e8-81d3e4e8849c
md"""
**3C. Joint.** Given both readings, ``(\phi, y)``. The neuron state removes the leg angle's confusion of stance with swing, and the diagonal is left, widest early in stance. So is the cross early in swing: around the leg's backward turn the leg passes the same angles going back and coming forward, and the neuron, on its lower branch, follows the leg angle, so neither reading tells the way back from the way forward.
"""

# ╔═╡ 8cd368f7-295c-517e-be9e-4c30409778d2
"""Phases left open by a noisy reading of both."""
panel_posterior_both = posterior_map(phase_posterior, :both; title="Joint (φ, y)", colorbar=true)

# ╔═╡ db69588a-fb2f-5cbf-8b8c-d577b21addfd
md"""
**3D. Transfer entropy over the cycle.** From the leg to the neuron (orange) and from the neuron to the leg (dark brown), at the lag ``\Delta`` above, with swing shaded. A quarter cycle is shown past each end so that the cycle comes back around, and the lines break at the step, where they jump.

- **Leg → neuron** runs all the way round, through the neuron's sense of the leg angle: about 0.8 bit on average in stance and 1.5 bits in swing. It peaks at 3.3 bits early in swing, at the leg's backward turn (``\theta \approx 4.7``), where the neuron's input a lag later depends most on the leg's velocity, which only the leg's state carries.
- **Neuron → leg** is nil through stance and swing (below ``10^{-6}`` bit in mid-stance): the fast neuron sits on its branch, its state set by the leg angle, and tells nothing about the leg's future that the leg's own state does not. It rises only at the transitions the neuron triggers: to 1.7 bits just before the step and 0.24 bit at the raise. The step stops the leg (``\omega \mapsto 0``), so its timing changes the leg's state at once; the raise keeps ``\omega`` and only turns the push around.

So the phase information runs from the leg to the neuron continuously and comes back from the neuron to the leg only at the mode switches, mostly at the step. This is the posterior maps' finding in dynamic form: the leg's state says where the Walker is within a mode, and the neuron's state says when the mode changes, which the leg cannot foresee. (The jumps at the step come from counting the phase from the last step; `OscillatoryBehavior.jl` §8 discusses them.)
"""

# ╔═╡ 09771a80-8069-509f-bc09-c5f24c484a00
"""The transfer entropies between the Walker's leg and its neuron over the cycle, a quarter cycle of the other end shown past each end; swing shaded."""
panel_transfer_entropy = let lin = cycle_linearization, te = transfer
    margin = π / 4
    tail, head = lin.θ .> 2π - margin, lin.θ .< margin
    xs = [lin.θ[tail] .- 2π; NaN; lin.θ; NaN; lin.θ[head] .+ 2π]
    broken(v) = [v[tail]; NaN; v; NaN; v[head]]   # lines broken at the step, where the transfer entropy jumps
    top_bits = 1.08 * max(maximum(te.leg_to_neuron), maximum(te.neuron_to_leg))
    p = plot(; xlabel="phase θ from the step (rad)", ylabel="transfer entropy (bits)", xlims=(-margin, 2π + margin),
        xticks=([0, π / 2, π, lin.raise, 2π], ["step", "π/2", "π", "raise", "step"]), ylims=(0, top_bits),
        title="Transfer entropy over the cycle, lag Δ = $(round(te.Δ; digits=2))", legend=:outertop, legend_column=-1)
    vspan!(p, [lin.raise - 2π, 0.0]; color=INK_MUTED, alpha=0.12, linewidth=0, label="")
    vspan!(p, [lin.raise, 2π]; color=INK_MUTED, alpha=0.12, linewidth=0, label="")
    vline!(p, [0.0, 2π]; color=INK_MUTED, linestyle=:dot, linewidth=1, label="")
    annotate!(p, lin.raise / 2, 0.97 * top_bits, text("stance", 8, INK_SECONDARY, :center, :top))
    annotate!(p, (lin.raise + 2π) / 2, 0.97 * top_bits, text("swing", 8, INK_SECONDARY, :center, :top))
    plot!(p, xs, broken(te.leg_to_neuron); color=MODEL_COLORS.rpg, linewidth=2.5, label="leg → neuron")
    plot!(p, xs, broken(te.neuron_to_leg); color=SWING_BROWN, linewidth=2.5, label="neuron → leg")
end

# ╔═╡ 52d47a18-928a-52e6-977b-59c3a3a0b32c
md"""
**What the leg angle and the neuron tell about the limit-cycle phase.** The transfer entropy above follows the states of the leg and the neuron. The phase of the limit cycle is one number, the asymptotic phase ``\Theta`` of §1, a function of the whole state. Near the cycle a deviation ``\delta x`` of ``(y, \phi, \omega, d)`` moves it by
```math
\delta\Theta = z(\theta) \cdot \delta x ,
```
with ``z`` the gradient of the asymptotic phase on the cycle (`phase_gradient`). Without noise a state's phase advances at the cycle's rate whatever the state, so the propagators of `cycle_linearization` carry ``z`` back along the cycle, ``z_j = \Phi_j^{\top} z_{j+1}``, from the vector that one turn of the cycle gives back unchanged, scaled so that ``z \cdot \dot x = 2\pi/T``. The cell below it checks ``z`` against runs from nudged states.

For the noisy Walkers of 3D (the same noise, the same lag ``\Delta``, seen a time ``s`` after their last step), take what the leg angle ``\phi`` and the neuron state ``y`` at time ``t`` tell about the phase a lag later, ``I(\Theta_{t+\Delta};\, \phi_t, y_t)``. The phase is taken a lag later because its present value is an exact function of the present state; over the lag the noise moves it. The angular velocity ``\omega`` and the distance ``d`` are not among the sources and are not conditioned on. This information is split in four (the partial information decomposition of Williams and Beer 2010), with the redundancy the smaller of the two single informations (the minimum mutual information of Barrett 2015):

- **redundant**, what either one tells: ``\min\{I(\Theta_{t+\Delta}; \phi_t),\, I(\Theta_{t+\Delta}; y_t)\}``;
- **angle only** and **neuron only**, what one tells beyond the redundant part (with this redundancy one of the two is nil at every phase);
- **integrated**, what only the two together tell: ``I(\Theta_{t+\Delta}; \phi_t, y_t) - \max\{I(\Theta_{t+\Delta}; \phi_t),\, I(\Theta_{t+\Delta}; y_t)\}``.

Like the transfer entropies, these are closed forms of the Gaussian deviations (`gaussian_cmi`), in the small-noise limit: they describe Walkers close to the cycle, and cannot show that the leg angle confuses stance with swing, which 3A–C show.
"""

# ╔═╡ 24500ad0-1c5c-5f1f-bbb1-a02c5274a784
"""
The gradient z of the Walker's asymptotic phase at each grid time of `cycle_linearization`, for the
deviation (y, φ, ω, d): the left eigenvector of one turn of the cycle with multiplier 1, scaled so
that z ⋅ ẋ = 2π/T, carried back along the cycle by the propagators, zⱼ = Φⱼᵀ zⱼ₊₁.
"""
phase_gradient = let lin = cycle_linearization
    turn = foldl((A, Φ) -> Φ * A, lin.Φ; init=Matrix(1.0I, 4, 4))   # one turn of the cycle, from the first grid time
    e = eigen(Matrix(turn'))
    z = real.(e.vectors[:, argmin(abs.(e.values .- 1))])
    z .*= 2π / cycles.rpg.period / (z ⋅ lin.v[1])
    Z = Vector{Vector{Float64}}(undef, lin.n)
    Z[end] = lin.Φ[end]' * z
    for j in lin.n-1:-1:1
        Z[j] = lin.Φ[j]' * Z[j+1]
    end
    Z
end

# ╔═╡ 817d092a-5075-54e7-bfbe-f5169c58b584
# `phase_gradient` checked: z ⋅ ẋ in units of 2π/T at every grid time (1 everywhere), and its largest relative
# difference, at six grid times, from the phase shifts of runs from nudged states (read off their last step in six periods).
let lin = cycle_linearization, Z = phase_gradient, T = cycles.rpg.period, tol = (reltol=1e-11, abstol=1e-13), δ = 1e-5
    s = ((1:lin.n) .- 0.5) .* lin.Δt
    X, ref = cycle_states(s)
    last_step(j, i, δ) = step_times(mode_at(ref, s[j]), setindex(X[j], X[j][i] + δ, i), 6T; tol...)[1][end]
    from_runs(j) = [-2π / T * (last_step(j, i, δ) - last_step(j, i, -δ)) / 2δ for i in 1:4]
    (rate=round.(extrema(Z[j] ⋅ lin.v[j] * T / 2π for j in 1:lin.n); digits=6),
        against_runs=round(maximum(norm(Z[j] - from_runs(j)) / norm(Z[j]) for j in (1, 100, 256, 340, 400, 512)); sigdigits=2))
end

# ╔═╡ d4c811c6-ede7-5eec-9072-442861b329aa
"""
    phase_information_parts(lin, fl, Z, lag)

What the leg angle φ and the neuron state y tell about the asymptotic phase `lag` time steps later, in
bits at each grid phase of `lin`, for the fluctuations `fl` and the phase gradient `Z`: the information
of the whole state (`full`), of the two together (`pair`) and of each (`angle`, `neuron`), and the
parts of `pair` by minimum mutual information: `redundant`, `angle_only`, `neuron_only`, `integrated`.
"""
function phase_information_parts(lin, fl, Z, lag)
    n = lin.n
    info = (full=zeros(n), pair=zeros(n), angle=zeros(n), neuron=zeros(n))
    Threads.@threads for j in 1:n
        R = zeros(4, 4)
        for i in mod1.(j .+ (0:lag-1), n)   # the noise gathered over the lag
            R = lin.Φ[i] * R * lin.Φ[i]' + fl.Q[i]
        end
        z = Z[mod1(j + lag, n)]
        # y, φ, ω, d at t (1–4) and the phase at t + Δ (5): the phase at t, moved by the noise of the lag
        W = [Matrix(1.0I, 4, 4) zeros(4); Z[j]' 1.0]
        S = W * [fl.C[j] zeros(4); zeros(4)' z' * R * z] * W'
        info.full[j] = gaussian_cmi(S, [5], [1, 2, 3, 4], Int[])
        info.pair[j] = gaussian_cmi(S, [5], [1, 2], Int[])
        info.angle[j] = gaussian_cmi(S, [5], [2], Int[])
        info.neuron[j] = gaussian_cmi(S, [5], [1], Int[])
    end
    redundant = min.(info.angle, info.neuron)
    (; info..., redundant, angle_only=info.angle .- redundant, neuron_only=info.neuron .- redundant,
        integrated=info.pair .- max.(info.angle, info.neuron))
end

# ╔═╡ 77d70363-26b4-54ce-bbb9-2eb9f7f1b304
"""The parts of the Walker's phase information under `PROCESS_NOISE`, at the lag of the transfer entropy."""
phase_parts = let lin = cycle_linearization
    phase_information_parts(lin, walker_fluctuations(lin, PROCESS_NOISE), phase_gradient, round(Int, transfer.Δ / lin.Δt))
end

# ╔═╡ 0cc7341a-fc16-59fd-a37e-625606663aa6
# Each quantity's mean in stance and in swing and its peak, and the integrated part's peak at other noise ratios and lags.
let lin = cycle_linearization, p = phase_parts, lag = round(Int, transfer.Δ / lin.Δt)
    stance = lin.θ .< lin.raise
    average(v, sel) = round(sum(v[sel]) / count(sel); digits=2)
    peak(v) = (bits=round(maximum(v); digits=2), θ=round(lin.θ[argmax(v)]; digits=2))
    integrated(noise, lag) = peak(phase_information_parts(lin, walker_fluctuations(lin, noise), phase_gradient, lag).integrated)
    (quantities=[(quantity=k, stance=average(p[k], stance), swing=average(p[k], .!stance), peak=peak(p[k])) for k in keys(p)],
        integrated_by_neuron_noise=[(neuron=r, peak=integrated((neuron=r, leg=1.0), lag)) for r in (0.25, 1.0, 4.0)],
        integrated_by_lag=[(Δ=round(l * lin.Δt; digits=2), peak=integrated(PROCESS_NOISE, l)) for l in (lag ÷ 4, lag, 4lag)])
end

# ╔═╡ bf676518-1139-536e-87e9-81bc793d357c
begin
    """Colours of the parts of an information decomposition: redundant, neuron only, angle only, integrated (synergistic)."""
    PART_COLORS = (redundant=COLORS.blue, neuron=SWING_BROWN, angle=MODEL_COLORS.rpg, integrated=COLORS.aqua)

    """The grid phases of `lin` with a quarter cycle of the other end past each end, broken at the step (as in 3D)."""
    wrapped_phase(lin; margin=π / 4) = [lin.θ[lin.θ.>2π-margin] .- 2π; NaN; lin.θ; NaN; lin.θ[lin.θ.<margin] .+ 2π]

    """`v`, one value per grid phase of `lin`, laid out as `wrapped_phase`."""
    wrapped(lin, v; margin=π / 4) = [v[lin.θ.>2π-margin]; NaN; v; NaN; v[lin.θ.<margin]]

    """
        cycle_panel(lin; margin = π/4, ticks, kwargs...)

    An empty panel over the walking cycle, the phase counted from the step, with swing shaded and the
    steps dotted, as 3D's; `kwargs` go to `plot`.
    """
    function cycle_panel(lin; margin=π / 4, ticks=([0, π / 2, π, lin.raise, 2π], ["step", "π/2", "π", "raise", "step"]), kwargs...)
        p = plot(; xlims=(-margin, 2π + margin), xticks=ticks, kwargs...)
        vspan!(p, [lin.raise - 2π, 0.0]; color=INK_MUTED, alpha=0.12, linewidth=0, label="")
        vspan!(p, [lin.raise, 2π]; color=INK_MUTED, alpha=0.12, linewidth=0, label="")
        vline!(p, [0.0, 2π]; color=INK_MUTED, linestyle=:dot, linewidth=1, label="")
    end
end

# ╔═╡ dc973ca1-6332-576b-90ad-9dafe0981a44
md"""
**4A. What tells the limit-cycle phase.** For the Walker at each phase ``\theta``, the information about the limit-cycle phase a lag later held by the whole state (dotted) and by the leg angle and the neuron state together (dashed), and the four parts of the latter; swing shaded, a quarter cycle past each end, as in 3D.

- **Redundant.** Most of what the pair tells, either one tells: about 1.8 bits on average in stance and 2.6 in swing. It is nil just after the step and again at the leg's backward turn (``\theta \approx 4.7``), where neither the angle (0.04 bit) nor the neuron (nil) tells the phase.
- **The pair against the whole state.** The angle and the neuron together tell less than the whole state: 2.0 against 2.8 bits on average in stance, 3.0 against 4.7 in swing. The rest is held by the angular velocity and the distance ``d``.
- **Angle only** is the next largest part: about 0.2 bit through most of the cycle, with peaks late in stance (1.7 bits at ``\theta \approx 3.3``) and before the step (1.2 bits at ``\theta \approx 5.9``).
- **Neuron only** and **integrated** are too small to see on this scale; 4B enlarges them.
"""

# ╔═╡ 1e6a33e1-3d9d-54fc-b870-e90424228e91
"""4A: what the whole state, the leg angle and the neuron together, and each part of their information tell about the limit-cycle phase a lag later, over the cycle; swing shaded."""
panel_phase_parts = let lin = cycle_linearization, p = phase_parts, x = wrapped_phase(cycle_linearization)
    plt = cycle_panel(lin; ylabel="information about the phase (bits)", ylims=(0, 1.05 * maximum(p.full)),
        title="What tells the limit-cycle phase a lag later, Δ = $(round(transfer.Δ; digits=2))", legend=:outertop, legend_column=3)
    plot!(plt, x, wrapped(lin, p.full); color=INK_MUTED, linewidth=1.5, linestyle=:dot, label="whole state (y, φ, ω, d)")
    plot!(plt, x, wrapped(lin, p.pair); color=INK, linewidth=1.5, linestyle=:dash, label="angle and neuron together")
    plot!(plt, x, wrapped(lin, p.redundant); color=PART_COLORS.redundant, linewidth=2.5, label="redundant")
    plot!(plt, x, wrapped(lin, p.angle_only); color=PART_COLORS.angle, linewidth=2.5, label="angle only")
    plot!(plt, x, wrapped(lin, p.neuron_only); color=PART_COLORS.neuron, linewidth=2.5, label="neuron only")
    plot!(plt, x, wrapped(lin, p.integrated); color=PART_COLORS.integrated, linewidth=2.5, label="integrated")
end

# ╔═╡ acfc425a-2d52-5ab4-b2fd-bc91c312c334
md"""
**4B. What the neuron adds.** The three small parts of 4A on a scale of one bit (the angle's peaks run off it).

- **Neuron only** appears only around the raise and the backward turn: at most 0.25 bit just after the raise (``\theta \approx 4.2``) and 0.15 bit before the turn (``\theta \approx 4.6``). Elsewhere the angle tells whatever the neuron tells.
- **Integrated** has two peaks. It is 0.51 bit at the raise and 0.88 bit just after it (``\theta \approx 4.2``), and 0.49 bit at the backward turn (``\theta \approx 4.7``), where it is nearly all of the pair's 0.54 bit. It is at most 0.02 bit through stance up to ``\theta \approx 3.8`` and in swing from ``\theta \approx 5.3``.

So near the cycle the neuron adds to what the leg angle tells about the phase only across the raise and the leg's backward turn; through the rest of the cycle it repeats part of what the angle tells. The sizes depend on the noise and on the lag (the summary cell above). The integrated peak is 2.0 bits, at the backward turn, with the neuron's noise a quarter of the leg's, and 0.32 bit, just after the raise, with four times the leg's. It is 1.1 bits at a quarter of the lag and 0.65 bit at four times the lag, both just after the raise.
"""

# ╔═╡ 03a8338e-9f7d-52af-983a-9ed3e656131d
"""4B: the parts of 4A other than the redundant one, up to `top` bits."""
panel_phase_parts_detail = let lin = cycle_linearization, p = phase_parts, x = wrapped_phase(cycle_linearization), top = 1.0
    plt = cycle_panel(lin; xlabel="phase θ from the step (rad)", ylabel="information about the phase (bits)", ylims=(0, top),
        title="The same, enlarged", legend=:outertop, legend_column=3)
    plot!(plt, x, wrapped(lin, p.angle_only); color=PART_COLORS.angle, linewidth=2.5, label="angle only")
    plot!(plt, x, wrapped(lin, p.neuron_only); color=PART_COLORS.neuron, linewidth=2.5, label="neuron only")
    plot!(plt, x, wrapped(lin, p.integrated); color=PART_COLORS.integrated, linewidth=2.5, label="integrated")
end

# ╔═╡ 52616d44-a875-5c6a-9971-5969047ba24a
md"""
**How the leg angle and the neuron carry their information forward.** The integrated information decomposition (Mediano et al. 2021) takes the information that the two variables at time ``t`` hold about the two a lag later,
```math
I\bigl(y_{t+\Delta}, \phi_{t+\Delta};\, y_t, \phi_t\bigr),
```
and splits it by how it is held at each of the two times: redundantly (by either variable), by the neuron only, by the angle only, or synergistically (only by the two together). Four ways at ``t`` times four at ``t + \Delta`` give sixteen parts, the lattice of the decomposition: the angle-only part at ``t`` that is neuron-only at ``t + \Delta`` is what passes from the angle to the neuron, the redundant part that stays redundant is what both keep, and so on. There is no separate target here: this is about the two variables' own future, not about the phase.

The sources and the targets are the neuron state ``y`` and the leg angle ``\phi`` alone; the angular velocity ``\omega`` and the distance ``d`` are left out and not conditioned on. The Walkers, the noise and the lag are 3D's. The redundancies are again minimum mutual informations: of two sources about a target the smaller of their informations, and the double redundancy (redundant at both times) the smallest of the four informations between one variable at ``t`` and one at ``t + \Delta``; the other parts follow from the mutual informations among the four variables (`information_lattice`). With this redundancy a part can come out negative.
"""

# ╔═╡ b81e2f4a-c3dd-598c-8cd1-426a739075f6
begin
    """The ways two variables hold information, in the order of the rows (at t) and columns (at t + Δ) of the lattice."""
    LATTICE_PARTS = ("redundant", "neuron only", "angle only", "synergistic")

    """
        information_lattice(lin, fl, lag)

    The integrated information decomposition (Mediano et al. 2021) of the neuron state y and the leg angle φ
    at each grid phase of `lin`, at a lag of `lag` time steps, for the fluctuations `fl`: `atoms[r, c]`, in
    bits, is the information held at t in the way r and at t + Δ in the way c (`LATTICE_PARTS`), and `total`
    is their sum, I(y, φ at t + Δ; y, φ at t). Redundancies by minimum mutual information.
    """
    function information_lattice(lin, fl, lag)
        n = lin.n
        atoms = [zeros(n) for _ in 1:4, _ in 1:4]
        total = zeros(n)
        Threads.@threads for j in 1:n
            P, R = Matrix(1.0I, 4, 4), zeros(4, 4)
            for i in mod1.(j .+ (0:lag-1), n)   # carry the deviation and the noise on by the lag
                P = lin.Φ[i] * P
                R = lin.Φ[i] * R * lin.Φ[i]' + fl.Q[i]
            end
            W = [I zeros(4, 4); P I]
            S = (W*[fl.C[j] zeros(4, 4); zeros(4, 4) R]*W')[[1, 2, 5, 6], [1, 2, 5, 6]]   # y, φ at t (1, 2) and at t + Δ (3, 4)
            info(now, later) = gaussian_cmi(S, later, now, Int[])
            # between the neuron (N), the angle (A) or both (NA) at t and the neuron (n), the angle (a) or both (na) at t + Δ
            Nn, Na, An, Aa = info([1], [3]), info([1], [4]), info([2], [3]), info([2], [4])
            NAn, NAa, Nna, Ana, NAna = info([1, 2], [3]), info([1, 2], [4]), info([1], [3, 4]), info([2], [3, 4]), info([1, 2], [3, 4])
            a = zeros(4, 4)                                          # rows: at t; columns: at t + Δ
            a[1, 1] = min(Nn, Na, An, Aa)                            # the double redundancy
            a[1, 2] = min(Nn, An) - a[1, 1]                          # the sources' redundancy about the neuron, the angle, both
            a[1, 3] = min(Na, Aa) - a[1, 1]
            a[1, 4] = min(Nna, Ana) - sum(a[1, 1:3])
            a[2, 1] = min(Nn, Na) - a[1, 1]                          # the targets' redundancy about the neuron, the angle, both
            a[3, 1] = min(An, Aa) - a[1, 1]
            a[4, 1] = min(NAn, NAa) - sum(a[1:3, 1])
            a[2, 2] = Nn - a[1, 1] - a[1, 2] - a[2, 1]               # one variable at t with one at t + Δ
            a[2, 3] = Na - a[1, 1] - a[1, 3] - a[2, 1]
            a[3, 2] = An - a[1, 1] - a[1, 2] - a[3, 1]
            a[3, 3] = Aa - a[1, 1] - a[1, 3] - a[3, 1]
            a[2, 4] = Nna - sum(a[1, :]) - sum(a[2, 1:3])            # one variable at t with both at t + Δ
            a[3, 4] = Ana - sum(a[1, :]) - sum(a[3, 1:3])
            a[4, 2] = NAn - sum(a[1:3, 1]) - sum(a[1:3, 2]) - a[4, 1]   # both at t with one variable at t + Δ
            a[4, 3] = NAa - sum(a[1:3, 1]) - sum(a[1:3, 3]) - a[4, 1]
            a[4, 4] = NAna - sum(a)                                  # the rest: synergistic at both times
            for r in 1:4, c in 1:4
                atoms[r, c][j] = a[r, c]
            end
            total[j] = NAna
        end
        (; atoms, total)
    end
end

# ╔═╡ 26c67071-b25e-5533-911b-77f0a90e85ae
"""The integrated information decomposition of the Walker's neuron state and leg angle under `PROCESS_NOISE`, at the lag of the transfer entropy."""
lattice = let lin = cycle_linearization
    information_lattice(lin, walker_fluctuations(lin, PROCESS_NOISE), round(Int, transfer.Δ / lin.Δt))
end

# ╔═╡ 189a31f2-4a65-50d9-9eae-6abc0ea93cbe
# Each part's mean in stance and in swing, its smallest value and its peak; and the total.
let lin = cycle_linearization, stance = cycle_linearization.θ .< cycle_linearization.raise
    average(v, sel) = round(sum(v[sel]) / count(sel); digits=2)
    peak(v) = (bits=round(maximum(v); digits=2), θ=round(lin.θ[argmax(v)]; digits=2))
    row(name, v) = (part=name, stance=average(v, stance), swing=average(v, .!stance), least=round(minimum(v); digits=2), peak=peak(v))
    [[row(LATTICE_PARTS[r] * " → " * LATTICE_PARTS[c], lattice.atoms[r, c]) for r in 1:4 for c in 1:4]; row("total", lattice.total)]
end

# ╔═╡ b2051ffd-7a1f-536d-9d06-2ea9f7676df6
md"""
**5. The lattice.** Each panel is one part, in bits over the cycle: the row is how the information is held at ``t``, the column how it is held at ``t + \Delta``; swing shaded. Every panel has its own scale, and its largest value (in size) is written in it.

- **Redundant → redundant** is most of the total: about 2.3 bits on average in stance and 2.5 in swing, of 3.2 and 3.6. It is nil just after the step (up to ``\theta \approx 0.25``) and at the leg's backward turn.
- **The angle's row** comes next. Angle only → redundant is about 0.4 bit in stance and 0.3 in swing; angle only → angle only is about 0.3 bit in stance, fading toward the raise. Angle only → neuron only, what passes from the angle to the neuron, appears only ahead of the transitions: 0.78 bit before the step and 0.37 bit before the raise.
- **The neuron's row** is empty: none of its parts reaches 0.05 bit in size. The neuron state holds nothing about the pair's future that the angle does not hold too.
- **The synergistic row** is active only around the transitions and the backward turn: synergistic → redundant is 0.76 bit just after the raise and 0.5 bit at the backward turn; synergistic → angle only is 0.58 bit just after the raise and 0.87 bit just before the step; synergistic → synergistic is at most 0.25 bit, before the step.
- **Before the step** (``\theta > 6.03``) the Walker steps within the lag, and the parts there are narrow peaks: the largest is redundant → neuron only, 5.6 bits.

Some parts are negative, down to −0.25 bit (synergistic → neuron only, before the step); this comes with the minimum mutual information redundancy.
"""

# ╔═╡ a9b103d1-eaea-5e5a-98a1-4a58693ae52c
"""5: the sixteen parts of `lattice` over the cycle, rows by how the information is held at t and columns by how it is held at t + Δ; each panel on its own scale, with its largest value written in it; swing shaded."""
panel_lattice = let lin = cycle_linearization, x = wrapped_phase(cycle_linearization)
    colors = (PART_COLORS.redundant, PART_COLORS.neuron, PART_COLORS.angle, PART_COLORS.integrated)
    panels = map([(r, c) for c in 1:4, r in 1:4]) do (r, c)   # row by row
        v = lattice.atoms[r, c]
        lo, hi = min(0.0, minimum(v)), max(maximum(v), 0.05)
        pad = 0.08 * (hi - lo)
        p = cycle_panel(lin; ticks=([0, π, lin.raise, 2π], r == 4 ? ["step", "π", "raise", "step"] : fill("", 4)),
            ylims=(lo - (lo < 0 ? pad : 0.0), hi + pad), legend=false, tickfontsize=7,
            title=r == 1 ? "at t + Δ: " * LATTICE_PARTS[c] : "", xlabel=r == 4 ? "phase θ from the step" : "",
            ylabel=c == 1 ? "at t: " * LATTICE_PARTS[r] * "\n(bits)" : "")
        hline!(p, [0.0]; color=INK_MUTED, linewidth=0.8)
        plot!(p, x, wrapped(lin, v); color=colors[r], linewidth=2)
        annotate!(p, π / 2, hi + 0.3pad, text("largest " * string(round(v[argmax(abs.(v))]; digits=2)), 7, INK_SECONDARY, :center, :top))
    end
    plot(panels...; layout=(4, 4), size=(1500, 1050), left_margin=6Plots.mm, bottom_margin=4Plots.mm,
        plot_title="Integrated information decomposition of the neuron state y and the leg angle φ, lag Δ = $(round(transfer.Δ; digits=2))",
        plot_titlefontsize=12)
end

# ╔═╡ 19beb501-ef3e-53ad-a795-92250c561e94
md"""
## Figures

The panels above, put together and saved to `Figures/` as PNG and PDF. A panel is changed in its own `panel_…` cell, a figure's layout, size and margins in its cell here.
"""

# ╔═╡ 601a995a-5f91-52f1-9400-c072f5980bf0
"""
    panel_letters!(plt; letters = nothing, size = 14, inset = (1.5, 1.0))

Letter the panels of the figure `plt` A, B, C, … in the order of its subplots (row by row), each in
bold at the top left of its panel's box, `inset` millimetres in from the corner (right, down), or
with `letters` (one string per subplot, "" for none, e.g. for the second axes of a two-axes panel).
The letters sit on one transparent subplot laid over the whole figure, so that they are placed alike
on 2D and 3D panels.
"""
function panel_letters!(plt; letters=nothing, size=14, inset=(1.5, 1.0))
    panels = copy(plt.subplots)
    letters = something(letters, [string('A' + k - 1) for k in eachindex(panels)])
    Plots.prepare_output(plt)   # lay the figure out, to find where its panels are
    w, h = Plots.width(plt.layout.bbox), Plots.height(plt.layout.bbox)
    over = length(panels) + 1
    plot!(plt; inset=bbox(0, 0, 1, 1), subplot=over, framestyle=:none, background_color_subplot=:transparent,
        background_color_inside=:transparent, xlims=(0, 1), ylims=(0, 1), legend=false, grid=false, ticks=false)
    for (sp, letter) in zip(panels, letters)
        isempty(letter) && continue
        x = (Plots.left(sp.bbox) + inset[1] * Plots.mm) / w
        y = 1 - (Plots.top(sp.bbox) + inset[2] * Plots.mm) / h
        annotate!(plt, x, y, text(letter, size, INK, :left, :top, "helvetica bold"); subplot=over)
    end
    plt
end

# ╔═╡ 2bc63fd0-4054-5336-ab30-2d110b81be4d
md"""
**Figure 1**, `Oscillator1` (§1): the flow and the limit cycle of (A) the AP Oscillator and (B) the Walker, and the asymptotic phase and the isochrons of (C) the AP Oscillator and (D) the Walker.
"""

# ╔═╡ 80842a03-be24-536c-a4e3-9fc351683ca6
"""Figure 1: each model's flow and cycle (A, B) and its isochrons (C, D), lettered and saved to `Figures/`."""
figure_cycles_isochrons = let
    plt = plot(panel_bingham_flow, panel_walker_flow, panel_bingham_isochrons, panel_walker_isochrons;
        layout=grid(2, 2; widths=[0.44, 0.56]), size=(1150, 1000), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    save_figure(panel_letters!(plt), "Oscillator1")
end

# ╔═╡ 257ec077-aebe-5c33-8a8e-d9759b2837e3
md"""
**Figure 2**, `Oscillator2` (§2): (A) each model's limb over two periods, with its period and amplitude; (B) amplitude against the amplitude parameter, the AP Oscillator's ``c/b`` and the Walker's ``w``; (C) the Walker's period over its neuron's time constant and its body's gains, with the tempo's line; (D) period against the period parameter, the AP Oscillator's ``k`` and the Walker's tempo ``k_{\mathrm{t}}``. B and D give each model relative to its default.
"""

# ╔═╡ 50995748-0744-5653-8c88-a096b05e337c
"""Figure 2: each model's limb over time (A), amplitude against the amplitude parameter (B), the Walker's period over its tempo parameters (C) and period against the period parameter (D), lettered and saved to `Figures/`."""
figure_period_amplitude = let
    top = plot(panel_series_bingham, panel_series_walker; layout=(1, 2))
    bottom = plot(panel_amplitude, panel_period_map, panel_period; layout=(1, 3))
    plt = plot(top, bottom; layout=grid(2, 1; heights=[0.42, 0.58]), size=(1300, 900), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    save_figure(panel_letters!(plt; letters=["A", "", "B", "C", "D"]), "Oscillator2")   # A spans both axes of the top row
end

# ╔═╡ ec9da5fb-3436-5244-acde-bc0ff15fc19c
md"""
**Figure 3**, `Oscillator3` (§3): which phases a noisy reading of (A) the Walker's leg angle, (B) its neuron state or (C) both leaves open, and (D) the transfer entropy between the leg and the neuron over the cycle.
"""

# ╔═╡ 0844edfb-ea0e-5fd2-a5cd-cb21e1cb491e
"""Figure 3: the phases a noisy reading leaves open (A–C) and the transfer entropy over the cycle (D), lettered and saved to `Figures/`."""
figure_phase_information = let
    top = plot(panel_posterior_leg, panel_posterior_neuron, panel_posterior_both; layout=grid(1, 3; widths=[0.285, 0.285, 0.43]))
    plt = plot(top, panel_transfer_entropy; layout=grid(2, 1; heights=[0.52, 0.48]), size=(1150, 820),
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    save_figure(panel_letters!(plt), "Oscillator3")
end

# ╔═╡ 58d309cc-ec75-5d5b-9416-50495c7dad69
md"""
**Figure 4**, `Oscillator4` (§3): (A) what the Walker's whole state, its leg angle and neuron state together, and each part of the two's information tell about the limit-cycle phase a lag later, over the cycle; (B) the parts other than the redundant one, enlarged.
"""

# ╔═╡ a986bc11-ff92-51ef-a22c-f8a03fead55e
"""Figure 4: the information about the limit-cycle phase and its parts (A) and the small parts enlarged (B), lettered and saved to `Figures/`."""
figure_phase_parts = let
    plt = plot(panel_phase_parts, panel_phase_parts_detail; layout=grid(2, 1; heights=[0.58, 0.42]), size=(1150, 860),
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    save_figure(panel_letters!(plt), "Oscillator4")
end

# ╔═╡ a964de67-1a89-580a-8924-7a2c52ea7365
md"""
**Figure 5**, `Oscillator5` (§3): the integrated information decomposition of the Walker's neuron state and leg angle over the cycle, the sixteen parts by how the information is held at ``t`` (rows) and at ``t + \Delta`` (columns).
"""

# ╔═╡ 276b8456-339a-5a2d-b6d5-bab45d8459d3
"""Figure 5: the lattice of the integrated information decomposition, saved to `Figures/`."""
figure_lattice = save_figure(panel_lattice, "Oscillator5")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HybridSystems = "2207ec0c-686c-5054-b4d2-543502888820"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MathematicalSystems = "d14a8603-c872-5ed3-9ece-53e0e82e39da"
OrdinaryDiffEqTsit5 = "b1df2697-797e-41e3-8120-5422d3b24e4a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
HybridSystems = "~0.4.4"
LazySets = "~7.1.0"
MathematicalSystems = "~0.14.6"
OrdinaryDiffEqTsit5 = "~2.1.5"
Plots = "~1.41.7"
StaticArrays = "~1.9.23"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "bd8a180bc170c272ef7e5203e42d96289bff6917"

[[deps.ADTypes]]
deps = ["PrecompileTools"]
git-tree-sha1 = "629de23e1c16911b439dabd2303c08af9575b226"
registries = "General"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.24.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "dee8388a9bec89c8713ad176f96960ae2f98598a"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.4"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
registries = "General"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "1aec1ff0dcaa83a484e7f72097562fa3102c6722"
registries = "General"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.30.2"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceFillArraysExt = "FillArrays"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceGPUArraysCoreTrackerExt = ["GPUArraysCore", "Tracker"]
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BinaryHeaps]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0adb6d8d0a4bae93a07d79c8d59eaf617f4c59da"
registries = "General"
uuid = "b2a6c25c-c996-4615-94ab-6e519ffb8690"
version = "1.1.0"

[[deps.BracketingNonlinearSolve]]
deps = ["CommonSolve", "ConcreteStructs", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "b1b04c236ff7ed92e69d64b620448b3da00933d4"
registries = "General"
uuid = "70df07ce-3d50-431d-a3e7-ca6ddb60ac1e"
version = "1.12.8"

    [deps.BracketingNonlinearSolve.extensions]
    BracketingNonlinearSolveChainRulesCoreExt = ["ChainRulesCore", "ForwardDiff"]
    BracketingNonlinearSolveForwardDiffExt = "ForwardDiff"

    [deps.BracketingNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
registries = "General"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
registries = "General"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7b841680738c19948120f6e4cf8d0200518bd564"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+0"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
registries = "General"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
registries = "General"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a7cf827a79b378a381e31e5c7feca853ec19cb94"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.4"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "a633156e74096f90b3b5e833529292b651b5671a"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.3"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
registries = "General"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a72c3b5ce6d2a477f55b5c9b8756e91e695c67f6"
registries = "General"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.8"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
registries = "General"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
registries = "General"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
registries = "General"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "BracketingNonlinearSolve", "ConcreteStructs", "DocStringExtensions", "FastBroadcast", "FastClosures", "FastPower", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "RespecializeParams", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "87a851d87d4a1ea5dae17f349aad2ade9a587f08"
registries = "General"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "7.21.4"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseDynamicQuantitiesExt = "DynamicQuantities"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseFlexUnitsExt = "FlexUnits"
    DiffEqBaseForwardDiffExt = ["ForwardDiff"]
    DiffEqBaseGTPSAExt = "GTPSA"
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseMooncakeExt = "Mooncake"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseSparseArraysExt = "SparseArrays"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FlexUnits = "76e01b6b-c995-4ce6-8559-91e72a3d4e95"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
registries = "General"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
registries = "General"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "0693d8b0a4608ff289d228ab4c598df5894845cd"
registries = "General"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.21"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = ["GPUArraysCore", "Adapt"]
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceHyperHessiansExt = "HyperHessians"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    HyperHessians = "06b494a0-c8e0-40cc-ad32-d99506a00a6c"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
registries = "General"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EnzymeCore]]
git-tree-sha1 = "156ede9181d50b3b21b74d98a7edb08bc7bbdbf1"
registries = "General"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.23"

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

    [deps.EnzymeCore.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
registries = "General"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Espresso]]
deps = ["LinearAlgebra", "Statistics", "Test"]
git-tree-sha1 = "0be6cc94fc2d31fa1abd4f209e976ad09bfd195f"
registries = "General"
uuid = "6912e4f1-e036-58b0-9138-08d1e6358ea9"
version = "0.6.4"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
registries = "General"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "7bd13840b4148949e290748071bbe6826b3eabcd"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "b90aee4f191474eeda3dfa0cb588635f80820cf1"
registries = "General"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "1.5.0"

    [deps.FastBroadcast.extensions]
    FastBroadcastPolyesterExt = "Polyester"
    FastBroadcastStaticExt = "Static"
    FixedSizeArraysExt = "FixedSizeArrays"

    [deps.FastBroadcast.weakdeps]
    FixedSizeArrays = "3821ddf9-e5b5-40d5-8e25-6813ab96b5e2"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    Static = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
registries = "General"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastPower]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6bcbd090fc6b6f022fc0ad96f77ba31b9577282"
registries = "General"
uuid = "a4df4552-cc26-4903-aec0-212e50a0e84b"
version = "1.5.1"

    [deps.FastPower.extensions]
    FastPowerEnzymeExt = "Enzyme"
    FastPowerForwardDiffExt = "ForwardDiff"
    FastPowerMeasurementsExt = "Measurements"
    FastPowerMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    FastPowerMooncakeExt = "Mooncake"
    FastPowerReverseDiffExt = "ReverseDiff"
    FastPowerTrackerExt = "Tracker"

    [deps.FastPower.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "417289ac840d201005d9ed16789a46defccbe36d"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.1"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FindFirstFunctions]]
deps = ["PrecompileTools"]
git-tree-sha1 = "93eb9afa3a53b29d8a49714b83878225402b3281"
registries = "General"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "3.4.1"

    [deps.FindFirstFunctions.extensions]
    FindFirstFunctionsEnzymeCoreExt = "EnzymeCore"
    FindFirstFunctionsMooncakeExt = "Mooncake"

    [deps.FindFirstFunctions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "3b0f72e2ffef1a139ac450725c9ecd01c0a3c050"
registries = "General"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.6"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
registries = "General"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "2bcce3ad6f6977d617928d7707fdc86ac83cce03"
registries = "General"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "1.13.0"

    [deps.FunctionWrappersWrappers.extensions]
    FunctionWrappersWrappersEnzymeExt = ["Enzyme", "EnzymeCore"]
    FunctionWrappersWrappersMooncakeExt = "Mooncake"

    [deps.FunctionWrappersWrappers.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
registries = "General"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "1d706bd23e5d2d407bfd369499ee6f96afb0c3ad"
registries = "General"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.2.1"

[[deps.GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6aa6294ba949ccfc380463bf50ff988b46de5bc7"
registries = "General"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+1"

[[deps.GMP_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.3.0+2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "0b2b3e8b8fb7c1f5ddbf498be2eb2b3b37c9c0ed"
registries = "General"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
registries = "General"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
registries = "General"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Inflate", "LinearAlgebra", "Random", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "dbb2b976f605b220dfe139d6db9f3859680da09e"
registries = "General"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.15.0"

    [deps.Graphs.extensions]
    GraphsSharedArraysExt = "SharedArrays"

    [deps.Graphs.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    SharedArrays = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.HybridSystems]]
deps = ["FillArrays", "Graphs", "LinearAlgebra", "MappedArrays", "MathematicalSystems"]
git-tree-sha1 = "8ed6051d1020f05b5957254dfe24d25fb8acf602"
registries = "General"
uuid = "2207ec0c-686c-5054-b4d2-543502888820"
version = "0.4.4"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
registries = "General"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "1c531bf0f8a5c60a340926e058fd3f209b5eef5d"
registries = "General"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.12"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticMakieExt = "Makie"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
registries = "General"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "9939eb6f799a963b3392ac4a1d49d3fc0646eb80"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.11.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "c21108220410d7b77d0385c487259abb7e5c97db"
registries = "General"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.32.1"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LazySets]]
deps = ["ExprTools", "GLPK", "IntervalArithmetic", "JuMP", "LinearAlgebra", "Random", "ReachabilityBase", "Reexport", "SparseArrays"]
git-tree-sha1 = "8575c336bd347522349cf625541393a2acec001c"
registries = "General"
uuid = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
version = "7.1.0"

    [deps.LazySets.extensions]
    LazySetsCDDLibExt = "CDDLib"
    LazySetsDistributedExt = ["Distributed", "SharedArrays"]
    LazySetsDistributionsExt = ["Distributions", "Random", "ReachabilityBase"]
    LazySetsExpokitExt = "Expokit"
    LazySetsExponentialUtilitiesExt = "ExponentialUtilities"
    LazySetsGeometryBasicsExt = ["GeometryBasics", "Polyhedra"]
    LazySetsIntervalBoxesExt = ["IntervalArithmetic", "IntervalBoxes"]
    LazySetsIntervalConstraintProgrammingExt = ["IntervalBoxes", "IntervalConstraintProgramming", "Symbolics"]
    LazySetsIntervalMatricesExt = ["IntervalArithmetic", "IntervalMatrices", "LinearAlgebra", "ReachabilityBase"]
    LazySetsIpoptExt = "Ipopt"
    LazySetsJavisExt = "Javis"
    LazySetsMakieExt = ["Makie", "Polyhedra"]
    LazySetsMiniQhullExt = "MiniQhull"
    LazySetsOptimExt = "Optim"
    LazySetsPolyhedraExt = ["JuMP", "GLPK", "Polyhedra"]
    LazySetsRangeEnclosuresExt = ["IntervalArithmetic", "RangeEnclosures"]
    LazySetsRecipesBaseExt = ["ReachabilityBase", "RecipesBase"]
    LazySetsSCSExt = ["MathOptInterface", "SCS"]
    LazySetsSetProgExt = ["JuMP", "SetProg"]
    LazySetsStaticArraysCoreExt = ["ReachabilityBase", "StaticArraysCore"]
    LazySetsSymEngineExt = "SymEngine"
    LazySetsSymbolicsExt = ["SymbolicUtils", "Symbolics"]
    LazySetsTaylorModelsExt = ["IntervalArithmetic", "TaylorModels", "TaylorSeries"]
    LazySetsWriteVTKExt = "WriteVTK"

    [deps.LazySets.weakdeps]
    CDDLib = "3391f64e-dcde-5f30-b752-e11513730f60"
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Expokit = "a1e7a1ef-7a5d-5822-a38c-be74e1bb89f4"
    ExponentialUtilities = "d4d017d3-3776-5f7e-afef-a10c40355c18"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IntervalBoxes = "43d83c95-ebbb-40ec-8188-24586a1458ed"
    IntervalConstraintProgramming = "138f1668-1576-5ad7-91b9-7425abbf3153"
    IntervalMatrices = "5c1f47dc-42dd-5697-8aaa-4d102d140ba9"
    Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
    Javis = "78b212ba-a7f9-42d4-b726-60726080707e"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
    MiniQhull = "978d7f02-9e05-4691-894f-ae31a51d76ca"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"
    Polyhedra = "67491407-f73d-577b-9b50-8179a7c68029"
    RangeEnclosures = "1b4d18b6-9e5d-11e9-236c-f792b01831f8"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SCS = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"
    SetProg = "39881422-4b75-5582-a5c7-0feb14562a65"
    SharedArrays = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    TaylorModels = "314ce334-5f6e-57ae-acf6-00b6e903104a"
    TaylorSeries = "6aa5eb33-94cf-58f4-a9d0-e4b2c4fc25ea"
    WriteVTK = "64499a7a-5c06-52f2-abe2-ccb03c286192"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
registries = "General"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
registries = "General"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptInterface]]
deps = ["CodecBzip2", "CodecZlib", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "637aa0115265323850b125287833341a6f7b9b54"
registries = "General"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.54.0"

    [deps.MathOptInterface.extensions]
    MathOptInterfaceBenchmarkToolsExt = "BenchmarkTools"
    MathOptInterfaceCliqueTreesExt = "CliqueTrees"

    [deps.MathOptInterface.weakdeps]
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"

[[deps.MathematicalSystems]]
deps = ["Espresso", "InteractiveUtils", "LinearAlgebra", "MacroTools", "MultivariatePolynomials", "SparseArrays"]
git-tree-sha1 = "5a459ff534e915ec93c70da90f2d051b8833541c"
registries = "General"
uuid = "d14a8603-c872-5ed3-9ece-53e0e82e39da"
version = "0.14.6"
weakdeps = ["LazySets", "RecipesBase", "Requires"]

    [deps.MathematicalSystems.extensions]
    LazySetsExt = "LazySets"
    RecipesBaseExt = "RecipesBase"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "PrecompileTools"]
git-tree-sha1 = "f2cde0ae772162f20287b803e623966d0d94dea9"
registries = "General"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.8"
weakdeps = ["SparseArrays"]

    [deps.MaybeInplace.extensions]
    MaybeInplaceSparseArraysExt = "SparseArrays"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.10.20"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
registries = "General"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.MultivariatePolynomials]]
deps = ["DataStructures", "LinearAlgebra", "MutableArithmetics", "StarAlgebras"]
git-tree-sha1 = "65f1160de342181533efa7061f86539d420ab81d"
registries = "General"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.20"

    [deps.MultivariatePolynomials.extensions]
    LatexifyExt = "Latexify"
    MultivariatePolynomialsChainRulesCoreExt = "ChainRulesCore"

    [deps.MultivariatePolynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "158af80b4d50a7bdc37c8357c4c4554c335cfe5f"
registries = "General"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.9.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.NonlinearSolveBase]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "CommonSolve", "Compat", "ConcreteStructs", "DifferentiationInterface", "EnumX", "EnzymeCore", "FastClosures", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "LogExpFunctions", "Markdown", "MaybeInplace", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "RespecializeParams", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "00768f7037ef9d876ce94dbdc253f82720cf3f5d"
registries = "General"
uuid = "be0214bd-f91f-a760-ac4e-3421ce2b2da0"
version = "2.55.0"

    [deps.NonlinearSolveBase.extensions]
    NonlinearSolveBaseBandedMatricesExt = "BandedMatrices"
    NonlinearSolveBaseChainRulesCoreExt = "ChainRulesCore"
    NonlinearSolveBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    NonlinearSolveBaseForwardDiffExt = "ForwardDiff"
    NonlinearSolveBaseLineSearchExt = "LineSearch"
    NonlinearSolveBaseLinearSolveExt = "LinearSolve"
    NonlinearSolveBaseMooncakeExt = ["ChainRulesCore", "Mooncake"]
    NonlinearSolveBaseReverseDiffExt = "ReverseDiff"
    NonlinearSolveBaseSparseArraysExt = "SparseArrays"
    NonlinearSolveBaseSparseMatrixColoringsExt = "SparseMatrixColorings"
    NonlinearSolveBaseTrackerExt = "Tracker"

    [deps.NonlinearSolveBase.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LineSearch = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
    LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "38a93f17e431141c6470bb67a88952a7c4f0e928"
registries = "General"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.34+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.OrdinaryDiffEqCore]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "BinaryHeaps", "CommonSolve", "ConstructionBase", "DiffEqBase", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "FastPower", "FindFirstFunctions", "FunctionWrappers", "FunctionWrappersWrappers", "InteractiveUtils", "LinearAlgebra", "Logging", "MacroTools", "MuladdMacro", "PrecompileTools", "Preferences", "Printf", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "5a995982a30ce31999c5504569234ecb91a6e0ef"
registries = "General"
uuid = "bbf590c4-e513-4bbe-9b18-05decba2e5d8"
version = "4.19.0"

    [deps.OrdinaryDiffEqCore.extensions]
    OrdinaryDiffEqCoreMooncakeExt = "Mooncake"
    OrdinaryDiffEqCorePolyesterExt = "Polyester"
    OrdinaryDiffEqCoreSparseArraysExt = "SparseArrays"

    [deps.OrdinaryDiffEqCore.weakdeps]
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.OrdinaryDiffEqTsit5]]
deps = ["CommonSolve", "DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "371de3eb5985febd9d7babb0937520653f2b15ca"
registries = "General"
uuid = "b1df2697-797e-41e3-8120-5422d3b24e4a"
version = "2.1.5"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.47.0+0"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
registries = "General"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
registries = "General"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "cd00e28071a75c98664663f87c2ae447d25c0503"
registries = "General"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "1.7.1"

    [deps.PreallocationTools.extensions]
    PreallocationToolsEnzymeCoreExt = "EnzymeCore"
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "99fd8f32ab6728ff205cddd24e2d9ea6cbb498d3"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.0"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"
    PrettyTablesWriteDocxExt = "WriteDocx"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    WriteDocx = "d049ceea-54ee-41d7-a26f-ba29db3b6599"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
registries = "General"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
registries = "General"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
registries = "General"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
registries = "General"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
registries = "General"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.ReachabilityBase]]
deps = ["ExprTools", "InteractiveUtils", "LinearAlgebra", "Random", "Requires", "SparseArrays"]
git-tree-sha1 = "8c4ee9dd72d7ba962d5d4a0974f7e8f98fe275f6"
registries = "General"
uuid = "379f33d0-9447-4353-bd03-d664070e549f"
version = "0.3.6"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0c0dcffc464cda2ff33fc8ffc1dd1bdd53cd98be"
registries = "General"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.4.0"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
registries = "General"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "GPUArraysCore", "LinearAlgebra", "PrecompileTools", "RecipesBase", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "e6391348505df3c9c33008c0fec050862b834bf4"
registries = "General"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "4.5.5"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsCUDAExt = "CUDA"
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsFastBroadcastPolyesterExt = ["FastBroadcast", "Polyester"]
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsMooncakeExt = "Mooncake"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStatisticsExt = "Statistics"
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.RespecializeParams]]
deps = ["FunctionWrappersWrappers", "PrecompileTools"]
git-tree-sha1 = "140c074bb63518a0e1ec2c8aba2a8a5719bcf21c"
registries = "General"
uuid = "9fe22ead-9e00-4db2-8b46-706a60d40f5e"
version = "1.3.0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
registries = "General"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "PrecompileTools", "SHA", "Serialization"]
git-tree-sha1 = "9bb075002b40849582905394bb3031fb77aa102d"
registries = "General"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.27"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FindFirstFunctions", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "LoggingExtras", "Markdown", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "Random", "RecipesBase", "RecursiveArrayTools", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "85f85c3c5b949e7d691047e758e3fdb77c21310c"
registries = "General"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "3.58.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDifferentiationInterfaceExt = "DifferentiationInterface"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseEnzymeExt = "Enzyme"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseFunctionPropertiesExt = "FunctionProperties"
    SciMLBaseMakieExt = ["Makie", "GeometryBasics"]
    SciMLBaseMeasurementsExt = "Measurements"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReactantCoreExt = "ReactantCore"
    SciMLBaseReactantExt = "Reactant"
    SciMLBaseReverseDiffExt = ["ReverseDiff", "ForwardDiff"]
    SciMLBaseStaticArraysExt = "StaticArrays"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore", "ZygoteRules", "FillArrays"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    FunctionProperties = "f62d2435-5019-4c03-9749-2d4c77af0cbc"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReactantCore = "a3311ec8-5e00-46d5-b541-4f83e724a433"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.SciMLJacobianOperators]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DifferentiationInterface", "FastClosures", "LinearAlgebra", "SciMLBase", "SciMLOperators"]
git-tree-sha1 = "781d5413c2991e581da86979b0d9f86f9a034a1c"
registries = "General"
uuid = "19f34311-ddf3-4b8b-af20-060888a46c0e"
version = "0.1.20"

[[deps.SciMLLogging]]
deps = ["Logging", "LoggingExtras", "PrecompileTools", "Preferences"]
git-tree-sha1 = "3aafee7edc2ec7e71f50dcc6816cb031fac390a0"
registries = "General"
uuid = "a6db7da4-7206-11f0-1eab-35f2a5dbe1d1"
version = "2.1.0"

    [deps.SciMLLogging.extensions]
    SciMLLoggingTracyExt = "Tracy"

    [deps.SciMLLogging.weakdeps]
    Tracy = "e689c965-62c8-4b79-b2c5-8359227902fd"

[[deps.SciMLOperators]]
deps = ["Accessors", "Adapt", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "2712b6eeef282b5cd8c38660e68c001a4b9b24e2"
registries = "General"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.30.3"

    [deps.SciMLOperators.extensions]
    SciMLOperatorsLoopVectorizationExt = "LoopVectorization"
    SciMLOperatorsSparseArraysExt = "SparseArrays"

    [deps.SciMLOperators.weakdeps]
    LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
registries = "General"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.SciMLStructures]]
deps = ["ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "4b8fc0ffe22b4ecba7a02f1fec0b3757db014d0c"
registries = "General"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.6"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
registries = "General"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
registries = "General"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
registries = "General"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StarAlgebras]]
deps = ["LinearAlgebra", "MutableArithmetics", "SparseArrays"]
git-tree-sha1 = "235b1f9d287bbf34083b3d0829343a7942c0ad1c"
registries = "General"
uuid = "0c0c59c1-dc5f-42e9-9a8b-b5dc384a6cd1"
version = "0.3.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "de349d5585559b692f518b6d3fc7c179d64b2364"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.23"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StringManipulation]]
deps = ["PrecompileTools", "StyledStrings"]
git-tree-sha1 = "9e3a02d73a8f0a9be04e677dbef758aab40f8bea"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.6.1"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ad435e30854ed0f50f4b7f67edd71664afecc5d8"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.10.0"

    [deps.StructUtils.extensions]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "PrecompileTools", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "7eb6da9656581ac9fbffaef3ef7950a090a5002a"
registries = "General"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.55"
weakdeps = ["PrettyTables"]

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "PrettyTables", "Printf", "Tables"]
git-tree-sha1 = "2244393f04554e24b54896b430b2ab3850a2232d"
registries = "General"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "1.2.2"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
registries = "General"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
registries = "General"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
registries = "General"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
registries = "General"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
registries = "General"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
registries = "General"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
registries = "General"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
registries = "General"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
registries = "General"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
registries = "General"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
registries = "General"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
registries = "General"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
registries = "General"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
registries = "General"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
registries = "General"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
registries = "General"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
registries = "General"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
registries = "General"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "9a5e6e8f1a7c40941465c04840145e527f0d6b4b"
registries = "General"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.5.0+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "df77b22eb77051b3d0ef30aab979684f7482c66f"
registries = "General"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.48.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.2+0"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
registries = "General"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
registries = "General"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1210ba774d3427387d307bf1f416d699b7c39417"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
registries = "General"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
registries = "General"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
registries = "General"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
registries = "General"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.70.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.3+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
registries = "General"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╠═c8e259bd-ee05-5a01-b3f3-041f90e88eba
# ╟─6a668412-35fe-5490-a116-5d17e58f2463
# ╟─9fadde5b-0de4-5f35-be20-f56a9652312e
# ╠═37559631-8146-574d-937f-87937916fa28
# ╠═9293fc3c-70a1-551f-818c-5c3351bf9d58
# ╠═dbfe1397-bd52-565d-828b-3ccf6987d497
# ╟─936359bc-1138-5c1e-9ea8-c07b766f0d0b
# ╠═5e24b94c-5a42-56a6-bb43-d36508289c42
# ╠═af440e5c-b7ff-550e-9119-deeb3d4c7299
# ╠═c6f5c4cb-cc55-57d6-a777-ab4cade819e2
# ╠═bb049417-5b2e-5874-ab69-a3032f748d60
# ╠═34cba43a-1415-589a-8890-3db840d56045
# ╠═4bdf9cb7-ec83-5b85-9a7c-92f932075ead
# ╠═9e3c912e-1399-59f1-9583-a2fa6f67ec27
# ╟─39dffbd4-ce74-51c4-b66f-c54427af51ca
# ╠═bb2a3a73-893c-5b03-8a2e-3767cc6642ab
# ╠═1ab1462a-11ad-52c7-ae74-db848691f38f
# ╠═48361514-d110-5f81-8109-0eefc9342f4c
# ╠═d5154c7d-e8ff-58c6-9e4f-350f0734ad2e
# ╠═45731442-548e-5dc9-abbe-9415046875c4
# ╟─7d639d49-7f6d-5f58-ab45-45b423a11ca9
# ╠═8f5fc848-e32b-5e85-a5c9-b072115b8c14
# ╠═99601001-0dd8-5301-b63f-e54624b8e240
# ╠═3fe70423-3ede-5f2a-b638-7daaf4e8d177
# ╠═a026ce9f-663d-5380-bacf-44a812d7da5f
# ╠═1d325bda-3929-53dc-823c-a1dab00b723e
# ╠═241b08d5-3a45-534c-8e2c-07cd8fffdc1f
# ╠═1adc49de-20b1-5e7b-9294-2715ee344557
# ╠═8dd71357-3bee-5ec5-a0d1-e32de585c54b
# ╟─eced8e23-18d8-5c59-bec2-67b6a04e2569
# ╠═aebfa6c1-94c4-5147-bc72-85a6ff778d3c
# ╟─f0c94cc4-bbfb-52aa-9eab-ec019cf4bc3f
# ╠═6081c9f2-fc31-5c71-aad4-9fc3bea06da2
# ╟─706ecbf6-52f0-5dba-9236-b2cb496b359f
# ╠═6e363438-2f18-5ec5-af13-60394bd7b7b2
# ╟─c04e9129-06ac-581a-b075-5ad0af9ddf47
# ╠═73bc3e0c-725b-5cb7-8e48-d9f507a299c3
# ╟─61033d71-7a1f-5f2a-b9e1-ca2c7a0a2fdb
# ╠═37549ce4-1ff3-518e-84ff-22fd5cbadabf
# ╠═b07aeb1f-6c56-5a75-94c0-c337b1a08398
# ╠═e66d216b-8832-5ea2-90bf-d6b5bf9bc054
# ╠═bda21176-06a2-5550-aa43-78890fa2aaee
# ╠═73b02df6-a852-522d-b792-68c456b44641
# ╠═d13d93e1-7efe-5e79-85b4-907d08387557
# ╠═e62dac68-adf7-555a-950a-1d6877168df0
# ╠═754e48c6-94fe-512a-9862-17f616636c30
# ╠═d7c2a79d-829d-5d3f-949d-9f4e65da9f27
# ╠═26358daa-f01b-5690-8ca8-9a18d31c1256
# ╠═b02a317c-d23c-5390-a8d0-0b390f63e9ad
# ╠═c1eaabd2-863d-5f08-b176-96ce8a17d25a
# ╟─fcd02053-3e73-51ab-9ff3-1f5cf75a6214
# ╠═93b47daa-8251-50fa-8dee-6ce33d207574
# ╠═67ed563c-6c01-5012-9534-8a2f721c149f
# ╟─0c0200bb-3329-57ed-8db5-02b31640425d
# ╠═df3fec1b-01cc-5c6a-af24-51449a86f5b9
# ╟─60160519-4337-5c94-b6af-0a31a0b6d2a2
# ╠═ccab8dc2-5bc4-52a6-848a-c592b12a3753
# ╟─e559ce9d-f0ab-5294-a5d9-fe209fe7e092
# ╠═8c9e8edd-d561-5405-97a7-af15a911ebcc
# ╟─7c1d0b55-591f-50ed-b930-2f22ac3826a4
# ╠═9638f418-856d-5232-8d31-6a62f14e0e49
# ╠═ebd07e1f-7590-59cd-a18f-d2bf9f074ad2
# ╠═b59a6d1d-e76c-5cfa-960e-fc7da0c48ffe
# ╠═154a84dd-2220-5b80-8101-006c71936b8b
# ╟─a5d004ee-6090-5865-931e-166785781290
# ╠═812ce438-74af-5b9d-b90b-95e3f1250149
# ╠═868b17cb-3c76-5349-adb9-e8c4aa91ef8d
# ╠═c233065a-b044-5ccf-9f77-78aa672147dd
# ╠═e4bde1f4-aa5c-5bac-bce1-25e356887177
# ╠═957508eb-3c75-5007-bb5c-c619da65c58c
# ╠═b1ec5ec6-febf-5240-a230-629e11d7e0ab
# ╠═196cd759-e4ac-52d4-b9b6-a177b5febf99
# ╠═f83d5699-3902-583c-a549-021d9865bfa2
# ╠═4888772a-0ed5-5d35-86df-85ae210d6577
# ╟─8666d1d5-050a-572d-b28f-8452d3b3a34e
# ╠═f11f3bdf-39e7-543f-ae07-0717d39beebb
# ╟─fd344984-024c-5dfe-a0ee-fd981faa3c68
# ╠═563d5ab4-0baf-5bbb-9a47-347e0ce34551
# ╟─a7eaae3b-f124-5e0f-b4e8-81d3e4e8849c
# ╠═8cd368f7-295c-517e-be9e-4c30409778d2
# ╟─db69588a-fb2f-5cbf-8b8c-d577b21addfd
# ╠═09771a80-8069-509f-bc09-c5f24c484a00
# ╟─52d47a18-928a-52e6-977b-59c3a3a0b32c
# ╠═24500ad0-1c5c-5f1f-bbb1-a02c5274a784
# ╠═817d092a-5075-54e7-bfbe-f5169c58b584
# ╠═d4c811c6-ede7-5eec-9072-442861b329aa
# ╠═77d70363-26b4-54ce-bbb9-2eb9f7f1b304
# ╠═0cc7341a-fc16-59fd-a37e-625606663aa6
# ╠═bf676518-1139-536e-87e9-81bc793d357c
# ╟─dc973ca1-6332-576b-90ad-9dafe0981a44
# ╠═1e6a33e1-3d9d-54fc-b870-e90424228e91
# ╟─acfc425a-2d52-5ab4-b2fd-bc91c312c334
# ╠═03a8338e-9f7d-52af-983a-9ed3e656131d
# ╟─52616d44-a875-5c6a-9971-5969047ba24a
# ╠═b81e2f4a-c3dd-598c-8cd1-426a739075f6
# ╠═26c67071-b25e-5533-911b-77f0a90e85ae
# ╠═189a31f2-4a65-50d9-9eae-6abc0ea93cbe
# ╟─b2051ffd-7a1f-536d-9d06-2ea9f7676df6
# ╠═a9b103d1-eaea-5e5a-98a1-4a58693ae52c
# ╟─19beb501-ef3e-53ad-a795-92250c561e94
# ╠═601a995a-5f91-52f1-9400-c072f5980bf0
# ╟─2bc63fd0-4054-5336-ab30-2d110b81be4d
# ╠═80842a03-be24-536c-a4e3-9fc351683ca6
# ╟─257ec077-aebe-5c33-8a8e-d9759b2837e3
# ╠═50995748-0744-5653-8c88-a096b05e337c
# ╟─ec9da5fb-3436-5244-acde-bc0ff15fc19c
# ╠═0844edfb-ea0e-5fd2-a5cd-cb21e1cb491e
# ╟─58d309cc-ec75-5d5b-9416-50495c7dad69
# ╠═a986bc11-ff92-51ef-a22c-f8a03fead55e
# ╟─a964de67-1a89-580a-8924-7a2c52ea7365
# ╠═276b8456-339a-5a2d-b6d5-bab45d8459d3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
