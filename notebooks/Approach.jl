### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 794af616-4626-52e5-a088-2a4e2aada361
begin
    # Packages used by the shared modules, named here so Pluto installs them.
    using MathematicalSystems
    import HybridSystems, LazySets
    using OrdinaryDiffEqTsit5
    using Plots
    using LinearAlgebra
    # The floating index of sections.
    using PlutoUI: TableOfContents

    include(joinpath(@__DIR__, "..", "src", "BehaviorTools.jl"))
    include(joinpath(@__DIR__, "..", "src", "AgentModels.jl"))
    using .BehaviorTools, .AgentModels
    using .AgentModels: heading_error
    figure_style!()
end

# ╔═╡ a1f3c2d4-6b7e-4c59-8d2a-3e4f5a6b7c8d
TableOfContents()

# ╔═╡ 566196d3-53ee-5ec0-a4e1-dfb4c1ee7454
md"""
# Approach, version 3: the chapter's vehicle and Warren's law

**New to these models?** Start with the two introductions at the head of §1 and §2. They explain each figure, and the code that makes it, in plain words. The rest of each section gives the details and checks the numbers. The index on the right stays in view as you scroll; click a heading to jump to it.

Version 3 is `ApproachV2.jl` with figure 2's three-parameter charts (2C, 2D) in the colours of figure 1. Their far walls show the regions they cut, and the sheet of critical damping is a slightly transparent pale yellow, the colour of the overshoot above it. Version 2 drew both in grey. Nothing else changes: the models, the numbers and figure 1 are version 2's.

Figures for the approach example in the chapter's Results (`chap3.tex`), in the chapter's notation. Two models steer to a stationary goal at the origin: the chapter's Type B Braitenberg vehicle and Fajen & Warren's (2003) steering law. Both are MathematicalSystems.jl `BlackBoxContinuousSystem`s run with the shared `trajectory` (`BehaviorTools.jl`). Warren's law is `warren_approach` from `AgentModels.jl`, and the vehicle is `chapter_vehicle`, defined below.

This notebook is a shorter companion to `ApproachBehavior.jl`, with one change of model. There the vehicle moves at the speed its motors give, which grows as it nears the stimulus. Here, as in the chapter, a constant forward drive (Hotton's "forward neuron") moves it at a constant speed ``V``, the walker's speed. The charts for a goal at a fixed distance do not depend on the speed, so they are those of `ApproachBehavior.jl` §4. The paths are not.

Figures, saved to `Figures/` as PNG and PDF:

1. `Approach1`: the paths of (A) the vehicle and (B) Warren's walker, and the parameter charts of (C) the vehicle's two crossed synapses and (D) Warren's damping and stiffness.
2. `Approach2`: (A) the two steering functions and (B) the heading error over time, and the three-parameter charts of (C) the vehicle's lag, decay and motor strength ``(\tau, c_d, m)`` and (D) Warren's damping, decay and stiffness ``(b, c_1, k_t)``.

Each section opens with its method and the cells that compute it, followed by checks of the numbers its text quotes. Then come the helpers that draw its panels, and then the panels themselves, each `panel_…` cell below the text that describes it. The `figure_…` cells in **Figures**, at the bottom of the notebook, lay the panels out into the two figures, letter them A, B, … and save them; the text above each panel begins with its figure and letter (1A, 1B, …). The settings of a panel worth changing (limits, starting headings, the camera) are named at the top of its cell.

Orange is the Braitenberg vehicle and blue is Warren's law, in every panel. The regions of the charts have the same colours in both figures: green approach, pale yellow approach (overshoot), pale pink avoid and lavender wander. In the charts, a black line is a bifurcation, a white line a degenerate one, and grey dots a boundary that is not a bifurcation; the dash gives the kind of bifurcation (§1, and a legend under each figure).
"""

# ╔═╡ c7b8b1ab-b73b-5594-8c9f-a891d3b804c4
md"""
## The two models

**Braitenberg vehicle.** A Type B vehicle (crossed excitatory connections) in Rañó's three-dimensional reduction, with Hotton's constant forward drive and the chapter's rotational lag:
```math
\dot x = V\cos\phi, \qquad \dot y = V\sin\phi, \qquad \ddot\phi = -\frac{1}{\tau}\,\dot\phi - \frac{m}{\tau}\,\lVert\nabla S\rVert\,\sin\psi_t, \qquad \lVert\nabla S\rVert = e^{-c_d r}.
```
This is the chapter's heading equation after its three steps. The gradient term is expanded, ``\tfrac{\delta}{d}\nabla F\cdot\hat e_p = \tfrac{\delta}{d}F'(S)\,\lVert\nabla S\rVert\sin\psi_t``. The synapses are symmetric, ``F'(S) = f``. The morphological and synaptic factors are absorbed into the motor response strength ``m = \tfrac{\delta}{d}f``. The sign is that of Type B: the vehicle turns toward the stimulus.

**Warren's law.** Fajen & Warren's (2003) law for walking to a goal, at the same constant speed:
```math
\dot x = V\cos\phi, \qquad \dot y = V\sin\phi, \qquad \ddot\phi = -b\,\dot\phi - k_t\,\psi_t\,\bigl(e^{-c_1 d} + c_2\bigr).
```
In the chapter's control-law form ``\dot x = v``, ``\dot v = F(I) - bv - kx``, the position ``x`` is the heading ``\phi`` and ``v = \dot\phi`` is the turning rate. The information is ``I = \psi_t``, the forcing is ``F(I) = -k_t\,\psi_t\,(e^{-c_1 d} + c_2)``, and there is no spring on the heading itself (``k = 0``): the stiffness enters through ``F``.

**Variables**

| symbol | meaning |
|:--|:--|
| ``x, y`` | position (m); the goal is at the origin |
| ``\phi`` | heading (rad); the chapter's ``\theta_0`` |
| ``\dot\phi`` | turning rate (rad/s); the chapter's ``v`` in Warren's law |
| ``\theta`` | direction of the goal, seen from the agent |
| ``\psi_t = \phi - \theta`` | heading error (rad), the chapter's ``\theta_0 - \theta``, taken in ``[-\pi, \pi)`` |
| ``r``, ``d`` | distance to the goal (m), written ``r`` for the vehicle and ``d`` in Warren's law |
| ``s_{\mathrm l}, s_{\mathrm r}`` | left and right sensor readings |
| ``m_{\mathrm l}, m_{\mathrm r}`` | left and right motor outputs (wheel speeds, m/s) |

**Parameters** (`VEHICLE`, and `WARREN` from `AgentModels.jl`)

| symbol | value | what it sets | source |
|:--|:--|:--|:--|
| ``V`` | 1 m/s | forward speed of both agents | Warren's walking speed |
| ``\tau`` | 0.31 s | the vehicle's rotational lag | matched: ``1/b`` |
| ``m`` | 2.31 /s | the vehicle's motor response strength, ``\tfrac{\delta}{d}f`` | matched: ``k_t/b`` |
| ``c_d`` | 0.40 /m | decay of the stimulus gradient with distance | matched: ``c_1`` |
| ``f`` | 1 /s | synaptic weight, ``F(S) = fS``; ``f_{\mathrm l}, f_{\mathrm r}`` when the two crossed synapses differ | chosen |
| ``\delta`` | 1 m | sensor separation | illustrative: the match fixes only ``m`` |
| ``\delta/d`` | 2.31 | sensor separation over vehicle width ``d`` (0.43 m) | ``m/f`` |
| ``b`` | 3.25 /s | Warren's damping | fitted (Fajen & Warren 2003) |
| ``k_t`` | 7.5 /s² | Warren's stiffness toward the goal | fitted |
| ``c_1`` | 0.40 /m | decay of the stiffness with distance | fitted |
| ``c_2`` | 0.40 | stiffness that does not decay with distance | fitted; 0 in the mapping, the paths and the charts |

As in the chapter, ``d`` is the vehicle's width inside ``\delta/d`` and the goal distance in Warren's law.

**The mapping.** For a goal at a fixed direction and distance (``r = d``), with ``c_2 = 0`` and ``\sin\psi_t \approx \psi_t``, the vehicle's heading equation is Warren's with
```math
b = \frac{1}{\tau}, \qquad k_t = \frac{m}{\tau}, \qquad c_1 = c_d; \qquad\text{and back,}\quad \tau = \frac{1}{b}, \qquad m = \frac{k_t}{b}.
```
The vehicle above is the one matched to Warren's fit. Since both agents move at the same constant speed, the two models then differ in only two places, on the plane as at a fixed goal: ``\sin\psi_t`` against ``\psi_t``, and Warren's ``c_2``.

**Departures from the chapter text.**

- The chapter calls the stimulus Gaussian. A Gaussian ``S = e^{-r^2/2\sigma^2}`` has ``\lVert\nabla S\rVert = \tfrac{r}{\sigma^2}e^{-r^2/2\sigma^2}``. The factor ``e^{-c_d r}`` is the gradient of an exponential stimulus, ``S = e^{-c_d r}/c_d``, which is the one used here.
- The chapter writes ``e^{-c_d}``. Here the distance is written out, ``e^{-c_d r}``, so that ``c_d`` is a decay rate like Warren's ``c_1``.
- The step to ``\dot v = -bv - k_t\psi_t e^{-c_1 d}`` drops ``c_2``. The chapter attributes this to constant speed, but both agents already move at a constant speed: the step is the assumption ``c_2 = 0``.
"""

# ╔═╡ 10522db1-318c-58da-b436-4ccfb439aee7
"""
The chapter's vehicle matched to Fajen & Warren's (2003) fit: forward speed `V` (m/s), rotational lag
`τ` (s), motor response strength `m` (1/s) and stimulus-gradient decay `cd` (1/m). For the synapse chart:
the synaptic weight `f` (1/s), the sensor separation `δ` (m, illustrative) and `δd` = δ/d, with m = (δ/d) f.
"""
VEHICLE = let f = 1.0, m = WARREN.k / WARREN.b
    (V=WARREN.speed, τ=1 / WARREN.b, m, cd=WARREN.c1, f, δ=1.0, δd=m / f)
end

# ╔═╡ dc99a48e-03e9-5691-83b0-b6d17ba7de7d
"""
    chapter_vehicle(p = VEHICLE)

The chapter's Type B Braitenberg vehicle with a stimulus at the origin, state (x, y, φ, φ̇):
ẋ = V cos φ, ẏ = V sin φ, φ̈ = −φ̇/τ − (m/τ) e^{−c_d r} sin ψₜ, with r the distance to the stimulus
and ψₜ the heading error.
"""
function chapter_vehicle(p=VEHICLE)
    function f(u)
        x, y, φ, ω = u
        [p.V * cos(φ), p.V * sin(φ), ω, (-ω - p.m * exp(-p.cd * hypot(x, y)) * sin(heading_error(x, y, φ))) / p.τ]
    end
    BlackBoxContinuousSystem(f, 4)
end

# ╔═╡ a3a24058-e49f-42c4-82f9-83bea10f3182
md"""
## 1. Paths and two-parameter charts

### Introduction: what figure 1 shows

*New to these models? This introduction and the next one explain figure 1, and the code that makes it, in plain words. The rest of the section, from "Section 1 in detail" on, gives the details and checks every number it quotes. The four panels appear one by one further down this section, each in its own cell, and the whole figure is at the bottom of the notebook under **Figures**.*

**The question.** An agent, an animal or a robot, sees a goal and moves toward it. How does it steer? Figure 1 compares two answers, in the same two colours in every panel:

- **The Braitenberg vehicle** (orange) is a small robot with two sensors at its front and a motor on each side. Each sensor drives the motor on the *opposite* side: the wires cross. If the goal lies to the right, the right sensor reads a stronger signal, so the left motor runs faster than the right one, and the vehicle turns right, toward the goal.
- **Warren's law** (blue) is a formula that Fajen and Warren (2003) fitted to people walking to a goal: turn toward the goal, and turn harder when you point further away from it or when it is closer.

**The two equations.** Both agents move forward at the same constant speed, 1 m/s, so only their steering differs. The steering depends on the *heading error* ``\psi_t``: the angle between the direction the agent points, its heading ``\phi``, and the direction of the goal. ``\psi_t = 0`` means pointing straight at the goal. Write ``\dot\phi`` for the turning rate (how fast the heading changes, in rad/s) and ``\ddot\phi`` for how fast the turning rate changes. The two models are
```math
\text{vehicle:}\quad \ddot\phi = -\frac{1}{\tau}\,\dot\phi - \frac{m}{\tau}\,e^{-c_d r}\,\sin\psi_t, \qquad\qquad \text{Warren:}\quad \ddot\phi = -b\,\dot\phi - k_t\,e^{-c_1 d}\,\psi_t .
```
Both say the same thing: the heading is pulled toward the goal like a weight on a spring, and a friction slows its swing so that it settles.

- The first term is the friction, or *damping*. For the vehicle it is ``1/\tau``, where the lag ``\tau`` = 0.31 s is roughly the time its turning takes to catch up with what its motors command. For Warren it is ``b`` = 3.25 /s.
- The second term is the pull, or *stiffness*: the motor strength ``m`` (vehicle) or the stiffness ``k_t`` (Warren), times a factor that weakens the pull as the goal gets further away. Here ``r`` and ``d`` both stand for the distance to the goal, and the decay rates ``c_d`` and ``c_1``, both 0.4 /m, set how fast the pull fades. (Warren's full law adds a constant ``c_2`` to ``e^{-c_1 d}``. This notebook sets ``c_2 = 0``, so that the two models match.)
- The one real difference is the last factor. Warren's law uses ``\psi_t`` itself: a spring whose pull keeps growing the further the agent points away. The vehicle uses ``\sin\psi_t``: a pendulum whose lowest point is "pointing at the goal". It pulls hardest when the goal is 90° to the side, and hardly at all when the goal is straight behind.

The vehicle's numbers are chosen to match Warren's fit: ``\tau = 1/b`` and ``m = k_t/b``. For small heading errors ``\sin\psi_t \approx \psi_t``, and the two equations are then the same.

**Top row (1A, 1B): paths.** Both agents start 4 m to the left of the goal (the star), pointing 5°, 15° or 25° above or below it, the range of heading errors in walking experiments. The computer *simulates* each agent: it works out from the equations where the agent goes, and draws its path. The two models' paths nearly coincide, at most 1.1 cm apart, because these heading errors are small.

**Bottom row (1C, 1D): parameter charts.** A model's parameters are its knobs. A *parameter chart* is a map of two knobs: each point is one setting of them, coloured by what the agent does with that setting. To keep the chart simple, the goal is held still, 2 m away in a fixed direction, and the chart asks only about the heading: where does it come to rest?

- **approach** (green): the heading settles pointing at the goal;
- **approach (overshoot)** (pale yellow): it swings past the goal's direction before it settles on it;
- **avoid** (pink): it settles pointing *away* from the goal;
- **wander** (lavender): it never settles; the vehicle keeps circling;
- **approach or wander**, **avoid or wander** (violet, 1C only): which of the two happens depends on how the agent starts.

1C's two knobs are the vehicle's crossed wires, or synapses: ``f_{\mathrm l}``, from the left sensor to the right motor, and ``f_{\mathrm r}``, from the right sensor to the left motor. 1D's two knobs are Warren's damping ``b`` and stiffness ``k_t``. The orange dot in 1C is the vehicle of 1A, and the blue dot in 1D is the law of 1B. Both lie in approach (overshoot).

Why the regions are where they are:

- *Wander in 1C.* Unequal wires give the vehicle a built-in turn, like a car whose wheels are out of line. Where the pull toward the goal can match the built-in turn, the heading comes to rest a little to one side of the goal. Where the built-in turn is stronger than the pull can ever be, the vehicle circles for ever. The solid black lines are where the two are just equal. Beyond the black dots the wires are so strong that a spinning vehicle can keep spinning even where it could also come to rest: those are the violet bands.
- *Avoid.* With both wires negative (lower left of 1C), or a negative stiffness ``k_t`` (1D), the "spring" pushes the heading away from the goal instead of pulling it in.
- *Wander in 1D.* A negative damping ``b`` is a friction that feeds the swing instead of slowing it, so each swing grows.

The lines between the regions say how the behaviour changes there (legend under figure 1, at the bottom of the notebook). A **bifurcation** (black) is a place where the kind of behaviour changes: here, a resting heading appears or disappears, or the circling starts. A white line is a *degenerate* bifurcation, a special case that occurs only because nothing steers on it or because the equation is exactly linear. Grey dots mark a border that is not a bifurcation: across it the heading starts to overshoot, but it still settles on the same heading.
"""

# ╔═╡ 9ce34595-0853-44fe-a6fb-c4a05547d278
md"""
### Introduction: how the code makes figure 1

**How a model is written.** Each model is a function that takes the state and returns how fast each part of it changes. Here is the vehicle's, from `chapter_vehicle` in the section above:
```julia
x, y, φ, ω = u    # the state: position, heading, turning rate
[p.V * cos(φ), p.V * sin(φ), ω, (-ω - p.m * exp(-p.cd * hypot(x, y)) * sin(heading_error(x, y, φ))) / p.τ]
```
The four entries are ``\dot x = V\cos\phi`` and ``\dot y = V\sin\phi`` (it moves at speed ``V`` = 1 m/s along its heading), ``\dot\phi = \omega`` (``\omega`` is the turning rate), and the vehicle's equation above for ``\dot\omega = \ddot\phi``. The goal is at the origin, so `hypot(x, y)` is the distance ``r`` to it, and `heading_error` (from `AgentModels.jl`) computes ``\psi_t``. `BlackBoxContinuousSystem(f, 4)` (from MathematicalSystems.jl) wraps the function as a system whose state is four numbers. Warren's law, `warren_approach` in `AgentModels.jl`, is written the same way.

**How a model is simulated.** `IVP(system, start)` pairs a system with its starting state, and `trajectory(IVP(...), (0.0, 30.0))` (from `BehaviorTools.jl`) runs it from time 0 to 30 s. It hands the equations to an ODE solver (`Tsit5`), which steps the state forward in short time steps, each chosen so that the error stays small. The result `sol` holds the times `sol.t` and the states `sol.u`. The option `callback=arrived` stops a run once the agent is within 10 cm of the goal.

**How the charts are made.** For a goal held still, the heading on its own is a smaller system of two numbers, ``\psi_t`` and ``\dot\psi_t``. Most borders in the charts follow from formulas for this system: where it can come to rest, and whether it swings about that rest. For 1D the formulas are short, and `panel_warren_chart` fills the regions with them directly. For 1C, `braitenberg_region` gives the region of every point. Only the dashed curves have no formula, and the code finds them by simulating (`circling_coexists`). A check cell then simulates the heading at 1024 points spread over 1C and compares the region it finds with `braitenberg_region` at each one.

**Where things are.** The cells of this section, in order:

| name | what it is |
|:--|:--|
| `VEHICLE` (above), `WARREN` (from `AgentModels.jl`) | the parameter values of the two models |
| `chapter_vehicle` (above), `warren_approach` (from `AgentModels.jl`) | the equations of the two models |
| `START`, `ERRORS` | where the paths start, and the six starting heading errors |
| `WARREN0` | Warren's law with ``c_2 = 0`` |
| `arrived` | stops a run within 10 cm of the goal |
| `vehicle_runs`, `warren_runs` | the six simulated runs of each model, drawn in 1A and 1B |
| `CHART` | the goal of the charts, held still 2 m away |
| `crossed_heading` | the vehicle's heading equation with its two wires set separately: the system behind 1C |
| `heading_systems` | the heading equations of both models with equal wires, used in figure 2 |
| `circling_coexists`, `homoclinic_γ`, `HOMOCLINIC`, `γ_homoclinic` | the dashed curves of 1C, found by simulation |
| `braitenberg_region` | the region of a point of 1C |
| `overshoot_hyperbola` | the dotted curve of 1C, where overshoot starts |
| `chart_label`, `paths_panel`, `PATHS_YLIMS` | a label, one set of paths drawn, and the y range that 1A and 1B share |
| `FILLS`, `REGIONS`, `LINE_KINDS`, `POINT_KINDS` | the colours of the regions and the styles of the lines and points |
| `evenly_spaced`, `dotted!`, `fold_angles`, `homoclinic_curve` | small helpers: the grey dots, and the lines of 1C |
| `synapse_regions!`, `synapse_bifurcations!` | fill the regions of 1C and draw its lines |
| `panel_vehicle_paths`, `panel_warren_paths`, `panel_synapse_chart`, `panel_warren_chart` | the panels 1A, 1B, 1C and 1D, each shown in its own cell |
| `figure_paths_charts` (under **Figures**, at the bottom) | puts the four panels and the legend together, letters them and saves the figure to `Figures/` |

The cells whose output is a table of numbers are checks: each recomputes the numbers quoted in the text just above it. A panel's settings (its limits, or the point it marks) are in the first line of its cell, and Pluto redraws the panel and the figure when one is changed.
"""

# ╔═╡ 35f60d3f-7c71-5d66-839c-b879c7bb0d28
md"""
### Section 1 in detail

Figure 1, `Approach1`: the paths of the vehicle (1A) and of Warren's walker (1B), each above its own parameter chart (1C, 1D). Besides its titles and axes, the figure carries only the names of the regions; this section is its key.

**The paths.** Both agents start 4 m to the left of the goal, at heading errors of ±5°, ±15° and ±25°, within the experimental range, and not yet turning. Both move at 1 m/s and stop within 10 cm of the goal. Warren's law is run with ``c_2 = 0``, as in the mapping and in its chart, so the two models differ only in ``\sin\psi_t`` against ``\psi_t``.
"""

# ╔═╡ 8dbe941e-d8a4-50af-a57f-b2d4a4e90de5
"Start of the paths (m): 4 m to the left of the goal, which is at the origin."
START = [-4.0, 0.0]

# ╔═╡ 2880201f-796c-5eed-af72-c8aa7a133e34
"Initial heading errors of the paths (°), within the experimental range ±25°; the goal is due right."
ERRORS = [-25, -15, -5, 5, 15, 25]

# ╔═╡ b7ea2177-2a75-59f6-a442-762a9c6533e8
"Warren's law with c₂ = 0, as in the mapping and the charts."
WARREN0 = merge(WARREN, (; c2=0.0))

# ╔═╡ 4ae435b1-648f-5e6c-9bb2-0607108bca23
"Ends a run within 10 cm of the goal."
arrived = ContinuousCallback((u, t, i) -> hypot(u[1], u[2]) - 0.1, terminate!);

# ╔═╡ 770c1295-6ded-559a-93d8-8384394db36d
begin
    # The paths of figure 1: each model from START at each heading error of ERRORS, not yet turning, until it is
    # within 10 cm of the goal (at most 30 s).
    warren_runs = [trajectory(IVP(warren_approach(WARREN0), [START; deg2rad(e); 0.0]), (0.0, 30.0);
        callback=arrived, saveat=0.01) for e in ERRORS]
    vehicle_runs = [trajectory(IVP(chapter_vehicle(VEHICLE), [START; deg2rad(e); 0.0]), (0.0, 30.0);
        callback=arrived, saveat=0.01) for e in ERRORS]
    nothing
end

# ╔═╡ 44364c74-3b97-5ad5-8128-6f78c7f4d3ba
md"""
**How close the paths are.** Within the experimental range the difference between ``\sin\psi_t`` and ``\psi_t`` is small, and the paths nearly coincide. Both reach the goal in 3.91–4.07 s, with the same times to 0.01 s, and swing out to the side by about 0.1, 0.3 and 0.5 m from ±5°, ±15° and ±25°. The vehicle's paths lie at most 1.1 cm from the walker's, from ±25°. At the start both steer toward the turning rate 0.47 ``\psi_t`` rad/s, the walker's ``\tfrac{k_t}{b}e^{-c_1 d}\psi_t`` and the vehicle's ``m e^{-c_d r}\sin\psi_t``, which at 25° is 3 % smaller, so the vehicle swings out 8 mm further. The next cell checks these numbers.
"""

# ╔═╡ 6a696545-2769-5ef6-b90a-6b9e550b8c81
let
    # Arrival times from the starts of figure 1 (missing: not within 10 cm after 30 s), the widest swing to the side,
    # and the largest gap between the two models' paths.
    arrival(sol) = hypot(sol.u[end][1:2]...) < 0.11 ? round(sol.t[end]; digits=2) : missing
    swing(sol) = round(maximum(abs(u[2]) for u in sol.u); digits=3)
    # both move at 1 m/s, so equal times are equal distances along the path
    gap(w, v) = round(100maximum(hypot((w(t)[1:2] .- v(t)[1:2])...) for t in 0:0.01:min(w.t[end], v.t[end])); digits=2)
    table = [(heading_error=e, warren_arrives_s=arrival(w), vehicle_arrives_s=arrival(v), warren_swing_m=swing(w), vehicle_swing_m=swing(v),
              path_gap_cm=gap(w, v)) for (e, w, v) in zip(ERRORS, warren_runs, vehicle_runs)]
    # The turning rates steered toward at 4 m, per unit heading error.
    r = 4.0
    gains_at_start = (walker=WARREN0.k / WARREN0.b * (exp(-WARREN0.c1 * r) + WARREN0.c2), vehicle=VEHICLE.m * exp(-VEHICLE.cd * r))
    (; table, gains_at_start)
end

# ╔═╡ 39d6d830-ea5e-50d2-ba2f-9434b1050c2e
md"""
**Outside the experimental range** (not drawn) the difference becomes qualitative. From 4 m the vehicle reaches the goal only for heading errors up to 125°, and escapes beyond; the walker reaches it from 150° and from 179°. At 150° the vehicle steers toward 0.23 rad/s and the walker toward 1.22 rad/s, since ``\psi_t`` is 2.6 there, five times ``\sin\psi_t``. The vehicle turns so slowly that it moves away, the gradient weakens, and it turns ever more slowly. The next cell checks these numbers.
"""

# ╔═╡ bbdccdbc-eb10-5625-a7da-0ced29a10b93
let
    # 1. The turning rates steered toward at 4 m and 150°.
    ψ, r = deg2rad(150), 4.0
    vehicle_rate = VEHICLE.m * exp(-VEHICLE.cd * r) * sin(ψ)
    walker_rate = WARREN0.k / WARREN0.b * ψ * (exp(-WARREN0.c1 * r) + WARREN0.c2)
    # 2. From 4 m: the largest heading error from which the vehicle reaches the goal (bisection), and what it does
    #    beyond (leaves past 30 m: escapes).
    far = ContinuousCallback((u, t, i) -> hypot(u[1], u[2]) - 30.0, terminate!)
    function outcome(r0, e; sys=chapter_vehicle(VEHICLE))
        sol = trajectory(IVP(sys, [-r0, 0.0, deg2rad(e), 0.0]), (0.0, 400.0);
            callback=CallbackSet(arrived, far), save_everystep=false)
        d = hypot(sol.u[end][1:2]...)
        d < 0.11 ? :reaches : d > 29 ? :escapes : :stays
    end
    lo, hi = 90.0, 180.0
    for _ in 1:25
        mid = (lo + hi) / 2
        outcome(4.0, mid) == :reaches ? (lo = mid) : (hi = mid)
    end
    (; vehicle_rate_at_4m_150deg=vehicle_rate, walker_rate_at_4m_150deg=walker_rate,
        largest_error_reaching_from_4m=lo, beyond=[outcome(4.0, e) for e in (lo + 1, 150, 180)],
        walker_from_4m=[outcome(4.0, e; sys=warren_approach(WARREN0)) for e in (150, 179)])
end

# ╔═╡ 72502e6f-6abe-51f4-a2d5-71a3447e1360
md"""
**The heading at a fixed goal.** The parameter charts, 1C and 1D, are charts of the two heading equations over two of their parameters, for a goal held at a fixed direction 2 m away (`CHART`). Each chart is about the heading error alone: where it settles, and whether it settles at all. The speed does not enter, so these are the charts of `ApproachBehavior.jl` §4 in the chapter's notation, 1C extended to ±8 /s to show its homoclinic curves.

*The vehicle's two synapses.* A crossed vehicle has a synapse ``f_{\mathrm l}`` from its left sensor to its right motor and ``f_{\mathrm r}`` from its right sensor to its left motor. Its sensors read ``s_{\mathrm{l,r}} = S \mp \tfrac{\delta}{2}\lVert\nabla S\rVert\sin\psi_t``, Rañó's first-order expansion of the stimulus across the sensors. Its motors run at ``m_{\mathrm r} = f_{\mathrm l}\,s_{\mathrm l}`` and ``m_{\mathrm l} = f_{\mathrm r}\,s_{\mathrm r}``, which command the turning rate ``(m_{\mathrm r} - m_{\mathrm l})/d``. With the mean synapse ``\bar f = (f_{\mathrm l} + f_{\mathrm r})/2`` and ``S = e^{-c_d r}/c_d``,
```math
\tau\,\ddot\psi_t = -\dot\psi_t + \frac{\delta}{d}\,e^{-c_d r}\left(\frac{f_{\mathrm l} - f_{\mathrm r}}{\delta c_d} - \bar f\sin\psi_t\right).
```
With symmetric synapses, ``f_{\mathrm l} = f_{\mathrm r} = f``, this is the vehicle's heading equation (checked below). It is a damped pendulum driven by a constant torque (Strogatz 1994, §8.5), and the place across the wedge is ``\gamma = (f_{\mathrm l} - f_{\mathrm r})/(\delta c_d \bar f)``, ``\pm 1`` on its edges.

*Warren's law.* At ``d`` = 2 m with ``c_2 = 0`` its heading equation, ``\ddot\psi_t = -b\dot\psi_t - k_t e^{-c_1 d}\psi_t``, is linear.

*Where the heading settles.* The regions of the synapse chart follow from the pendulum's rest points and their eigenvalues (`braitenberg_region`), apart from the homoclinic curves, which have no closed form. `HOMOCLINIC` finds them by shooting the saddle's unstable branch (`circling_coexists`). The checks below run the heading equation over the whole chart and compare.
"""

# ╔═╡ 9f305944-5da7-5429-863e-5f4746778e61
"""
The goal of the heading equations and charts, held at a fixed direction: distance `d` (m, the vehicle's
r), the factor `G` = e^{−c₁d} = e^{−c_d r} (Warren's stiffness factor and the vehicle's gradient), the
synapse chart's `a` = δc_d/2 and overshoot vertex `W` = 1/(4τ(δ/d)G), the synapses drawn, and the fitted
law's damping ratio there (c₂ = 0).
"""
CHART = let d = 2.0, p = VEHICLE
    G = exp(-p.cd * d)
    (; d, G, τ=p.τ, a=p.δ * p.cd / 2, W=1 / (4p.τ * p.δd * G), synapses=[-1.5, -1.0, -0.5, 0.5, 1.0, 1.5],
        damping_ratio=WARREN.b / (2sqrt(WARREN.k * G)))
end

# ╔═╡ 80f14db7-48f4-57ac-8afb-b85a52b8cf65
"""
Heading equation of the crossed vehicle with synapses `fl` (left sensor → right motor) and `fr` (right
sensor → left motor), for a goal held at a fixed direction where the gradient is `G`, state (ψₜ, ψ̇ₜ):
τ ψ̈ₜ = −ψ̇ₜ + (δ/d) G ((f_l − f_r)/(δ c_d) − f̄ sin ψₜ), f̄ = (f_l + f_r)/2.
"""
crossed_heading(fl, fr; G=CHART.G, τ=CHART.τ, p=VEHICLE) =
    BlackBoxContinuousSystem(u -> [u[2], (p.δd * G * ((fl - fr) / (p.δ * p.cd) - (fl + fr) / 2 * sin(u[1])) - u[2]) / τ], 2)

# ╔═╡ b33547d7-6138-5949-b72c-be68ce25a99b
"""Heading-only systems for the goal of `CHART`, state (ψₜ, ψ̇ₜ): Warren's law (c₂ = 0) and the vehicle."""
heading_systems(; G=CHART.G, p=VEHICLE) = (warren=BlackBoxContinuousSystem(u -> [u[2], -WARREN.b * u[2] - WARREN.k * G * u[1]], 2),
    vehicle=BlackBoxContinuousSystem(u -> [u[2], -(u[2] + p.m * G * sin(u[1])) / p.τ], 2))

# ╔═╡ 9e8ed77a-f7f6-59a2-ad6f-7acc42bf980d
"""
Whether the vehicle at the goal of `CHART`, with mean synapse `fbar` > 0 and `γ` = sin ψₜ* in (0, 1) (the place
across the wedge, f_l − f_r = 2a γ f̄), can circle although its heading has a stable rest point. Its heading
equation is ψ̈ = K(γ − sin ψ) − ψ̇/τ with K = (δ/d) G f̄/τ. The forward branch of the saddle's unstable manifold
either falls back to the stable heading (false) or passes the next saddle, ψ_s + 2π (true): then a circling
orbit exists as well.
"""
function circling_coexists(fbar, γ; G=CHART.G, τ=CHART.τ, p=VEHICLE)
    K = p.δd * G * fbar / τ
    ψs = π - asin(γ)                                   # the saddle
    λ = (-1 / τ + sqrt(1 / τ^2 - 4K * cos(ψs))) / 2    # its unstable eigenvalue, eigenvector (1, λ)
    sys = BlackBoxContinuousSystem(u -> [u[2], K * (γ - sin(u[1])) - u[2] / τ], 2)
    stop = CallbackSet(ContinuousCallback((u, t, i) -> u[2], terminate!), ContinuousCallback((u, t, i) -> u[1] - ψs - 2π, terminate!))
    sol = trajectory(IVP(sys, [ψs + 1e-8, 1e-8λ]), (0.0, 1e7); callback=stop, save_everystep=false, abstol=1e-13, reltol=1e-12)
    sol.u[end][1] > ψs + 2π - 1e-6
end

# ╔═╡ 4893a1d8-4517-550f-8166-2deffee30e9a
"""γ on the homoclinic curve at the mean synapse `fbar` (bisection); 1 where there is none."""
function homoclinic_γ(fbar; n=24)
    circling_coexists(fbar, 1 - 1e-7) || return 1.0
    lo, hi = 0.0, 1 - 1e-7
    for _ in 1:n
        mid = (lo + hi) / 2
        circling_coexists(fbar, mid) ? (hi = mid) : (lo = mid)
    end
    hi
end

# ╔═╡ 2c8a6a9d-fa54-5958-99d6-74c2d50a40a8
"""
The homoclinic curves of the synapse chart: the mean synapse `fc` where they leave the saddle-node lines (below
it no circling orbit coexists with a rest point), and γ along them for f̄ from `fc` to `fmax`.
"""
HOMOCLINIC = let fmax = 10.0
    lo, hi = 1.0, 4.0
    for _ in 1:40
        mid = (lo + hi) / 2
        circling_coexists(mid, 1 - 1e-7) ? (hi = mid) : (lo = mid)
    end
    fbar = hi .+ (fmax - hi) .* range(0, 1; length=50) .^ 2   # denser near f̄_c, where the curves leave the lines
    (; fc=hi, fbar=collect(fbar), γ=homoclinic_γ.(fbar))
end

# ╔═╡ 99c8ca49-7c74-57e8-a892-b03a01f2a6e4
"""γ on the homoclinic curve at |`fbar`|, interpolated in `HOMOCLINIC`; 1 below f̄_c."""
function γ_homoclinic(fbar)
    f = abs(fbar)
    f <= HOMOCLINIC.fc && return 1.0
    fs, γs = HOMOCLINIC.fbar, HOMOCLINIC.γ
    i = clamp(searchsortedlast(fs, f), 1, length(fs) - 1)
    t = (f - fs[i]) / (fs[i+1] - fs[i])
    (1 - t) * γs[i] + t * γs[i+1]
end

# ╔═╡ 80770646-8cbf-5260-82eb-fc0d6a34331c
"""
Region of the synapse chart at `fl`, `fr`: `:approach`, `:overshoot` (approach with overshoot, inside the
hyperbola f̄² − ((f_l − f_r)/2a)² = W²), `:away`, `:turning` (circling), or, between a saddle-node line and its
homoclinic curve, `:approach_or_turning` and `:away_or_turning` (which one depends on the start).
"""
function braitenberg_region(fl, fr; a=CHART.a, W=CHART.W)
    fbar, Δ = (fl + fr) / 2, fl - fr
    abs(Δ) > a * abs(fl + fr) && return :turning
    fbar != 0 && abs(Δ) / (2a * abs(fbar)) > γ_homoclinic(fbar) && return fbar < 0 ? :away_or_turning : :approach_or_turning
    fbar < 0 && return :away
    fbar^2 - (Δ / 2a)^2 > W^2 ? :overshoot : :approach
end

# ╔═╡ da884d59-72ca-51be-b42b-1726187bda8b
"""
The overshoot hyperbola f̄² − ((f_l − f_r)/2a)² = W² inside the square [−lim, lim]², on its approach branch
(`side` = 1) or turn-away branch (`side` = −1): `n` points (f_l, f_r), none if it misses the square.
"""
function overshoot_hyperbola(W; a=CHART.a, side=1, lim=2.0, n=121)
    W >= lim && return Tuple{Float64,Float64}[]
    T = log((lim / W + sqrt((lim / W)^2 - (1 - a^2))) / (1 + a))
    [(side * W * (cosh(t) + a * sinh(t)), side * W * (cosh(t) - a * sinh(t))) for t in range(-T, T; length=n)]
end

# ╔═╡ bc8d7e9d-51a6-57ae-b8c8-a1aeee25df1a
let
    # 1. The turning rate from sensors and motors against the chart's formula: s_{l,r} = S ∓ (δ/2)‖∇S‖ sin ψₜ with
    #    S = e^{−c_d r}/c_d, m_r = f_l s_l, m_l = f_r s_r, turning rate (m_r − m_l)/d with d = δ/(δ/d).
    p = VEHICLE
    motors_vs_formula = maximum(1:1000) do _
        fl, fr, ψ, r = 4rand() - 2, 4rand() - 2, 2π * rand() - π, 6rand()
        G = exp(-p.cd * r)
        sl, sr = G / p.cd - p.δ / 2 * G * sin(ψ), G / p.cd + p.δ / 2 * G * sin(ψ)
        abs((fl * sl - fr * sr) / (p.δ / p.δd) - p.δd * G * ((fl - fr) / (p.δ * p.cd) - (fl + fr) / 2 * sin(ψ)))
    end
    # 2. Symmetric synapses give the vehicle: φ̈ at distance r, heading error ψ and turning rate ω, goal due right.
    r, ψ, ω = 3.0, 0.7, 0.2
    symmetric = maximum(abs(vector_field(crossed_heading(f, f; G=exp(-p.cd * r)), [ψ, ω])[2] -
                            vector_field(chapter_vehicle(merge(p, (; m=p.δd * f))), [-r, 0.0, ψ, ω])[4]) for f in CHART.synapses)
    (; motors_vs_formula, symmetric_vs_vehicle=symmetric, wheel_base_m=p.δ / p.δd)
end

# ╔═╡ 0b65fae3-4a61-5c1d-be00-da756ec8278c
let
    # The analytic regions against runs of the heading equation over the whole chart (±8): from rest at four headings
    # (does any come to rest, and where?) and from a fast spin (does it circle?). Both: the outcome depends on the
    # start. Overshoot from the eigenvalues at the heading the runs settle to (complex: overshoot).
    "The heading `sys` comes to rest at from `u0`, or `nothing` if it is still circling after 200 s."
    function rests_at(sys, u0)
        u = trajectory(IVP(sys, u0), (0.0, 200.0); save_everystep=false).u[end]
        v = trajectory(IVP(sys, u), (0.0, 50.0); save_everystep=false).u[end]
        abs(v[1] - u[1]) > 2π || abs(v[2]) > 1e-4 ? nothing : v[1]
    end
    function run_region(fl, fr)
        sys = crossed_heading(fl, fr)
        rests = filter(!isnothing, [rests_at(sys, [ψ0, 0.0]) for ψ0 in (-3π / 4, -π / 4, π / 4, 3π / 4)])
        isempty(rests) && return :turning
        circles = isnothing(rests_at(sys, [0.0, 60.0 * sign(fl - fr)]))
        ψT = first(rests)
        cos(ψT) < 0 && return circles ? :away_or_turning : :away
        circles && return :approach_or_turning
        h = 1e-6
        J = reduce(hcat, [(vector_field(sys, [ψT, 0.0] .+ h .* e) .- vector_field(sys, [ψT, 0.0] .- h .* e)) ./ 2h for e in ([1.0, 0.0], [0.0, 1.0])])
        isreal(eigvals(J)) ? :approach : :overshoot
    end
    fs = range(-7.9, 7.9; length=32)
    cells = vec([(fl, fr) for fl in fs, fr in fs])
    simulated = Vector{Symbol}(undef, length(cells))
    Threads.@threads for i in eachindex(cells)
        simulated[i] = run_region(cells[i]...)
    end
    disagreements = count(simulated .!= [braitenberg_region(c...) for c in cells])
    # The homoclinic curves: 1 % of the band's width inside them both outcomes occur, 1 % outside only rest.
    across_curve = map((3.0, 4.0, 6.0)) do fbar
        γh = γ_homoclinic(fbar)
        at(γ) = (fbar * (1 + CHART.a * γ), fbar * (1 - CHART.a * γ))
        (; fbar, γ_homoclinic=round(γh; digits=4), inside=run_region(at(γh + 0.01(1 - γh))...), outside=run_region(at(γh - 0.01(1 - γh))...))
    end
    (; grid_cells=length(cells), disagreements, regions_seen=sort(unique(simulated)), across_curve, fc=HOMOCLINIC.fc, W=CHART.W, a=CHART.a)
end

# ╔═╡ 691d3375-d17d-5173-afaa-9c4c62a9bf84
"A label in a chart: `s` in `size` points, in the secondary ink unless given a `color`."
chart_label(s; size=8, halign=:center, rotation=0, color=INK_SECONDARY) = text(s, size, color, halign; rotation)

# ╔═╡ ece996ab-edcd-5da7-828f-763eb45ad8f2
"""Paths `runs` from `START` with the heading errors `ERRORS`, in `color`; the y axis is stretched, not equal to x."""
function paths_panel(runs, title, color; xlims=(-4.3, 0.3), ylims=(-1.0, 1.0))
    p = plot(; xlims, ylims, xlabel="x (m)", ylabel="y (m)", title, legend=false)
    for sol in runs
        plot!(p, first.(sol.u), getindex.(sol.u, 2); color, linewidth=1.5)
    end
    scatter!(p, START[1:1], START[2:2]; color=INK, markersize=3, markerstrokewidth=0)
    scatter!(p, [0.0], [0.0]; marker=:star5, markersize=8, color=INK, markerstrokewidth=0)
    p
end

# ╔═╡ 7a516e0e-b6d4-52af-9fa7-115c16ab2831
"One y scale for both path panels, from the widest swing of either model."
PATHS_YLIMS = let swing = maximum(abs(u[2]) for runs in (warren_runs, vehicle_runs) for sol in runs for u in sol.u)
    (-1, 1) .* ceil(1.2swing; digits=1)
end

# ╔═╡ 37d35c20-1991-50ee-ac9d-b4cf14a7929f
"""
Grey `away`: the pale ground of the legends, against which white (degenerate) lines show.
"""
FILLS = (; away="#e1e0d9")

# ╔═╡ 70b6b969-e8ed-5f2e-b06d-bc261de05201
"""
Pastel colours of the regions of both parameter charts: the heading settles toward the goal without overshoot,
with overshoot, settles pointing away, or never settles. Tints of the categorical palette chosen so that
regions that share a border stay apart under colour-vision deficiency (worst bordering pair CVD ΔE 10.2).
`either` fills the bands where the outcome depends on the start (the heading settles or circles); against
every neighbour it is at least ΔE 19 apart, for normal and colour-deficient vision.
"""
REGIONS = (approach="#a6d4a6", overshoot="#fcf1d9", away="#fcebf1", never="#c8c2f1", either="#8f82dc")

# ╔═╡ 594c138b-1002-5647-b497-ea04252e321f
"""
Line styles of the charts. The colour says whether a line is a bifurcation: black, a bifurcation; white, a
degenerate one; small grey dots (`spacing` of the plot apart, drawn by `dotted!`), not a bifurcation. The dash
says which kind, in black and white alike: solid, a fold (an eigenvalue passes through zero); dashed, a
homoclinic bifurcation; dash-dot, a Hopf bifurcation (a pair of eigenvalues crosses the imaginary axis).
"""
LINE_KINDS = (fold=(color=INK, linewidth=1.6), homoclinic=(color=INK, linewidth=1.6, linestyle=:dash),
    degenerate_fold=(color=:white, linewidth=2.4), degenerate_hopf=(color=:white, linewidth=2.4, linestyle=:dashdot),
    not_bifurcation=(color=INK_MUTED, markersize=1.8, spacing=0.016))

# ╔═╡ b291984d-1792-50e4-81ff-a5718666219d
"Points of the charts: where a homoclinic curve leaves its fold line (black), and where nothing steers (white)."
POINT_KINDS = (fold_meets_homoclinic=(color=INK, markersize=4.5, markerstrokewidth=0), degenerate=(color=:white, markersize=5, markerstrokewidth=0))

# ╔═╡ 3b0b14d2-82d3-50b8-933a-9caf1f1bae18
"""
Points spaced `spacing` apart along the polyline `pts` (tuples), with arc length measured after dividing each
coordinate by `scale` (the axis lengths, so that the spacing is a fraction of the plot).
"""
function evenly_spaced(pts, scale, spacing)
    u = [a ./ scale for a in pts]
    s = [0.0; cumsum([hypot((u[i+1] .- u[i])...) for i in 1:length(u)-1])]
    map(range(0, s[end]; step=spacing)) do t
        i = clamp(searchsortedlast(s, t), 1, length(s) - 1)
        w = (t - s[i]) / max(s[i+1] - s[i], eps())
        pts[i] .+ w .* (pts[i+1] .- pts[i])
    end
end

# ╔═╡ d5d4755f-fd2a-5e54-9c28-2faa81e19530
"""
Draw a line that is not a bifurcation, the curve `xs`, `ys`, as grey dots. `scale` is the plot's (x, y) extent
in data units per plot width, so the dots are evenly spaced on the page.
"""
function dotted!(p, xs, ys; scale)
    q = evenly_spaced(collect(zip(xs, ys)), scale, LINE_KINDS.not_bifurcation.spacing)
    scatter!(p, first.(q), last.(q); color=LINE_KINDS.not_bifurcation.color, markersize=LINE_KINDS.not_bifurcation.markersize, markerstrokewidth=0)
end

# ╔═╡ 5d569900-0d01-5d93-bb40-f2f073499b81
"The angles of the synapse chart's saddle-node lines f_r = f_l (1 ∓ a)/(1 ± a), a = δc_d/2."
fold_angles(a=CHART.a) = (atan((1 - a) / (1 + a)), atan((1 + a) / (1 - a)))

# ╔═╡ 06841119-64a5-55da-b33d-11ad72ce1976
"""
A curve of `HOMOCLINIC` in the synapse chart, as points (f_l, f_r) at the place `γs` across the wedge: in the
approach (`s` = 1) or avoid (`s` = −1) wedge, below (`side` = 1) or above (`side` = −1) the diagonal. With all
`γs` = 1 it is the saddle-node line beside it.
"""
homoclinic_curve(s, side, γs; a=CHART.a, H=HOMOCLINIC) = [(s * f * (1 + side * a * γ), s * f * (1 - side * a * γ)) for (f, γ) in zip(H.fbar, γs)]

# ╔═╡ 8a503b49-54f0-555e-b754-96c9e5a78b21
"""
Fill the regions of the synapse chart of half-width `lim`: the approach and avoid wedges between the saddle-node
lines and wander outside them, approach with overshoot inside its hyperbola, and the bands between each saddle-node
line and its homoclinic curve, where the outcome depends on the start.
"""
function synapse_regions!(p; lim)
    sector(θ1, θ2; R=3lim) = Shape([0.0; R .* cos.(range(θ1, θ2; length=60))], [0.0; R .* sin.(range(θ1, θ2; length=60))])
    θlo, θhi = fold_angles()
    plot!(p, sector(θlo, θhi); fillcolor=REGIONS.approach, linewidth=0)
    plot!(p, sector(θhi, θlo + π); fillcolor=REGIONS.never, linewidth=0)
    plot!(p, sector(θlo + π, θhi + π); fillcolor=REGIONS.away, linewidth=0)
    plot!(p, sector(θhi + π, θlo + 2π); fillcolor=REGIONS.never, linewidth=0)
    poly = [overshoot_hyperbola(CHART.W; lim); [(lim, lim)]]
    plot!(p, Shape(first.(poly), last.(poly)); fillcolor=REGIONS.overshoot, linewidth=0)
    n = length(HOMOCLINIC.fbar)
    for s in (1, -1), side in (1, -1)
        band = [homoclinic_curve(s, side, ones(n)); reverse(homoclinic_curve(s, side, HOMOCLINIC.γ))]
        plot!(p, Shape(first.(band), last.(band)); fillcolor=REGIONS.either, linewidth=0)
    end
    p
end

# ╔═╡ 7b0fa879-d51c-51f5-b201-296f4a5b393f
"""
Draw the lines and points of the synapse chart of half-width `lim` in the styles of `LINE_KINDS` and `POINT_KINDS`:
the overshoot hyperbola (not a bifurcation), the homoclinic curves, the saddle-node lines, the points where the
curves leave the lines, and the origin, where nothing steers (degenerate).
"""
function synapse_bifurcations!(p; lim)
    branch = overshoot_hyperbola(CHART.W; lim)
    dotted!(p, first.(branch), last.(branch); scale=(2lim, 2lim))
    for s in (1, -1), side in (1, -1)
        homo = homoclinic_curve(s, side, HOMOCLINIC.γ)
        plot!(p, first.(homo), last.(homo); LINE_KINDS.homoclinic...)
    end
    for θ in fold_angles()
        plot!(p, [-2lim * cos(θ), 2lim * cos(θ)], [-2lim * sin(θ), 2lim * sin(θ)]; LINE_KINDS.fold...)
    end
    a, fc = CHART.a, HOMOCLINIC.fc
    meets = [(s * fc * (1 + side * a), s * fc * (1 - side * a)) for s in (1, -1) for side in (1, -1)]
    scatter!(p, first.(meets), last.(meets); POINT_KINDS.fold_meets_homoclinic...)
    scatter!(p, [0.0], [0.0]; POINT_KINDS.degenerate...)
end

# ╔═╡ 0154da58-76d4-57b0-af61-8fd3a841fff5
md"""
**1A. The vehicle's paths.** The vehicle (orange) from the start (the black dot), 4 m to the left of the goal (the star), at the six heading errors. The y axis is stretched against the x axis to show the small swings to the side; 1A and 1B share its scale.
"""

# ╔═╡ a4997fff-8e99-5c16-beeb-e34aaed3886f
"""The vehicle's paths from `START` (1A)."""
panel_vehicle_paths = let title = "Braitenberg (Type B, V = 1 m/s, τ = $(round(VEHICLE.τ; digits=2)) s)"
    paths_panel(vehicle_runs, title, MODEL_COLORS.braitenberg; ylims=PATHS_YLIMS)
end

# ╔═╡ db854b22-08b1-56d5-9dcc-7fe574179466
md"""
**1B. Warren's paths.** Warren's walker (blue), with ``c_2 = 0``, from the same starts. Its paths nearly coincide with the vehicle's, at most 1.1 cm apart.
"""

# ╔═╡ ff9373bd-afcd-59e3-be67-092b22544062
"""Warren's paths from `START` (1B), with c₂ = 0."""
panel_warren_paths = let title = "Warren (Fajen & Warren 2003, c₂ = 0)"
    paths_panel(warren_runs, title, MODEL_COLORS.warren; ylims=PATHS_YLIMS)
end

# ╔═╡ 8471b2ab-b74f-565e-8606-a9552805e479
md"""
**1C, 1D. Parameter charts.** The heading equations above, for the goal 2 m away, over two of their parameters.

*Held fixed.* 1C, the vehicle: lag ``\tau`` = 0.31 s, decay ``c_d`` = 0.4 /m, goal distance ``r`` = 2 m (so the gradient is ``e^{-c_d r}`` = 0.45), sensor separation ``\delta`` = 1 m and ``\delta/d`` = 2.31 (wheel base ``d`` = 0.43 m). 1D, Warren: goal distance ``d`` = 2 m, decay ``c_1`` = 0.4 /m and ``c_2 = 0``. Neither chart depends on the speed ``V``.

The colours mean the same in both panels:

- **approach** (green): the heading settles toward the goal without overshoot;
- **approach (overshoot)** (pale yellow): it settles toward the goal with overshoot;
- **avoid** (pale pink): it settles pointing away from the goal;
- **wander** (lavender): it never settles. The vehicle circles; Warren's heading oscillates with growing amplitude;
- **approach or wander** and **avoid or wander** (violet, 1C only): it settles or circles, depending on how it starts.

The lines are drawn by what happens on them (legend under the charts). The colour says whether a line is a bifurcation:

- **black**, a bifurcation: rest points of the heading, or its circling orbit, appear, disappear or meet;
- **white**, a degenerate bifurcation: stability changes, but on the line a whole family of states is neutral (every heading is at rest, or every swing repeats). This happens only because the steering vanishes there or the law is exactly linear;
- **grey dots**, not a bifurcation: the eigenvalues of the settled heading turn from real to complex, which starts the overshoot, but the phase portrait does not change. A stable node and a stable focus are topologically equivalent.

The dash says which kind, in black and white alike:

- **solid**, a fold: an eigenvalue passes through zero, and a stable rest point meets an unstable one;
- **dashed**, a homoclinic bifurcation: no eigenvalue passes through zero, but the circling orbit appears from a loop through a saddle;
- **dash-dot**, a Hopf bifurcation: a pair of eigenvalues crosses the imaginary axis. Here it occurs only in its degenerate form.

The black dots mark where a homoclinic curve leaves a fold line, and the white dot the origin, where nothing steers.

The orange point is the vehicle of 1A, ``f_{\mathrm l} = f_{\mathrm r} = f`` = 1 /s, and the blue point is Warren's law of 1B, ``(b, k_t)`` = (3.25, 7.5) with ``c_2 = 0``. Both lie in approach (overshoot): at 2 m both overshoot slightly, with damping ratio 0.89.
"""

# ╔═╡ 760f0219-7c04-593c-bf3e-91953c563c7d
md"""
**1C. The two synapses.** The vehicle's heading equation over its synapses ``f_{\mathrm l}`` and ``f_{\mathrm r}``.

- *Folds* (the solid black lines through the origin), also called saddle-nodes. The rest points satisfy ``\sin\psi_t^* = \gamma``, and the stable one has ``\bar f\cos\psi_t^* > 0``. They exist only while ``\lvert\gamma\rvert \le 1``. On the lines ``f_{\mathrm r} = f_{\mathrm l}(1 \mp a)/(1 \pm a)``, with ``a = \delta c_d/2`` = 0.2, the stable and unstable headings merge, and past them the heading turns for ever. Up to ``\lvert\bar f\rvert`` = 2.20 /s (the black dots) the circling orbit is born on the line itself, with an infinite period (a saddle-node on an invariant circle). Beyond it the fold is an ordinary one: the circling orbit already exists, and the line only removes the rest points.
- *Homoclinic curves* (the dashed black curves inside the wedges). Beyond ``\lvert\bar f\rvert`` = 2.20 /s, where the damping ratio at ``\psi_t = 0`` falls below 0.60, the pendulum swings so freely that a circling orbit already exists before the saddle-node line. It is born where the saddle's unstable branch comes back to the saddle one turn later, a homoclinic orbit. Each curve leaves its fold line at a black dot, a saddle-node separatrix loop. Between each curve and its line (violet) the vehicle comes to rest or circles, depending on how it starts: at ``\bar f`` = 4 /s the band spans ``\gamma`` from 0.912 to 1. The curves are `HOMOCLINIC` (above), checked by simulation. A continuation of the circling orbit with BifurcationKit.jl, outside this notebook, ends on the same curve.
- *The origin* (white dot). With ``f_{\mathrm l} = f_{\mathrm r} = 0`` nothing steers and every heading is at rest; approach, avoid and wander meet there.
- *Overshoot* (the grey dots, a hyperbola). Near the stable heading the vehicle is a damped oscillator with damping ratio ``\zeta = 1/\bigl(2\sqrt{\tau\,\tfrac{\delta}{d}\,e^{-c_d r}\,\bar f\cos\psi_t^*}\bigr)``. It overshoots (``\zeta < 1``) inside the hyperbola ``\bar f^2 - \bigl((f_{\mathrm l} - f_{\mathrm r})/\delta c_d\bigr)^2 = W^2``, with ``W = 1/\bigl(4\tau\,\tfrac{\delta}{d}\,e^{-c_d r}\bigr)`` = 0.78 /s. Its asymptotes are the saddle-node lines. Turning away has the mirror-image hyperbola in the avoid wedge, which is not drawn.

The vehicle approaches for ``\bar f > 0`` between the lines, avoids for ``\bar f < 0`` between the lines, and wanders outside the lines, circling left below the diagonal (``f_{\mathrm l} > f_{\mathrm r}``) and right above it (``f_{\mathrm r} > f_{\mathrm l}``). The match to Warren fixes only ``\delta/d = m/f``, so ``\delta`` = 1 m, which sets the width of the wedge, is illustrative.
"""

# ╔═╡ 46241319-db71-51be-a806-5f4619a25590
"""
The vehicle's synapse chart (f_l, f_r) at the goal of `CHART` (1C): its regions and bifurcations, the vehicle
`point`, and the names of the regions, with `leaders` to those too narrow to hold their names.
"""
panel_synapse_chart = let lim = 8.0, point = (VEHICLE.f, VEHICLE.f)
    leaders = (((2.3, -0.45), (0.5, 0.45)), ((5.9, 2.85), (7.0, 4.85)), ((-5.9, -2.85), (-7.0, -4.85)))
    names = [(0.5lim, 0.5lim, chart_label("approach (overshoot)"; color=INK, rotation=45)),
        (2.35, -0.6, chart_label("approach"; color=INK, halign=:left)),
        (5.9, 2.6, chart_label("approach or wander"; color=INK)), (-5.9, -2.6, chart_label("avoid or wander"; color=INK)),
        (-0.6lim, -0.6lim, chart_label("avoid"; color=INK, rotation=45)),
        (-0.45lim, 0.45lim, chart_label("wander"; color=INK)), (0.45lim, -0.45lim, chart_label("wander"; color=INK))]
    p = plot(; aspect_ratio=:equal, xlims=(-lim, lim), ylims=(-lim, lim), grid=false, legend=false,
        xlabel="fₗ: left sensor → right motor (1/s)", ylabel="fᵣ: right sensor → left motor (1/s)",
        title="Braitenberg: crossed synapses (τ = $(round(CHART.τ; digits=2)) s, r = 2 m)")
    synapse_regions!(p; lim)
    synapse_bifurcations!(p; lim)
    scatter!(p, [point[1]], [point[2]]; color=MODEL_COLORS.braitenberg, markersize=7, markerstrokecolor=:white, markerstrokewidth=1.5)
    for (from, to) in leaders
        plot!(p, [from[1], to[1]], [from[2], to[2]]; color=INK, linewidth=0.8)
    end
    annotate!(p, names)
    p
end

# ╔═╡ 8609cb61-75e3-5583-b5fb-c545e0a37e55
md"""
**1D. Damping and stiffness.** Warren's heading equation is linear, so both of its bifurcations are degenerate and the chart has no black line.

- *Stiffness zero* (solid white, ``k_t = 0``). An eigenvalue passes through zero and ``\psi_t = 0`` turns from stable into a saddle: approach becomes avoid. On the line nothing steers, and every heading is at rest.
- *Damping zero* (dash-dot white, ``b = 0`` with ``k_t > 0``). A pair of eigenvalues crosses the imaginary axis, so the heading's swings go from damped to growing: approach becomes wander. This is the eigenvalue condition of a Hopf bifurcation, but the cubic coefficient that would set a cycle's amplitude is zero (BifurcationKit.jl classes it as singular). On the line every swing repeats at its own amplitude, and no cycle is born on either side.
- *Critical damping* (the grey dots, the parabola ``b^2 = 4 k_t e^{-c_1 d}``), with overshoot to its left. Below ``k_t = 0`` the chart shows only down to ``-5``: the whole half-plane is avoid.
"""

# ╔═╡ d0b6d08d-12ba-5be8-b358-72e967113e63
"""
Warren's chart (b, k_t) at the goal of `CHART`, c₂ = 0 (1D): its regions, the degenerate lines k_t = 0 (an
eigenvalue at zero) and b = 0 (Hopf), critical damping (not a bifurcation), and the law `point`; drawn square
like the synapse chart.
"""
panel_warren_chart = let bmin = -3.0, bmax = 10.0, kmin = -5.0, kmax = 25.0, point = (WARREN.b, WARREN.k)
    bs = range(0, bmax; length=200)
    crit = min.(bs .^ 2 ./ (4CHART.G), 1.05kmax)          # critical damping b² = 4 k_t e^{−c₁d}
    p = plot(; xlims=(bmin, bmax), ylims=(kmin, kmax), aspect_ratio=(bmax - bmin) / (kmax - kmin), grid=false, legend=false,
        xlabel="damping b (1/s)", ylabel="stiffness kₜ (1/s²)", title="Warren: damping and stiffness (d = 2 m, c₂ = 0)")
    plot!(p, Shape([bmin, bmax, bmax, bmin], [1.05kmin, 1.05kmin, 0, 0]); fillcolor=REGIONS.away, linewidth=0)
    plot!(p, Shape([bmin, 0, 0, bmin], [0, 0, kmax, kmax] .* 1.05); fillcolor=REGIONS.never, linewidth=0)
    plot!(p, Shape([bs; reverse(bs)], [zeros(length(bs)); reverse(crit)]); fillcolor=REGIONS.approach, linewidth=0)
    plot!(p, Shape([bs; reverse(bs)], [crit; fill(1.05kmax, length(bs))]); fillcolor=REGIONS.overshoot, linewidth=0)
    keep = crit .<= kmax
    dotted!(p, bs[keep], crit[keep]; scale=(bmax - bmin, kmax - kmin))
    plot!(p, [bmin, bmax], [0, 0]; LINE_KINDS.degenerate_fold...)
    plot!(p, [0, 0], [0, 1.05kmax]; LINE_KINDS.degenerate_hopf...)
    scatter!(p, [point[1]], [point[2]]; color=MODEL_COLORS.warren, markersize=7, markerstrokecolor=:white, markerstrokewidth=1.5)
    annotate!(p, [(7.5, 6.0, chart_label("approach"; color=INK)), (2.6, 19.0, chart_label("approach (overshoot)"; color=INK)),
        (3.5, kmin / 2, chart_label("avoid"; color=INK)), (-1.5, 12.5, chart_label("wander"; color=INK))])
    p
end

# ╔═╡ 7747c305-c23d-510f-b2da-5becdc46fcc3
md"""
**From 1C to 1D.** Only the diagonal of 1C, ``f_{\mathrm l} = f_{\mathrm r} = f``, has a counterpart in 1D. A symmetric vehicle with synapse ``f`` and lag ``\tau`` is the point ``(b, k_t) = (1/\tau, m/\tau)``, with ``m = \tfrac{\delta}{d}f``. So the orange point maps onto the blue one: the fitted law is the vehicle ``f`` = 1 with ``\tau`` = 0.31 s. One synapse at every lag is a ray ``k_t/b = m`` from the origin, with ``\tau = 1/b`` along it; one lag at every synapse is the vertical line ``b = 1/\tau``. The vertex of the hyperbola in 1C, ``f = W`` = 0.78 /s, is where the line ``b = 1/\tau`` = 3.25 /s crosses the dotted parabola in 1D: both are ``\zeta = 1``.
"""

# ╔═╡ c40ee4ca-c2b4-4c62-b706-d8e767272b6b
md"""
## 2. Steering laws, heading error and three-parameter charts

### Introduction: what figure 2 shows

*Figure 2 looks closely at the one difference between the two models, and then turns three knobs at a time instead of two. As in §1, the panels appear one by one further down, and the whole figure is at the bottom under **Figures**.*

**2A: the two steering rules.** For each heading error ``\psi_t``, a model's turning rate is drawn toward a target, the turning rate at which ``\ddot\phi = 0``. Setting ``\ddot\phi = 0`` in the equations of §1 gives the target ``-m\,e^{-c_d r}\sin\psi_t`` for the vehicle and ``-\tfrac{k_t}{b}\,e^{-c_1 d}\,\psi_t`` for Warren. The minus sign means that the agent turns so as to shrink the error. The factor in front of ``\sin\psi_t`` or ``\psi_t``, the *gain*, is the same for the two matched models (``m = k_t/b``), so 2A divides it out and plots only ``-\sin\psi_t`` (orange) and ``-\psi_t`` (blue). Inside the grey band, the ±25° of the walking experiments, the two curves lie on top of each other. Outside it they part: Warren's keeps growing, while the vehicle's falls back to zero as the goal moves behind it. That is why, far outside the experimental range, the vehicle can lose the goal (§1, "Outside the experimental range").

**2B: the heading error over time.** The goal is held still 2 m away, as in the charts, and the code simulates the heading alone, starting from heading errors of 0.2, 1.2 and 2.2 rad, not yet turning. Warren is solid and the vehicle dashed. From 0.2 rad the curves coincide. From 2.2 rad the vehicle turns more slowly at first, because there ``\sin\psi_t`` is much smaller than ``\psi_t``. Both overshoot a little, by under 1 % of the starting error, too little to see at this scale.

**2C, 2D: three knobs at a time.** With equal wires the vehicle has three knobs: its lag ``\tau``, the decay ``c_d`` (how fast the goal's signal fades with distance) and its motor strength ``m``. Warren's law has its damping ``b``, decay ``c_1`` and stiffness ``k_t`` (with ``c_2 = 0``). Each chart is a box: the lag or damping runs across, the decay into the page, and the motor strength or stiffness up. Each point inside the box is one setting of the three knobs, and the regions of figure 1 become volumes. A picture cannot show colour inside a volume, so the code paints each region's colour onto the far walls and the floor of the box, and draws the borders between the regions as sheets:

- *the flat white sheet*, ``m = 0`` (2C) or ``k_t = 0`` (2D): below it the agent turns away from the goal (avoid, pink);
- *the curved pale yellow sheet*, critical damping: above it the heading overshoots (approach (overshoot)), and between it and the flat sheet it settles without overshoot (approach, green). The grey dotted lines across it are slices at the decays 0, 0.2, …, 1 /m. The dark one, at the fitted decay 0.4 /m, is the dotted curve of 1D;
- in 2D only, *the grey wall* ``b = 0``: behind it the damping is negative and the heading wanders (lavender).

**Where the curved sheet comes from.** For small heading errors, where ``\sin\psi_t \approx \psi_t``, both heading equations are a damped spring,
```math
\ddot\psi_t = -B\,\dot\psi_t - K\,\psi_t ,
```
with damping ``B`` and stiffness ``K``. A damped spring overshoots when its stiffness is large compared with its damping: exactly when ``B^2 < 4K``. For Warren's law ``B = b`` and ``K = k_t e^{-c_1 d}``, so the sheet is ``k_t = b^2 e^{c_1 d}/4``. For the vehicle ``B = 1/\tau`` and ``K = (m/\tau)\,e^{-c_d r}``, so the sheet is ``m = e^{c_d r}/(4\tau)``. The sheet rises with the decay, because a faster decay leaves less pull at 2 m, and a weaker pull needs a larger ``m`` or ``k_t`` to overshoot.

The two fitted models are a single point in both charts, the orange dot on the blue ring, because ``(b, c_1, k_t) = (1/\tau, c_d, m/\tau)``: 2C and 2D are the same set of models, measured with different knobs. The point lies above the curved sheet, so both models overshoot, as 2B shows.
"""

# ╔═╡ 87dc75eb-1ca1-4f19-a67a-2f89192e6fe4
md"""
### Introduction: how the code makes figure 2

**2A and 2B.** `panel_steering_laws` only plots the two functions for ``-\pi \le \psi_t \le \pi``. `panel_heading_error` takes the heading equations for the goal held still (`heading_systems()` from §1, a system of two numbers, ``\psi_t`` and ``\dot\psi_t``, for each model), simulates them from each starting error with `trajectory` as in §1, and plots ``\psi_t`` against time.

**2C and 2D.** The sheets and walls are surfaces made of small triangles (`mesh3d!` from Plots.jl). Each 3D panel cell first lists its regions: a colour, the range of the across axis it covers, and formulas for its bottom and top (approach, for example, lies between ``m = 0`` and the curved sheet). `walls3d!` paints those regions onto the far walls and the floor, and the panel then draws the sheets, the dotted slices, the fitted models and the labels on top.

**Where things are.** The cells of this section, in order:

| name | what it is |
|:--|:--|
| the two check cells at the top | the overshoot of each run in 2B; and a test of the sheets: at 10 000 random settings of the three knobs, the region given by the sheet formulas agrees with the heading equation's own eigenvalues near ``\psi_t = 0`` (complex: it overshoots; one with a positive real part: it turns away) |
| `sheet!` | draws a curved sheet, such as critical damping, strip by strip |
| `degenerate3d!`, `flat_rect!` | the white edges, and the flat white sheet |
| `sections!` | the grey dotted slices across the curved sheet |
| `label3`, `models3d!` | a label, and the two fitted models drawn as one point |
| `CAMERA3D`, `TITLE3D` | the direction 2C is seen from, and where the 3D titles go |
| `wall_band!`, `walls3d!` | paint the regions onto the far walls and the floor |
| `DECAYS` | the decays at which the sheets are built and the slices drawn |
| `panel_steering_laws`, `panel_heading_error`, `panel_vehicle_chart3d`, `panel_warren_chart3d` | the panels 2A, 2B, 2C and 2D |
| `figure_steering_parameters` (under **Figures**, at the bottom) | puts the four panels and the legend together, letters them and saves the figure to `Figures/` |

To see a 3D chart from another side, change `CAMERA3D` (for 2C), or `camera` in the first line of `panel_warren_chart3d` (for 2D): the pair is (azimuth, elevation) in degrees. The walls follow the camera, so the coloured ones stay at the back.
"""

# ╔═╡ b134718b-dbd7-5625-850e-94dae828fc86
md"""
### Section 2 in detail

Figure 2, `Approach2`: the two steering laws (2A) and the heading error over time (2B), and the three-parameter charts of the vehicle (2C) and of Warren's law (2D).

**Three-parameter charts.** With symmetric synapses the vehicle's heading equation has three parameters: the lag ``\tau``, the decay ``c_d`` and the motor strength ``m``. Warren's has the damping ``b``, the decay ``c_1`` and the stiffness ``k_t`` (with ``c_2 = 0``). Both charts are for a goal 2 m away and are laid out the same way: the lag or damping across, the decay into the page, the motor strength or stiffness up.

The two charts are one parameter space in two coordinate systems, ``(b, c_1, k_t) = (1/\tau, c_d, m/\tau)``, and each sheet of one is the image of the same sheet of the other. The second check below confirms this at random points, from the eigenvalues at ``\psi_t = 0``.

*Held fixed.* Both charts: the goal 2 m away (``r = d`` = 2 m). The vehicle has symmetric synapses and ``\delta/d`` = 2.31; Warren's law has ``c_2 = 0``. The speed does not enter.
"""

# ╔═╡ 7db5d7fb-9a74-5b77-a837-7d3f6540ab59
let
    # The heading error over time (2B), from rest at the three initial errors: overshoot past ψₜ = 0 of each run.
    sys = heading_systems()
    map((0.2, 1.2, 2.2)) do ψ0
        w = trajectory(IVP(sys.warren, [ψ0, 0.0]), (0.0, 8.0); saveat=0.001)
        v = trajectory(IVP(sys.vehicle, [ψ0, 0.0]), (0.0, 8.0); saveat=0.001)
        (; ψ0, warren_overshoot_percent=-100minimum(first.(w.u)) / ψ0, vehicle_overshoot_percent=-100minimum(first.(v.u)) / ψ0,
            max_gap_rad=maximum(abs(a[1] - b[1]) for (a, b) in zip(w.u, v.u)))
    end
end

# ╔═╡ dbf97e3c-6186-5667-bab6-0b3ae06ba2b1
let
    # 1. The critical decay of each vehicle on the line τ = 0.31 s: in the vehicle's chart, m = e^{c d}/(4τ) solved
    #    for c; in Warren's, by bisection, where the sheet k_t = b² e^{c d}/4 meets the point (1/τ, m/τ).
    τ, d = VEHICLE.τ, CHART.d
    ms = VEHICLE.m .* (0.5, 1.0, 1.5)
    function bisect(f, lo, hi)
        for _ in 1:60
            mid = (lo + hi) / 2
            f(mid) < 0 ? (lo = mid) : (hi = mid)
        end
        (lo + hi) / 2
    end
    left = [log(4τ * m) / d for m in ms]
    right = [bisect(c -> (1 / τ)^2 * exp(c * d) / 4 - m / τ, 0.0, 2.0) for m in ms]
    # 2. The sheets against the eigenvalues at ψₜ = 0, at random points of the vehicle's chart and their images in
    #    Warren's: complex eigenvalues are overshoot, a positive real one a saddle (turn away).
    function class(J)
        λ = eigvals(J)
        any(real.(λ) .> 0) ? :saddle : isreal(λ) ? :no_overshoot : :overshoot
    end
    sheet_class(τ, c, m) = m < 0 ? :saddle : m < exp(c * d) / (4τ) ? :no_overshoot : :overshoot
    disagreements = count(1:10_000) do _
        τ, c, m = rand(), rand(), 8rand() - 2
        G = exp(-c * d)
        vehicle = class([0 1; -m*G/τ -1/τ])
        warren = class([0 1; -(m / τ)*G -1/τ])
        !(vehicle == warren == sheet_class(τ, c, m))
    end
    (; critical_decay_left=left, critical_decay_right=right, damping_ratio_fitted_at_2m=CHART.damping_ratio, random_points=10_000, disagreements)
end

# ╔═╡ 902d7851-4f13-550b-b278-74b779fef3b8
"""
A sheet through the curves `p(y; n)` (`n` 3D points for each y in `ys`), one strip per pair of y, in the pale yellow
of overshoot unless given a `color`. It is drawn twice, the second time on the midpoints of `ys` and `n + 1` points,
so that each covers the other's seams (the light lines between triangles). Each pass has opacity `alpha`, so the sheet
has 1 − (1 − alpha)² (0.75 at the default): slightly transparent, so that the walls behind it tint it and it stands
apart from the yellow back wall.
"""
function sheet!(plt, p, ys; n=60, color=REGIONS.overshoot, alpha=0.5)
    for (zs, m) in ((ys, n), ([ys[1]; (ys[1:end-1] .+ ys[2:end]) ./ 2; ys[end]], n + 1))
        for (y1, y2) in zip(zs[1:end-1], zs[2:end])
            a, b = p(y1; n=m), p(y2; n=m)
            mesh3d!(plt, [first.(a); first.(b)], [getindex.(a, 2); getindex.(b, 2)], [last.(a); last.(b)];
                connections=[[(i, i + 1, m + i + 1) for i in 1:m-1]; [(i, m + i + 1, m + i) for i in 1:m-1]], color, alpha, linewidth=0)
        end
    end
end

# ╔═╡ 8b2460b1-e8f9-5cac-9b74-a3e0bdfdbac2
"""The edge `x`, `y`, `z` of a degenerate bifurcation in 3D, white in the dash of its `kind`, as in figure 1."""
degenerate3d!(plt, x, y, z; kind=LINE_KINDS.degenerate_fold) = plot3d!(plt, x, y, z; kind...)

# ╔═╡ 7ec4d7d5-2dbb-5fd5-8665-f356f1591df3
"""The rectangle `xs` × `ys` flat at height `z`, a degenerate bifurcation: white, with a white edge."""
function flat_rect!(plt, xs, ys, z; alpha=1.0)
    mesh3d!(plt, [xs[1], xs[2], xs[2], xs[1]], [ys[1], ys[1], ys[2], ys[2]], fill(float(z), 4); connections=[(1, 2, 3), (1, 3, 4)], color=:white, alpha, linewidth=0)
    degenerate3d!(plt, [xs[1], xs[2], xs[2], xs[1], xs[1]], [ys[1], ys[1], ys[2], ys[2], ys[1]], fill(float(z), 5))
end

# ╔═╡ a16b3c67-3659-5e91-89e2-f6fe0986cde1
"""
Draw the curves `p(c)` for the decays `cs` as sections of grey dots (critical damping is not a bifurcation), the
one at the fitted decay heavier. `scale` is the chart's extent along its three axes, which sets the spacing.
"""
function sections!(plt, p, cs; scale, fitted=WARREN.c1)
    for c in cs
        heavy = c ≈ fitted                      # larger dots, spaced wider in proportion
        q = evenly_spaced(p(c; n=400), scale, LINE_KINDS.not_bifurcation.spacing * (heavy ? 1.5 : 1.0))
        scatter3d!(plt, first.(q), getindex.(q, 2), last.(q); color=heavy ? INK_SECONDARY : INK_MUTED,
            markersize=LINE_KINDS.not_bifurcation.markersize * (heavy ? 1.4 : 1.0), markerstrokewidth=0)
    end
end

# ╔═╡ c29cf4e1-0623-5bbc-bb49-db785f456e94
"A label in a 3D chart: `s` in `size` points, in ink unless given a `color`."
label3(s; size=8, color=INK, halign=:center) = text(s, size, color, halign)

# ╔═╡ 6f55fc4b-ef01-5c4f-88ca-677274fffded
"""
The fitted vehicle and Warren's fitted law at (`x`, `y`, `z`), one point in either chart: an orange dot on a blue
ring, as in figure 3, named on either side.
"""
function models3d!(plt, x, y, z)
    scatter3d!(plt, [x], [y], [z]; color=MODEL_COLORS.warren, markersize=12, markerstrokewidth=0)
    scatter3d!(plt, [x], [y], [z]; color=MODEL_COLORS.braitenberg, markersize=6, markerstrokecolor=:white, markerstrokewidth=1.2)
    annotate!(plt, [(x, y, z, label3("Braitenberg     "; halign=:right)), (x, y, z, label3("     Warren"; halign=:left))])
end

# ╔═╡ ea1e4da3-92e7-5601-a48b-d21d0d7587e7
"The camera (azimuth, elevation) of the vehicle's 3D chart; Warren's sets its own."
CAMERA3D = (30, 15)

# ╔═╡ c52d58cf-6051-5885-8490-e0527422cc8b
"The 3D charts' titles, set down inside the plot area just above the box rather than at the top of the subplot."
TITLE3D = (; titlelocation=(0.5, 0.92), titlefontvalign=:bottom, titlefonthalign=:hcenter)

# ╔═╡ 2999d37f-0a09-57b1-9f80-125364d1f09b
"""
A band on a wall of a 3D chart: between the curves `lo(u)` and `hi(u)` for `u` in `us`, where `at(u, w)` puts the
wall's coordinates (u along it, w up it) into the chart. Drawn twice, as `sheet!`, so that no seams show.
"""
function wall_band!(plt, at, us, lo, hi; color)
    for vs in (us, [us[1]; (us[1:end-1] .+ us[2:end]) ./ 2; us[end]])
        a, b = [at(u, lo(u)) for u in vs], [at(u, hi(u)) for u in vs]
        n = length(vs)
        mesh3d!(plt, [first.(a); first.(b)], [getindex.(a, 2); getindex.(b, 2)], [last.(a); last.(b)];
            connections=[[(i, i + 1, n + i + 1) for i in 1:n-1]; [(i, n + i + 1, n + i) for i in 1:n-1]], color, linewidth=0)
    end
end

# ╔═╡ f2d8cf49-0132-5602-83c6-649064e8a8eb
"""
The far walls of the box `xl` × `yl` × `zl` seen from `camera` (azimuth, elevation), coloured as figure 1 colours its
charts: the back and side walls on the far side from the camera's azimuth show the `regions` they cut, and the floor
the regions that reach down to it. Each region is its `color`, the range `xs` of x it spans, and its bottom
`lo(x, y)` and top `hi(x, y)`, clipped to the box.
"""
function walls3d!(plt, xl, yl, zl, regions; camera=CAMERA3D, n=400)
    xw, yw = sind(camera[1]) >= 0 ? xl[1] : xl[2], cosd(camera[1]) >= 0 ? yl[2] : yl[1]
    clip(z) = clamp(z, zl...)
    for r in regions
        all(r.lo(x, y) <= zl[1] for x in r.xs, y in yl) &&
            wall_band!(plt, (u, w) -> (u, w, zl[1]), [r.xs...], u -> yl[1], u -> yl[2]; r.color)
    end
    for r in regions
        wall_band!(plt, (u, w) -> (u, yw, w), range(r.xs...; length=n), u -> clip(r.lo(u, yw)), u -> clip(r.hi(u, yw)); r.color)
        r.xs[1] <= xw <= r.xs[2] &&
            wall_band!(plt, (u, w) -> (xw, u, w), range(yl...; length=n), u -> clip(r.lo(xw, u)), u -> clip(r.hi(xw, u)); r.color)
    end
end

# ╔═╡ 762c03d5-3098-50d7-9734-87a01710e7d6
"The decays of the 3D charts: every one, for the sheets, and the sections drawn as grey dots."
DECAYS = (all=range(0, 1; length=41), sections=0.0:0.2:1.0)

# ╔═╡ 48a1c711-7126-57c5-938d-36a58d9bc095
md"""
**2A. Steering laws.** The turning rate each model steers toward, per unit gain, against the heading error. Divided by ``\tau``, the vehicle's heading equation is ``\ddot\phi = \tfrac{1}{\tau}\bigl(-\dot\phi - m e^{-c_d r}\sin\psi_t\bigr)``: its turning rate relaxes toward ``-m e^{-c_d r}\sin\psi_t``. Warren's, written as ``\ddot\phi = b\bigl(-\dot\phi - \tfrac{k_t}{b}e^{-c_1 d}\psi_t\bigr)`` with ``c_2 = 0``, relaxes toward ``-\tfrac{k_t}{b}e^{-c_1 d}\psi_t``. The gain is the same, ``m = k_t/b``, and so is the lag, ``\tau = 1/b``. Per unit gain the two laws are ``-\sin\psi_t`` and ``-\psi_t``. The shaded band is the experimental range of heading errors, ±25° (0.436 rad); at its edge the two differ by ``1 - \sin\psi_t/\psi_t`` = 3 %.
"""

# ╔═╡ 803d309c-98eb-55db-83dd-899bac6625d6
"""The two steering functions per unit gain, −sin ψₜ and −ψₜ, with the experimental range ±25° shaded (2A)."""
panel_steering_laws = let ψ = range(-π, π; length=401), band = deg2rad(25)
    p = plot(; xlabel="heading error ψₜ (rad)", ylabel="turning rate steered toward, per unit gain", legend=:topright,
        title="Steering laws", xticks=([-π, -π / 2, 0, π / 2, π], ["−π", "−π/2", "0", "π/2", "π"]))
    vspan!(p, [-band, band]; color=INK_MUTED, alpha=0.15, linewidth=0, label="experimental range (±25°)")
    plot!(p, ψ, -ψ; color=MODEL_COLORS.warren, label="Warren: −ψₜ")
    plot!(p, ψ, -sin.(ψ); color=MODEL_COLORS.braitenberg, label="Braitenberg: −sin ψₜ")
    p
end

# ╔═╡ 50005106-2c9b-5ec1-a242-3b66ed5e726b
md"""
**2B. Heading error over time.** The two heading equations for a goal held at a fixed direction 2 m away, with ``c_2 = 0``: Warren's ``\ddot\psi_t = -b\dot\psi_t - k_t e^{-c_1 d}\psi_t`` (solid) and the vehicle's ``\tau\ddot\psi_t = -\dot\psi_t - m e^{-c_d r}\sin\psi_t`` (dashed). Both start from heading errors of 0.2, 1.2 and 2.2 rad, not yet turning. Both have the damping ratio ``b/\bigl(2\sqrt{k_t e^{-c_1 d}}\bigr)`` = 0.89 there, so both overshoot, by less than 1 % (checked above). From 0.2 rad the curves coincide; from 2.2 rad the vehicle turns more slowly at first, because ``\sin\psi_t < \psi_t``.
"""

# ╔═╡ 8f6e0244-b4e2-5647-9f54-28c2fd408ba7
"""Heading error over time at the goal of `CHART`, from rest at the heading errors `starts`: Warren solid, the vehicle dashed (2B)."""
panel_heading_error = let T = 8.0, starts = (0.2, 1.2, 2.2)
    sys = heading_systems()
    ts = range(0, T; length=401)
    band = deg2rad(25)
    p = plot(; xlabel="time (s)", ylabel="heading error ψₜ (rad)", ylims=(-0.3, 2.4), xlims=(-0.9, T), legend=:topright,
        title="Heading error, goal at 2 m (c₂ = 0)")
    hspan!(p, [-band, band]; color=INK_MUTED, alpha=0.15, linewidth=0, label="experimental range (±25°)")
    for (i, ψ0) in enumerate(starts)
        w = trajectory(IVP(sys.warren, [ψ0, 0.0]), (0.0, T); saveat=ts)
        v = trajectory(IVP(sys.vehicle, [ψ0, 0.0]), (0.0, T); saveat=ts)
        plot!(p, ts, first.(w.u); color=MODEL_COLORS.warren, linewidth=1.5, label=i == 1 ? "Warren" : "")
        plot!(p, ts, first.(v.u); color=MODEL_COLORS.braitenberg, linewidth=1.5, linestyle=:dash, label=i == 1 ? "Braitenberg (τ = 1/b)" : "")
        annotate!(p, -0.1, ψ0, chart_label("$ψ0 rad"; size=7, halign=:right))
    end
    p
end

# ╔═╡ e6995bc2-3d8c-5d66-802b-b0ec638f63e6
md"""
**2C, 2D. Three-parameter charts.** The regions are coloured as in figure 1. A region is a volume here, so each chart colours the walls at the back of its box with the regions they cut. The back wall is the slice at the largest decay, 1 /m. The side wall is the slice at the smallest lag (2C) or damping (2D) drawn. The floor lies below the flat sheet, in avoid. The sheets are drawn as the lines of figure 1: white for a degenerate bifurcation, with edges in the dash of its kind, and grey dots for a sheet that is not a bifurcation (legend under the figure). In both charts:

- *The flat white sheet* (solid edges), ``m = 0`` in 2C and ``k_t = 0`` in 2D, a degenerate bifurcation with an eigenvalue at zero. Above it the heading settles toward the goal (approach). Below it ``\psi_t = 0`` is a saddle and the agent turns away (avoid). On the sheet nothing steers and every heading is at rest. The two sides are the same phase portrait turned by half a turn: the stable heading is ``\psi_t = 0`` above and ``\pi`` below. The sheet is a single crossing only because the synapses are held equal. In the synapse chart, 1C, it is the origin, and any difference between the synapses puts wander between approach and avoid.
- *The curved sheet,* critical damping, which is not a bifurcation: ``m = e^{c_d r}/(4\tau)`` in 2C and ``k_t = b^2 e^{c_1 d}/4`` in 2D. Between the flat sheet and the curved one the heading settles without overshoot (approach, green); above the curved sheet it overshoots (approach (overshoot)). The camera looks down onto the sheet from the overshoot side, so the sheet is drawn in the pale yellow of that region. It is slightly transparent, so the green walls behind it tint it and it stands apart from the yellow back wall. On the back wall it ends where green meets yellow. The grey dotted lines are its sections at decays of 0, 0.2, …, 1 /m. The heavy one, at 0.4 /m, is the dotted curve of Warren's chart, 1D.

Each chart marks both fitted models, which are one point: the Braitenberg vehicle, ``(\tau, c_d, m)`` = (0.31 s, 0.4 /m, 2.31 /s), as the orange dot, and Warren's law, ``(b, c_1, k_t)`` = (3.25 /s, 0.4 /m, 7.5 /s²), as the blue ring around it. The point lies above the curved sheet, so both overshoot. The curved sheet rises with the decay, because a steeper profile weakens the gradient at 2 m. So a vehicle of fixed lag and motor strength overshoots below a critical decay, ``\ln(4\tau m)/r``, and not above it: at the fitted lag, for ``f`` = 0.5, 1 and 1.5 /s, at 0.18, 0.52 and 0.72 /m, the same in both charts (checked above). The fitted vehicle, ``f`` = 1 at 0.4 /m, is below its critical decay of 0.52 /m.
"""

# ╔═╡ da8a5616-d659-59ee-9b8a-ccf28bf7aa5f
md"""
**2C. The vehicle: lag, decay and motor strength.** The lag ``\tau`` across, the decay ``c_d`` into the page, the motor strength ``m`` up. As ``\tau \to 0`` the vehicle becomes Rañó's first-order vehicle, which never overshoots: the curved sheet rises without bound.
"""

# ╔═╡ 46fd4f37-2354-5f5a-a026-990f0631da5f
"""
The vehicle's three-parameter chart (τ, c_d, m) for the goal of `CHART` (2C): the plane m = 0, the critical-damping
sheet m = e^{c_d r}/(4τ) with its sections, and the fitted models.
"""
panel_vehicle_chart3d = let τmax = 1.0, mmin = -2.0, mmax = 6.0
    d, τ0 = CHART.d, VEHICLE.τ
    critical(τ, c) = exp(c * d) / (4τ)
    curve(c; n=60) = [(τ, c, critical(τ, c)) for τ in range(exp(c * d) / (4mmax), τmax; length=n)]
    regions = [(color=REGIONS.away, xs=(0, τmax), lo=(τ, c) -> mmin, hi=(τ, c) -> 0.0),
        (color=REGIONS.approach, xs=(0, τmax), lo=(τ, c) -> 0.0, hi=critical),
        (color=REGIONS.overshoot, xs=(0, τmax), lo=critical, hi=(τ, c) -> mmax)]
    p = plot(; legend=false, grid=false, camera=CAMERA3D, xlims=(0, τmax), ylims=(0, 1), yticks=0:0.5:1, zlims=(mmin, mmax),
        xlabel="lag τ (s)", ylabel="decay \$c_d\$ (1/m)", zlabel="motor strength m (1/s)",
        title="Braitenberg: lag, decay, motor strength (r = 2 m)", TITLE3D...)
    walls3d!(p, (0.0, τmax), (0.0, 1.0), (mmin, mmax), regions)
    flat_rect!(p, (0, τmax), (0, 1), 0.0)
    sheet!(p, curve, DECAYS.all)
    sections!(p, curve, DECAYS.sections; scale=(τmax, 1.0, mmax - mmin))
    models3d!(p, τ0, VEHICLE.cd, VEHICLE.m)
    # each label inside its region: overshoot above the sheet, no overshoot between it and m = 0
    annotate!(p, [(0.55, 0.0, 5.6, label3("approach (overshoot)")), (0.95, 0.95, 0.75, label3("approach"; size=7)),
        (0.5, 0.2, -1.6, label3("avoid")), (τmax, 0.0, 0.0, label3("  m = 0"; size=7, color=INK_SECONDARY, halign=:left))])
    p
end

# ╔═╡ c2edb3b3-110a-5975-9d94-a18ce26ff892
md"""
**2D. Warren: damping, decay and stiffness.** The damping ``b`` across, the decay ``c_1`` into the page, the stiffness ``k_t`` up. The chart is seen more from the front than 2C, so that the wall ``b = 0`` is nearly edge-on and the wander region beside it shows.

- *The grey wall* (white dash-dot edges), ``b = 0``, a degenerate Hopf bifurcation: the linear centre of 1D. It is shaded grey, unlike the white flat sheet, so that it stands out against the walls. Behind it, ``b < 0``, the heading never settles (wander, lavender on the walls). No vehicle lies there, since ``\tau = 1/b`` is positive. The lags of 2C, up to 1 s, are the dampings from 1 /s up.
"""

# ╔═╡ 203b2963-3285-5ef8-95de-a54266aa8ed8
"""
Warren's three-parameter chart (b, c₁, k_t) for the goal of `CHART`, c₂ = 0 (2D): the planes k_t = 0 and b = 0, the
critical-damping sheet k_t = b² e^{c₁d}/4 with its sections, and the fitted models, seen from `camera`.
"""
panel_warren_chart3d = let bmin = -3.0, bmax = 6.0, kmin = -10.0, kmax = 25.0, camera = (12, 22)
    d = CHART.d
    critical(b, c) = b^2 * exp(c * d) / 4
    curve(c; n=60) = [(b, c, critical(b, c)) for b in range(0, min(bmax, sqrt(4kmax * exp(-c * d))); length=n)]
    regions = [(color=REGIONS.away, xs=(bmin, bmax), lo=(b, c) -> kmin, hi=(b, c) -> 0.0),
        (color=REGIONS.never, xs=(bmin, 0.0), lo=(b, c) -> 0.0, hi=(b, c) -> kmax),
        (color=REGIONS.approach, xs=(0.0, bmax), lo=(b, c) -> 0.0, hi=critical),
        (color=REGIONS.overshoot, xs=(0.0, bmax), lo=critical, hi=(b, c) -> kmax)]
    p = plot(; legend=false, grid=false, camera, xlims=(bmin, bmax), ylims=(0, 1), yticks=0:0.5:1, zlims=(kmin, kmax),
        xlabel="damping b (1/s)", ylabel="decay \$c_1\$ (1/m)", zlabel="stiffness kₜ (1/s²)",
        title="Warren: damping, decay, stiffness (d = 2 m, c₂ = 0)", TITLE3D...)
    walls3d!(p, (bmin, bmax), (0.0, 1.0), (kmin, kmax), regions; camera)
    flat_rect!(p, (bmin, bmax), (0, 1), 0.0)
    # the wall b = 0 is shaded grey, so that it stands out against the coloured walls behind it
    mesh3d!(p, [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, kmax, kmax]; connections=[(1, 2, 3), (1, 3, 4)], color=INK_MUTED, alpha=0.45, linewidth=0)
    degenerate3d!(p, [0, 0, 0, 0], [0, 0, 1, 1], [0, kmax, kmax, 0]; kind=LINE_KINDS.degenerate_hopf)
    sheet!(p, curve, DECAYS.all)
    sections!(p, curve, DECAYS.sections; scale=(bmax - bmin, 1.0, kmax - kmin))
    models3d!(p, WARREN.b, WARREN.c1, WARREN.k)
    annotate!(p, [(4.5, 0.0, 21.0, label3("approach (overshoot)")), (5.0, 0.3, 3.0, label3("approach"; size=7)),
        (2.5, 0.2, -8.0, label3("avoid")), (-1.5, 0.0, 19.0, label3("wander"; size=7)),
        (bmax, 0.0, 0.0, label3("  kₜ = 0"; size=7, color=INK_SECONDARY, halign=:left))])
    p
end

# ╔═╡ fdedf975-73aa-5aec-a1f2-6119695b789e
md"""
## Figures

The panels above, put together and saved to `Figures/` as PNG and PDF, each figure with a legend of its lines under it. A panel is changed in its own `panel_…` cell; a figure's layout, size, margins and legend in its cell here.
"""

# ╔═╡ f6ee2c3d-6cab-539b-b4ad-8e5405a5b37d
"""
    panel_letters!(plt, n; size = 14, inset = (1.5, 1.0))

Letter the first `n` subplots of the figure `plt` A, B, C, … (row by row; the legend strip after them is not
lettered), in bold at the left of each panel, level with its title: `inset` millimetres in from the top left corner,
or beside a title set down into the plot area (the 3D charts). As in `OscillationV2.jl`, the letters sit on one
transparent subplot laid over the whole figure, so that they are placed alike on 2D and 3D panels.
"""
function panel_letters!(plt, n; size=14, inset=(1.5, 1.0))
    panels = plt.subplots[1:n]
    Plots.prepare_output(plt)   # lay the figure out, to find where its panels are
    w, h = Plots.width(plt.layout.bbox), Plots.height(plt.layout.bbox)
    over = length(plt.subplots) + 1
    plot!(plt; inset=bbox(0, 0, 1, 1), subplot=over, framestyle=:none, background_color_subplot=:transparent,
        background_color_inside=:transparent, xlims=(0, 1), ylims=(0, 1), legend=false, grid=false, ticks=false)
    for (k, sp) in enumerate(panels)
        x = (Plots.left(sp.bbox) + inset[1] * Plots.mm) / w
        loc, area = sp[:titlelocation], sp.plotarea
        y, valign = loc isa Tuple ? (1 - (Plots.bottom(area) - loc[2] * Plots.height(area)) / h, :bottom) :
            (1 - (Plots.top(sp.bbox) + inset[2] * Plots.mm) / h, :top)
        annotate!(plt, x, y, text(string('A' + k - 1), size, INK, :left, valign, "helvetica bold"); subplot=over)
    end
    plt
end

# ╔═╡ 0415cfff-77fd-5cca-bef9-5b6807501087
"""
A legend as a panel of its own: `entries` (a style from `LINE_KINDS` or `POINT_KINDS`, and its label), `columns`
to a row, on a pale ground so that white shows.
"""
function legend_panel(entries; columns=4, ground=FILLS.away)
    rows = cld(length(entries), columns)
    p = plot(; xlims=(0, columns), ylims=(0, rows), framestyle=:none, grid=false, legend=false, ticks=false)
    plot!(p, Shape([0, columns, columns, 0], [0, 0, rows, rows]); fillcolor=ground, linewidth=0)
    for (k, (style, label)) in enumerate(entries)
        x, y = (k - 1) % columns + 0.04, rows - 0.5 - (k - 1) ÷ columns
        if haskey(style, :spacing)            # not a bifurcation: grey dots
            scatter!(p, range(x, x + 0.2; length=7), fill(y, 7); color=style.color, markersize=style.markersize, markerstrokewidth=0)
        elseif haskey(style, :markersize)     # a point
            scatter!(p, [x + 0.1], [y]; style...)
        else                                  # a line
            plot!(p, [x, x + 0.2], [y, y]; style...)
        end
        annotate!(p, x + 0.26, y, text(label, 8, INK_SECONDARY, :left))
    end
    p
end

# ╔═╡ 9b55c94d-a70f-5937-8d0f-d92a335f9c69
md"""
**Figure 1**, `Approach1` (§1): the paths of (A) the vehicle and (B) Warren's walker, each above its parameter chart: (C) the vehicle's two crossed synapses and (D) Warren's damping and stiffness.
"""

# ╔═╡ 6c19226b-2fcf-567e-8cf2-ea520703f788
"""Figure 1: the paths (A, B) above the parameter charts (C, D), with the charts' legend, lettered and saved to `Figures/`."""
figure_paths_charts = let
    legend = legend_panel([LINE_KINDS.fold => "fold", LINE_KINDS.homoclinic => "homoclinic",
        POINT_KINDS.fold_meets_homoclinic => "homoclinic curve leaves the fold", LINE_KINDS.not_bifurcation => "not a bifurcation",
        LINE_KINDS.degenerate_fold => "degenerate, eigenvalue at zero", LINE_KINDS.degenerate_hopf => "degenerate Hopf",
        POINT_KINDS.degenerate => "degenerate: nothing steers"])
    plt = plot(plot(panel_vehicle_paths, panel_warren_paths; layout=(1, 2)), plot(panel_synapse_chart, panel_warren_chart; layout=(1, 2)), legend;
        layout=grid(3, 1; heights=[0.28, 0.63, 0.09]), size=(1300, 1180), left_margin=4Plots.mm, right_margin=6Plots.mm, bottom_margin=5Plots.mm)
    save_figure(panel_letters!(plt, 4), "Approach1")
end

# ╔═╡ b0c88ffa-ebd0-59bc-b8a0-fc1307cae37e
md"""
**Figure 2**, `Approach2` (§2): (A) the two steering laws and (B) the heading error over time, and the three-parameter charts of (C) the vehicle, ``(\tau, c_d, m)``, and (D) Warren's law, ``(b, c_1, k_t)``.
"""

# ╔═╡ 394b195b-b944-50c8-9d1d-0814037cded0
"""Figure 2: the steering laws and heading error (A, B) above the three-parameter charts (C, D), with their legend, lettered and saved to `Figures/`."""
figure_steering_parameters = let
    legend = legend_panel([LINE_KINDS.degenerate_fold => "degenerate, eigenvalue at zero",
        LINE_KINDS.degenerate_hopf => "degenerate Hopf", LINE_KINDS.not_bifurcation => "not a bifurcation"]; columns=3)
    plt = plot(plot(panel_steering_laws, panel_heading_error; layout=(1, 2)), plot(panel_vehicle_chart3d, panel_warren_chart3d; layout=(1, 2)), legend;
        layout=grid(3, 1; heights=[0.38, 0.555, 0.065]), size=(1200, 1180), left_margin=4Plots.mm, bottom_margin=4Plots.mm)
    save_figure(panel_letters!(plt, 4), "Approach2")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HybridSystems = "2207ec0c-686c-5054-b4d2-543502888820"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MathematicalSystems = "d14a8603-c872-5ed3-9ece-53e0e82e39da"
OrdinaryDiffEqTsit5 = "b1df2697-797e-41e3-8120-5422d3b24e4a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
HybridSystems = "~0.4.4"
LazySets = "~7.1.0"
MathematicalSystems = "~0.14.6"
OrdinaryDiffEqTsit5 = "~2.1.5"
Plots = "~1.41.7"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "12253660da6151bc1ae3a5963b1307b2aadc7e5f"

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

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

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
git-tree-sha1 = "61761f58648aa7217445f24f841839b78c712232"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.3"
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
git-tree-sha1 = "291665b547f137df070e4dd83e432b5fee8cc4a0"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.2"

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
git-tree-sha1 = "4e3938f036cf93f3c6acf7a7008848675d3d130f"
registries = "General"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "7.21.3"

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
git-tree-sha1 = "90ab232666984c3eafa9d14cc85f8ff51cf6c353"
registries = "General"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.22"

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
git-tree-sha1 = "c8f8eefadaa330d982bbf5e395c34e53a13ab668"
registries = "General"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "1.4.0"

    [deps.FastBroadcast.extensions]
    FastBroadcastPolyesterExt = "Polyester"
    FastBroadcastStaticExt = "Static"

    [deps.FastBroadcast.weakdeps]
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    Static = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
registries = "General"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastPower]]
deps = ["PrecompileTools"]
git-tree-sha1 = "3c7269c236978d434a16ebe99f9743b9d706270a"
registries = "General"
uuid = "a4df4552-cc26-4903-aec0-212e50a0e84b"
version = "1.5.0"

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
git-tree-sha1 = "086b5fbd032baf544678cc15b52b015e4c4aceb8"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.17.1"

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
git-tree-sha1 = "dcce413ffacb0be72f38b491ae4887730343d451"
registries = "General"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "3.4.0"

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

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

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
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
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
git-tree-sha1 = "4657a834b01ce00f2b94546095c4350617b9af2c"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.10.0"

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
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "b85e2797b2409570e84c4de46238c0ed5f6476ae"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.2"

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

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

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
git-tree-sha1 = "7fc67976fca2c577ca96257061e4785463bb9fb8"
registries = "General"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.8.1"

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
git-tree-sha1 = "f4f5ebf8bfd9bfafdc4116fb3715dfd50fe4a7bd"
registries = "General"
uuid = "be0214bd-f91f-a760-ac4e-3421ce2b2da0"
version = "2.54.1"

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
git-tree-sha1 = "f9b03759e9ef463718934fbed30820b39997511e"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.2"

[[deps.OrdinaryDiffEqCore]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "BinaryHeaps", "CommonSolve", "ConstructionBase", "DiffEqBase", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "FastPower", "FindFirstFunctions", "FunctionWrappers", "FunctionWrappersWrappers", "InteractiveUtils", "LinearAlgebra", "Logging", "MacroTools", "MuladdMacro", "PrecompileTools", "Preferences", "Printf", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "2815c99c179c5c9d1a82d63394a975b67ef6b69c"
registries = "General"
uuid = "bbf590c4-e513-4bbe-9b18-05decba2e5d8"
version = "4.18.1"

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

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

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
git-tree-sha1 = "5005266de4bfe50e53ff44a5cb5c540b6e47a254"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.0"

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
git-tree-sha1 = "258850fd9188f5b1ac44bb03ab9225b4084500eb"
registries = "General"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "4.5.3"

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
git-tree-sha1 = "46345faa1fa3f43c24c241653c3b2719672b8d99"
registries = "General"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "3.57.0"

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
    ReactantCore = "a3311ec8-5e00-46d5-b541-4f83e724a433"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.SciMLJacobianOperators]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DifferentiationInterface", "FastClosures", "LinearAlgebra", "SciMLBase", "SciMLOperators"]
git-tree-sha1 = "e58036956d27635a57276b05bdbbe8b22376a805"
registries = "General"
uuid = "19f34311-ddf3-4b8b-af20-060888a46c0e"
version = "0.1.19"

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
git-tree-sha1 = "6c97cd41488ccd362363ff9c883dafcff18da45c"
registries = "General"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.30.2"

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
git-tree-sha1 = "5c2f9dbf6f07eea6bc9e93f117b00b7939a79f9e"
registries = "General"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.5"

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
git-tree-sha1 = "b814d5005d6a529d740ffe06f8a86396f6501138"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.9.2"

    [deps.StructUtils.extensions]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
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

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
registries = "General"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "50eda8011f098869a76da69a524ff59e0298711a"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.2"

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
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
registries = "General"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "2e59214e017a55cb87474a00fa76035c82ac0e17"
registries = "General"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+2"

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
# ╠═794af616-4626-52e5-a088-2a4e2aada361
# ╠═a1f3c2d4-6b7e-4c59-8d2a-3e4f5a6b7c8d
# ╟─566196d3-53ee-5ec0-a4e1-dfb4c1ee7454
# ╟─c7b8b1ab-b73b-5594-8c9f-a891d3b804c4
# ╠═10522db1-318c-58da-b436-4ccfb439aee7
# ╠═dc99a48e-03e9-5691-83b0-b6d17ba7de7d
# ╟─a3a24058-e49f-42c4-82f9-83bea10f3182
# ╟─9ce34595-0853-44fe-a6fb-c4a05547d278
# ╟─35f60d3f-7c71-5d66-839c-b879c7bb0d28
# ╠═8dbe941e-d8a4-50af-a57f-b2d4a4e90de5
# ╠═2880201f-796c-5eed-af72-c8aa7a133e34
# ╠═b7ea2177-2a75-59f6-a442-762a9c6533e8
# ╠═4ae435b1-648f-5e6c-9bb2-0607108bca23
# ╠═770c1295-6ded-559a-93d8-8384394db36d
# ╟─44364c74-3b97-5ad5-8128-6f78c7f4d3ba
# ╠═6a696545-2769-5ef6-b90a-6b9e550b8c81
# ╟─39d6d830-ea5e-50d2-ba2f-9434b1050c2e
# ╠═bbdccdbc-eb10-5625-a7da-0ced29a10b93
# ╟─72502e6f-6abe-51f4-a2d5-71a3447e1360
# ╠═9f305944-5da7-5429-863e-5f4746778e61
# ╠═80f14db7-48f4-57ac-8afb-b85a52b8cf65
# ╠═b33547d7-6138-5949-b72c-be68ce25a99b
# ╠═9e8ed77a-f7f6-59a2-ad6f-7acc42bf980d
# ╠═4893a1d8-4517-550f-8166-2deffee30e9a
# ╠═2c8a6a9d-fa54-5958-99d6-74c2d50a40a8
# ╠═99c8ca49-7c74-57e8-a892-b03a01f2a6e4
# ╠═80770646-8cbf-5260-82eb-fc0d6a34331c
# ╠═da884d59-72ca-51be-b42b-1726187bda8b
# ╠═bc8d7e9d-51a6-57ae-b8c8-a1aeee25df1a
# ╠═0b65fae3-4a61-5c1d-be00-da756ec8278c
# ╠═691d3375-d17d-5173-afaa-9c4c62a9bf84
# ╠═ece996ab-edcd-5da7-828f-763eb45ad8f2
# ╠═7a516e0e-b6d4-52af-9fa7-115c16ab2831
# ╠═37d35c20-1991-50ee-ac9d-b4cf14a7929f
# ╠═70b6b969-e8ed-5f2e-b06d-bc261de05201
# ╠═594c138b-1002-5647-b497-ea04252e321f
# ╠═b291984d-1792-50e4-81ff-a5718666219d
# ╠═3b0b14d2-82d3-50b8-933a-9caf1f1bae18
# ╠═d5d4755f-fd2a-5e54-9c28-2faa81e19530
# ╠═5d569900-0d01-5d93-bb40-f2f073499b81
# ╠═06841119-64a5-55da-b33d-11ad72ce1976
# ╠═8a503b49-54f0-555e-b754-96c9e5a78b21
# ╠═7b0fa879-d51c-51f5-b201-296f4a5b393f
# ╟─0154da58-76d4-57b0-af61-8fd3a841fff5
# ╠═a4997fff-8e99-5c16-beeb-e34aaed3886f
# ╟─db854b22-08b1-56d5-9dcc-7fe574179466
# ╠═ff9373bd-afcd-59e3-be67-092b22544062
# ╟─8471b2ab-b74f-565e-8606-a9552805e479
# ╟─760f0219-7c04-593c-bf3e-91953c563c7d
# ╠═46241319-db71-51be-a806-5f4619a25590
# ╟─8609cb61-75e3-5583-b5fb-c545e0a37e55
# ╠═d0b6d08d-12ba-5be8-b358-72e967113e63
# ╟─7747c305-c23d-510f-b2da-5becdc46fcc3
# ╟─c40ee4ca-c2b4-4c62-b706-d8e767272b6b
# ╟─87dc75eb-1ca1-4f19-a67a-2f89192e6fe4
# ╟─b134718b-dbd7-5625-850e-94dae828fc86
# ╠═7db5d7fb-9a74-5b77-a837-7d3f6540ab59
# ╠═dbf97e3c-6186-5667-bab6-0b3ae06ba2b1
# ╠═902d7851-4f13-550b-b278-74b779fef3b8
# ╠═8b2460b1-e8f9-5cac-9b74-a3e0bdfdbac2
# ╠═7ec4d7d5-2dbb-5fd5-8665-f356f1591df3
# ╠═a16b3c67-3659-5e91-89e2-f6fe0986cde1
# ╠═c29cf4e1-0623-5bbc-bb49-db785f456e94
# ╠═6f55fc4b-ef01-5c4f-88ca-677274fffded
# ╠═ea1e4da3-92e7-5601-a48b-d21d0d7587e7
# ╠═c52d58cf-6051-5885-8490-e0527422cc8b
# ╠═2999d37f-0a09-57b1-9f80-125364d1f09b
# ╠═f2d8cf49-0132-5602-83c6-649064e8a8eb
# ╠═762c03d5-3098-50d7-9734-87a01710e7d6
# ╟─48a1c711-7126-57c5-938d-36a58d9bc095
# ╠═803d309c-98eb-55db-83dd-899bac6625d6
# ╟─50005106-2c9b-5ec1-a242-3b66ed5e726b
# ╠═8f6e0244-b4e2-5647-9f54-28c2fd408ba7
# ╟─e6995bc2-3d8c-5d66-802b-b0ec638f63e6
# ╟─da8a5616-d659-59ee-9b8a-ccf28bf7aa5f
# ╠═46fd4f37-2354-5f5a-a026-990f0631da5f
# ╟─c2edb3b3-110a-5975-9d94-a18ce26ff892
# ╠═203b2963-3285-5ef8-95de-a54266aa8ed8
# ╟─fdedf975-73aa-5aec-a1f2-6119695b789e
# ╠═f6ee2c3d-6cab-539b-b4ad-8e5405a5b37d
# ╠═0415cfff-77fd-5cca-bef9-5b6807501087
# ╟─9b55c94d-a70f-5937-8d0f-d92a335f9c69
# ╠═6c19226b-2fcf-567e-8cf2-ea520703f788
# ╟─b0c88ffa-ebd0-59bc-b8a0-fc1307cae37e
# ╠═394b195b-b944-50c8-9d1d-0814037cded0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
