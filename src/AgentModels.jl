"""
    AgentModels

Model pieces shared by more than one notebook in this directory, written as
MathematicalSystems.jl systems. The model-independent machinery (running systems, figure
style, limit cycles and phase) is in `BehaviorTools.jl`.

- The logistic σ and its inverse (`OscillatoryBehavior.jl`, `ActiveCategorization.jl`).
- Warren's steering laws to a goal and around an obstacle (Fajen & Warren 2003) and Braitenberg's
  vehicle in Rañó's reduction, crossed or uncrossed (`ApproachBehavior.jl`, `AvoidanceBehavior.jl`).
- The CTRNN τᵢ ẏᵢ = −yᵢ + Σⱼ Wᵢⱼ σ(gⱼ(yⱼ + θⱼ)) + Iᵢ, the nervous system of Beer's walkers
  (`OscillatoryBehavior.jl`) and of his visual categorization agent
  (`ActiveCategorization.jl`), as a `BlackBoxControlContinuousSystem` whose input is the
  external input I; and a reader for Beer's `.ns` network files.
- Beer's (2003) visual categorization agent and its falling object, ported from Setzler's C++
  (`~/Research/BeerModels`), as a `BlackBoxDiscreteSystem` (`ActiveCategorization.jl`,
  `CategorizationLaw.jl`).

A network's parameters are a NamedTuple `(; τ, θ, g, W)` with `W[i, j]` the weight from
neuron j to neuron i (the transpose of Beer's C++ `weights[i][j]`).

Include it in the same cell as BehaviorTools:

    begin
        include(joinpath(@__DIR__, "BehaviorTools.jl"))
        include(joinpath(@__DIR__, "AgentModels.jl"))
        using .BehaviorTools, .AgentModels
    end
"""
module AgentModels

using MathematicalSystems
using ..BehaviorTools: wrap_angle

export σ, logit
export ctrnn_parameters, ctrnn_outputs, ctrnn_system, read_ns
export WARREN, BRAITENBERG, warren_approach, warren_avoidance, braitenberg_vehicle
export VISUAL, ray_length, ray_angles, ray_inputs, categorization_agent

"""Logistic σ(x) = 1 / (1 + e⁻ˣ) and its inverse `logit(o) = ln(o / (1 − o))`."""
σ(x) = 1 / (1 + exp(-x))
logit(o) = log(o / (1 - o))

# ── Warren and Braitenberg ──────────────────────────────────────────────────────────────

"""
Fajen & Warren (2003): goal damping `b` (1/s), stiffness `k` (1/s²), distance decay `c1` (1/m)
and offset `c2`; obstacle stiffness `ko` (1/s²), angular decay `c3` (1/rad) and distance decay
`c4` (1/m); walking `speed` (m/s).
"""
const WARREN = (b=3.25, k=7.5, c1=0.40, c2=0.40, ko=198.0, c3=6.5, c4=0.8, speed=1.0)

"""
Braitenberg vehicle (Rañó's reduction) with motor output F = wS for a stimulus S = e^{−c₁r}:
motor gain `w` (m/s), stimulus decay `c1` (1/m, Warren's goal decay) and sensor spacing over
wheel base `δd`, matched to Warren's goal law so that its turning gain (δ/d) c₁ w is k/b.
"""
const BRAITENBERG = (w=1.0, c1=WARREN.c1, δd=WARREN.k / (WARREN.b * WARREN.c1 * 1.0))

"""Heading error ψ = φ − (direction from (x, y) to the origin), in [−π, π)."""
heading_error(x, y, φ) = wrap_angle(φ - atan(-y, -x))

"""Warren's walk to a goal at the origin, φ̈ = −b φ̇ − k ψ (e^{−c₁d} + c₂), state (x, y, φ, φ̇)."""
function warren_approach(p=WARREN)
    function f(u)
        x, y, φ, ω = u
        [p.speed * cos(φ), p.speed * sin(φ), ω,
            -p.b * ω - p.k * heading_error(x, y, φ) * (exp(-p.c1 * hypot(x, y)) + p.c2)]
    end
    BlackBoxContinuousSystem(f, 4)
end

"""Warren's walk past an obstacle at the origin, φ̈ = −b φ̇ + k_o ψ e^{−c₃|ψ|} e^{−c₄d}, state (x, y, φ, φ̇)."""
function warren_avoidance(p=WARREN)
    function f(u)
        x, y, φ, ω = u
        ψ = heading_error(x, y, φ)
        [p.speed * cos(φ), p.speed * sin(φ), ω, -p.b * ω + p.ko * ψ * exp(-p.c3 * abs(ψ)) * exp(-p.c4 * hypot(x, y))]
    end
    BlackBoxContinuousSystem(f, 4)
end

"""
    braitenberg_vehicle(p = BRAITENBERG; crossed = true)

Braitenberg's excitatory vehicle with a stimulus at the origin, in Rañó's reduction, state
(x, y, φ): ẋ = F cos φ, ẏ = F sin φ, φ̇ = ±(δ/d) ∇F⋅ê_p = ∓(δ/d) c₁ F sin ψ. Crossed connections
(Type B) turn it towards the stimulus, uncrossed ones (Type A) away from it.
"""
function braitenberg_vehicle(p=BRAITENBERG; crossed=true)
    turn = crossed ? -1 : 1
    function f(u)
        x, y, φ = u
        F = p.w * exp(-p.c1 * hypot(x, y))
        [F * cos(φ), F * sin(φ), turn * p.δd * p.c1 * F * sin(heading_error(x, y, φ))]
    end
    BlackBoxContinuousSystem(f, 3)
end

# ── CTRNN ───────────────────────────────────────────────────────────────────────────────

"""
    ctrnn_parameters(τ, θ, W; g = ones)

The NamedTuple `(; τ, θ, g, W)` of a CTRNN with time constants `τ`, biases `θ`, gains `g`
and weights `W` (`W[i, j]`: from neuron j to neuron i).
"""
ctrnn_parameters(τ, θ, W; g=ones(length(τ))) = (τ=float.(τ), θ=float.(θ), g=float.(g), W=float.(W))

"""Outputs oᵢ = σ(gᵢ (yᵢ + θᵢ)) of the states `y` of network `net`."""
ctrnn_outputs(net, y) = σ.(net.g .* (y .+ net.θ))

"""
    ctrnn_system(net)

The CTRNN `net` as the controlled system ẏ = (−y + W o(y) + I) ./ τ with input I,
a `BlackBoxControlContinuousSystem`: `vector_field(ctrnn_system(net), y, I)` is ẏ.
"""
function ctrnn_system(net)
    n = length(net.τ)
    BlackBoxControlContinuousSystem((y, I) -> (net.W * ctrnn_outputs(net, y) .+ I .- y) ./ net.τ, n, n)
end

"""
    read_ns(path)

Read a network saved by Beer's C++ `CTRNN::operator<<` (the `.ns` files of his SimpleSearch
package) as `ctrnn_parameters`: the size N, then N time constants, N biases, N gains and the
N × N weights, row i holding the weights from neuron i.
"""
function read_ns(path)
    v = parse.(Float64, split(read(path, String)))
    n = Int(v[1])
    length(v) == 1 + 3n + n^2 || error("$path: expected $(1 + 3n + n^2) numbers for $n neurons, found $(length(v))")
    from_to = permutedims(reshape(v[3n+2:end], n, n))   # [i, j]: i → j, as in the file
    ctrnn_parameters(v[2:n+1], v[n+2:2n+1], permutedims(from_to); g=v[2n+2:3n+1])
end

# ── Beer's visual categorization agent ──────────────────────────────────────────────────

"""Constants of `VisualAgent.h`, `VisualObject.h` and `SimpleVisualMain.cpp`."""
const VISUAL = (body=30.0, max_ray=220.0, input_gain=10.0, visual_angle=π / 6, rays=7, velocity_gain=5.0,
    width=400.0, dt=0.1, fall_speed=3.0, size=30.0, start_height=275.0)

"""
    ray_length(α, ax, shape, oy; p = VISUAL)

Length of the ray at angle `α` from the vertical, from the rim of the agent at (`ax`, 0) to the
object of `shape` centred at (0, `oy`), or `p.max_ray` if it misses (`ResetRay` and the objects'
`RayIntersection` in the C++). A `:circle` has diameter `p.size`, a `:line` is horizontal with
length `p.size`.
"""
function ray_length(α, ax, shape, oy; p=VISUAL)
    sx, sy = ax + p.body / 2 * sin(α), p.body / 2 * cos(α)
    vertical = abs(α) < 1e-7
    m = vertical ? Inf : 1 / tan(α)
    b = sy - m * sx
    if shape == :line
        xi = vertical ? sx : (oy - b) / m
        (abs(xi) > p.size / 2 || oy < 0) && return p.max_ray
        return min(p.max_ray, hypot(xi - sx, oy - sy))
    end
    R = p.size / 2
    if vertical
        abs(sx) > R && return p.max_ray
        return min(p.max_ray, abs(oy - sqrt(R^2 - sx^2) - sy))
    end
    A, B, C = 1 + m^2, 2m * (b - oy), b^2 - 2b * oy + oy^2 - R^2
    disc = B^2 - 4A * C
    disc < 0 && return p.max_ray
    min(p.max_ray, minimum(x -> hypot(x - sx, m * x + b - sy), ((-B + sqrt(disc)) / 2A, (-B - sqrt(disc)) / 2A)))
end

"""Angles of the agent's rays from the vertical, left to right: `p.rays` rays spanning `p.visual_angle`."""
ray_angles(p=VISUAL) = range(-p.visual_angle / 2, p.visual_angle / 2; length=p.rays)

"""
    ray_inputs(ax, shape, oy; lower = 0.0, p = VISUAL)

The agent's sensory inputs, one per ray: `p.input_gain` × (max ray − ray length) / max ray, for
the agent at (`ax`, 0) and the object of `shape` at height `oy`, seen `lower` below that height.
"""
ray_inputs(ax, shape, oy; lower=0.0, p=VISUAL) =
    [p.input_gain * (p.max_ray - ray_length(α, ax, shape, oy - lower; p)) / p.max_ray for α in ray_angles(p)]

"""
    categorization_agent(net, shape; hold = -Inf, path = nothing, lower = 0.0, p = VISUAL)

The agent and a falling `shape` (`:circle` or `:line`) as a `BlackBoxDiscreteSystem` of state
(neuron states, agent position relative to the object, object height). One step is one Euler
step of size `p.dt` of the C++, in its order: rays → sensory inputs, CTRNN step, agent moves with
the new motor outputs, object falls. The velocity is `p.velocity_gain` (o₁₃ − o₁₄). The object
falls until it reaches the height `hold`, then stays there; the agent is held within the
`p.width`-wide environment.

Probes: with `path` (a function of the object's height) the agent's position is imposed,
`path(oy)`, and the network runs open loop; with `lower` the rays see the object `lower` below
its height.
"""
function categorization_agent(net, shape; hold=-Inf, path=nothing, lower=0.0, p=VISUAL)
    n = length(net.τ)
    neural = ctrnn_system(net)
    function step(x)
        y, ax, oy = x[1:n], x[n+1], x[n+2]
        I = zeros(n)
        I[1:p.rays] = ray_inputs(ax, shape, oy; lower, p)
        y = y .+ p.dt .* vector_field(neural, y, I)
        o = ctrnn_outputs(net, y)
        oy = max(hold, oy - p.dt * p.fall_speed)
        ax = path === nothing ? clamp(ax + p.dt * p.velocity_gain * (o[13] - o[14]), -p.width / 2, p.width / 2) : path(oy)
        [y; ax; oy]
    end
    BlackBoxDiscreteSystem(step, n + 2)
end

end
