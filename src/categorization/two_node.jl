# The two-node circuits' simulator: the cells of ActiveCategorizationTwoNodeSearch.jl that the categorization pipeline
# evaluated by cell text (struct Circuit …, STEPS = 868 …, run_circuit, TAU_RANGE/circuit, stable_forces/held_forces),
# lifted verbatim. The notebook's non-constant globals are constants here; the numbers are the same.

"""
    Circuit(τ, θ, g, W, U, L, R)

A CTRNN agent without sensory neurons: N nodes, τᵢ ẏᵢ = −yᵢ + Σₖ Wᵢₖ oₖ + Σⱼ Uᵢⱼ Iⱼ with oᵢ = σ(gᵢ (yᵢ + θᵢ)), all
connected to all (`W` is N × N), and each ray's input Iⱼ weighted straight into every node (`U` is N × 7). The nodes'
outputs set the force F = 5 (o[R] − o[L]) on the agent's body, m v̇ = F − v (`run_circuit`).
"""
struct Circuit
    τ::Vector{Float64}
    θ::Vector{Float64}
    g::Vector{Float64}
    W::Matrix{Float64}
    U::Matrix{Float64}
    L::Int
    R::Int
end

"""Number of nodes of circuit `c`."""
nnodes(c::Circuit) = length(c.τ)

"""Output of node `i` at state `y`."""
node_out(c::Circuit, y, i) = σ(c.g[i] * (y[i] + c.θ[i]))

"""The mirror map of a circuit's nodes: i ↔ N + 1 − i (L ↔ R)."""
node_mirror(c::Circuit) = collect(nnodes(c):-1:1)

"""
    mirror_sums!(dy, c, o, I)

Each node's input Σₖ Wᵢₖ oₖ + Σⱼ Uᵢⱼ Iⱼ, summed in mirror pairs (rays j and 8 − j, nodes k and their mirror images)
in a fixed order. Mirror-image nodes then get bit-identical sums from mirror-image states; with a plain matrix
product the two sides' sums differ in the last bit, which a circuit whose rest is unstable amplifies.
"""
function mirror_sums!(dy, c::Circuit, o, I)
    m = node_mirror(c)
    for i in eachindex(dy)
        acc = 0.0
        for k in eachindex(m)
            if k < m[k]
                acc += c.W[i, k] * o[k] + c.W[i, m[k]] * o[m[k]]
            elseif k == m[k]
                acc += c.W[i, k] * o[k]
            end
        end
        for j in 1:3
            acc += c.U[i, j] * I[j] + c.U[i, 8-j] * I[8-j]
        end
        dy[i] = acc + c.U[i, 4] * I[4]
    end
    dy
end

"""Positions of a trial, 0.1 apart (`VISUAL.dt`): the object falls from 275 until its centre is below the agent's top, height 15 (868 positions, as the player's trials)."""
const STEPS = 868

"""The Euler step of the trials, 0.01: at 0.1 (AgentModels.jl's `VISUAL.dt`) some circuits do the task only because of the step (§1)."""
const DT = 0.01

"""Euler steps from one recorded position to the next (10)."""
const SUBSTEPS = round(Int, VISUAL.dt / DT)

"""Starts of the ordinary trials on one side (the other side is the mirror image)."""
const SIDE = 1.0:50.0

"""The agent's mass m in m v̇ = F − v (§1): 1, as in every fit."""
const MASS = 1.0

"""
    run_circuit(c, shape, d; mass = MASS, hold = -Inf, steps = STEPS, record = false)

Δx over a trial of circuit `c` from rest (v = 0) at offset `d`, with the object `shape` (`:circle` or `:line`), at the
`steps` positions 0.1 apart, each reached by `SUBSTEPS` Euler steps of `DT` = 0.01. Each Euler step is in the order of
`categorization_agent`: the rays and the outputs of the present state, then the nodes take a step (inputs summed by
`mirror_sums!`), the object falls (until it reaches `hold`), and the body of mass `mass` takes a step with the force
F = 5 (o_R − o_L) of the new outputs: v ← F + (v − F) e^{−DT/m}, the exact step of m v̇ = F − v for F held over the
step (v = F at m = 0), then Δx ← Δx + DT v, stopping (v = 0) at the world's edges. With `record` it also returns the
velocity `v`, the rays' inputs `I` (7 × steps) and the node states `Y` (N × steps), at the same positions.
"""
function run_circuit(c::Circuit, shape, d; mass=MASS, hold=-Inf, steps=STEPS, record=false)
    N, p = nnodes(c), VISUAL
    y, o, I, dy = zeros(N), zeros(N), zeros(7), zeros(N)
    decay = exp(-DT / mass)
    ax, oy, v = Float64(d), p.start_height, 0.0
    xs = zeros(steps)
    xs[1] = ax
    Is, Y, V = record ? (zeros(7, steps), zeros(N, steps), zeros(steps)) : (nothing, nothing, nothing)
    angles = ray_angles(p)
    for k in 1:steps
        for s in 1:SUBSTEPS
            for j in 1:7
                I[j] = p.input_gain * (p.max_ray - ray_length(angles[j], ax, shape, oy; p)) / p.max_ray
            end
            record && s == 1 && (Is[:, k] .= I)
            k == steps && break
            for i in 1:N
                o[i] = node_out(c, y, i)
            end
            mirror_sums!(dy, c, o, I)
            y .+= DT .* (dy .- y) ./ c.τ
            oy = max(hold, oy - DT * p.fall_speed)
            F = p.velocity_gain * (node_out(c, y, c.R) - node_out(c, y, c.L))
            v = F + (v - F) * decay
            ax += DT * v
            if abs(ax) > p.width / 2
                ax, v = clamp(ax, -p.width / 2, p.width / 2), 0.0
            end
        end
        k == steps && break
        xs[k+1] = ax
        record && (Y[:, k+1] .= y; V[k+1] = v)
    end
    record ? (; x=xs, v=V, I=Is, Y) : xs
end

"""Bounds on the time constants (the fit's)."""
const TAU_RANGE = (0.1, 50.0)

"""
    circuit(q)

The mirror-symmetric 2-node circuit of the free vector `q` (11 numbers): the log of the nodes' time constant τ, their
bias θ, the self weight W_LL = W_RR, the cross weight W_LR = W_RL and the ray weights U_L1 … U_L7 (U_Rj = U_L,8−j).
Node gains are 1.
"""
function circuit(q)
    τ = clamp(exp(q[1]), TAU_RANGE...)
    U = collect(q[5:11])
    Circuit([τ, τ], [q[2], q[2]], [1.0, 1.0], [q[3] q[4]; q[4] q[3]], [U'; reverse(U)'], 1, 2)
end

"""
Starts of the searches for stable states besides y = 0: the nodes' outputs o on a 5 × 5 grid from 0 to 1, each
turned into the state y = W o + E it drives (E the rays' drive).
"""
const SETTLE_STARTS = [(a, b) for a in 0:0.25:1 for b in 0:0.25:1]

"""
    settle(c, E, y)

Newton's method for a settled state of the 2-node circuit `c`, yᵢ = Σₖ Wᵢₖ oₖ + Eᵢ with the rays' drive `E`
(a pair) held fixed, from the state `y` (a pair), halving a step until it lowers the residual. Returns the state
and the largest residual left.
"""
function settle(c::Circuit, E, y)
    W11, W12, W21, W22 = c.W[1, 1], c.W[1, 2], c.W[2, 1], c.W[2, 2]
    (θ1, θ2), (g1, g2) = c.θ, c.g
    residual(y1, y2) = (u1 = σ(g1 * (y1 + θ1)); u2 = σ(g2 * (y2 + θ2));
        (W11 * u1 + W12 * u2 + E[1] - y1, W21 * u1 + W22 * u2 + E[2] - y2))
    y1, y2 = y
    r1, r2 = residual(y1, y2)
    for _ in 1:100
        r0 = max(abs(r1), abs(r2))
        r0 < 1e-12 && break
        o1, o2 = σ(g1 * (y1 + θ1)), σ(g2 * (y2 + θ2))
        d1, d2 = g1 * o1 * (1 - o1), g2 * o2 * (1 - o2)
        a, b, cc, d = W11 * d1 - 1, W12 * d2, W21 * d1, W22 * d2 - 1
        det = a * d - b * cc
        δ1, δ2 = (d * r1 - b * r2) / det, (a * r2 - cc * r1) / det
        t = 1.0
        s1, s2 = residual(y1 - δ1, y2 - δ2)
        while t > 1e-8 && max(abs(s1), abs(s2)) >= r0
            t /= 2
            s1, s2 = residual(y1 - t * δ1, y2 - t * δ2)
        end
        y1, y2, r1, r2 = y1 - t * δ1, y2 - t * δ2, s1, s2
    end
    ((y1, y2), max(abs(r1), abs(r2)))
end

"""Whether the state `y` of the 2-node circuit `c` is stable: the Jacobian W D − 1 (D the outputs' slopes) has a negative trace and a positive determinant."""
function is_stable(c::Circuit, y)
    o1, o2 = node_out(c, y, 1), node_out(c, y, 2)
    d1, d2 = c.g[1] * o1 * (1 - o1), c.g[2] * o2 * (1 - o2)
    a, b, cc, d = c.W[1, 1] * d1 - 1, c.W[1, 2] * d2, c.W[2, 1] * d1, c.W[2, 2] * d2 - 1
    a + d < 0 && a * d - b * cc > 0
end

"""
    stable_forces(c, E)

The forces F = 5 (o_R − o_L) of the stable settled states of circuit `c` under the rays' drive `E` (a pair): the
smallest and the largest, equal where the state is unique. Newton from y = 0 and from each of `SETTLE_STARTS`.
"""
function stable_forces(c::Circuit, E)
    p = VISUAL
    starts = SETTLE_STARTS::Vector{NTuple{2,Float64}}
    lo, hi = Inf, -Inf
    for k in 0:length(starts)
        a, b = k == 0 ? (0.0, 0.0) : starts[k]
        y0 = k == 0 ? (0.0, 0.0) : (c.W[1, 1] * a + c.W[1, 2] * b + E[1], c.W[2, 1] * a + c.W[2, 2] * b + E[2])
        y, r = settle(c, E, y0)
        (r < 1e-9 && is_stable(c, y)) || continue
        F = p.velocity_gain * (node_out(c, y, c.R) - node_out(c, y, c.L))
        lo, hi = min(lo, F), max(hi, F)
    end
    (lo, hi)
end

"""
    held_forces(c, shape, rel, h)

`stable_forces` with the agent held still and the object `shape` held with its centre at height `h` and `rel` to
the side of the agent (`rel` = −Δx, < 0 on the agent's left). `(NaN, NaN)` where no ray sees the object.
"""
function held_forces(c::Circuit, shape, rel, h)
    p = VISUAL
    I = [p.input_gain * (p.max_ray - ray_length(α, -rel, shape, h; p)) / p.max_ray for α in ray_angles(p)]
    all(iszero, I) && return (NaN, NaN)
    stable_forces(c, mirror_sums!(zeros(2), c, zeros(2), I))
end
