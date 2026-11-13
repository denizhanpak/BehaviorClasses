"""
    NNodeModel

The larger agents' simulator: bilaterally symmetric CTRNN agents of N = 2 … 8 nodes (the evolution's model, nnode/nmodel.jl,
on the two-node task, body and rays of the refit's refit/model.jl at dt 0.01, mass 1), and the rest-mode study of each
agent (nnode/rest_modes.jl). Lifted verbatim; the evolution's losses, searches and the refit's two-node fit code are left
out (the evolved agents and the refit are root inputs). A module of its own because it has its own `VISUAL` (no `dt`),
`σ`, `DT` and ray reader (`ray_len`, equal to AgentModels' `ray_length` to the last bit on these trials).
"""
module NNodeModel

using Statistics, LinearAlgebra, Random, Printf, Base.Threads
using ForwardDiff

export NCircuit, ncircuit, npath, final_dx, nsteps

# ── refit/model.jl: the task, body and rays ──────────────────────────────────────────────────────────────────────────
# The 2-node circuit and its task at a chosen Euler step. Same equations as the notebooks' scalar `final_dx` (mass 1)
# with the step `dt` and the step count as arguments: a trial always lasts 86.7 time units (867 steps of 0.1, 8670 of
# 0.01), the object's centre ending at height 15.

σ(x) = 1 / (1 + exp(-x))
const VISUAL = (body=30.0, max_ray=220.0, input_gain=10.0, visual_angle=π / 6, rays=7, velocity_gain=5.0,
    width=400.0, fall_speed=3.0, size=30.0, start_height=275.0)
const T_TRIAL = 86.7
nsteps(dt) = round(Int, T_TRIAL / dt)
const DT = 0.01
const SIDE = 1.0:50.0
const MASS = 1.0
const TAU_RANGE = (0.1, 50.0)

ray_angles(p=VISUAL) = range(-p.visual_angle / 2, p.visual_angle / 2; length=p.rays)
const RAYS = map(Tuple(ray_angles(VISUAL))) do α
    vertical = abs(α) < 1e-7
    (; ox=VISUAL.body / 2 * sin(α), sy=VISUAL.body / 2 * cos(α), vertical, m=vertical ? Inf : 1 / tan(α))
end

@inline function ray_len(r, ax, line::Bool, oy)
    p = VISUAL
    sx, sy, m, vertical = ax + r.ox, r.sy, r.m, r.vertical
    b = sy - m * sx
    if line
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
    x1, x2 = (-B + sqrt(disc)) / 2A, (-B - sqrt(disc)) / 2A
    min(p.max_ray, min(hypot(x1 - sx, m * x1 + b - sy), hypot(x2 - sx, m * x2 + b - sy)))
end
@inline ray_inputs7(ax, line, oy) =
    map(r -> VISUAL.input_gain * (VISUAL.max_ray - ray_len(r, ax, line, oy)) / VISUAL.max_ray, RAYS)
@inline rays_out_of_reach(oy) = oy - VISUAL.size / 2 - VISUAL.body / 2 > VISUAL.max_ray + 1


# ── nnode/nmodel.jl ──────────────────────────────────────────────────────────────────────────────────────────────────
# Bilaterally symmetric CTRNN agents of N = 2 … 8 nodes without sensory neurons (user, 2026-10-21). The two-node
# circuit's task, body and rays (refit/model.jl, dt 0.01, mass 1) with a bigger brain:
#   - m = N ÷ 2 mirror pairs (Lᵢ, Rᵢ); L₁ and R₁ drive the motor, v̇ = 5(o_R₁ − o_L₁) − v; the other pairs are interneurons;
#   - odd N adds one midline interneuron C, coupled symmetrically to both sides (C→Lᵢ = C→Rᵢ, Lᵢ→C = Rᵢ→C, and its ray
#     weights are mirror-symmetric, U_Cj = U_C,8−j);
#   - every node hears every node (itself included) and all 7 rays; each pair, and C, has its own τ and θ.
# Free vector (N = 2 is the two-node `q`: log τ, θ, W_s, W_c, U₁ … U₇):
#   [log τᵢ, θᵢ for i = 1…m; W_s (m×m, Lₖ→Lᵢ = Rₖ→Rᵢ); W_c (m×m, Rₖ→Lᵢ = Lₖ→Rᵢ); U (m×7, ray j → Lᵢ = ray 8−j → Rᵢ);
#    if C: log τ_C, θ_C, W_CC, W_iC (C→Lᵢ, m), W_Ci (Lᵢ→C, m), U_C (rays 1&7, 2&6, 3&5, 4)]
# 3 … 8 nodes: 20, 26, 37, 45, 58, 68 free numbers. Node inputs are summed in mirror pairs, so a mirror-image state
# gives bit-identical sums and Δx(−d) = −Δx(d) exactly.


struct NCircuit{M,C}
    τ::NTuple{M,Float64}
    θ::NTuple{M,Float64}
    Ws::NTuple{M,NTuple{M,Float64}}   # Ws[i][k]: Lₖ → Lᵢ
    Wc::NTuple{M,NTuple{M,Float64}}   # Wc[i][k]: Rₖ → Lᵢ
    U::NTuple{M,NTuple{7,Float64}}    # U[i][j]: ray j → Lᵢ
    τC::Float64
    θC::Float64
    WCC::Float64
    WiC::NTuple{M,Float64}            # C → Lᵢ
    WCi::NTuple{M,Float64}            # Lᵢ → C
    UC::NTuple{4,Float64}             # rays (1, 7), (2, 6), (3, 5), 4 → C
end

pairs_of(N) = N ÷ 2
has_centre(N) = isodd(N)
nparams(N) = (m = pairs_of(N); 9m + 2m^2 + (has_centre(N) ? 7 + 2m : 0))
tau_of(z) = clamp(exp(z), TAU_RANGE...)

function ncircuit(q, N)
    M, C = pairs_of(N), has_centre(N)
    length(q) == nparams(N) || error("expected $(nparams(N)) numbers for N = $N, got $(length(q))")
    τ = ntuple(i -> tau_of(q[2i-1]), M)
    θ = ntuple(i -> q[2i], M)
    o = 2M
    Ws = ntuple(i -> ntuple(k -> q[o+(i-1)*M+k], M), M); o += M^2
    Wc = ntuple(i -> ntuple(k -> q[o+(i-1)*M+k], M), M); o += M^2
    U = ntuple(i -> ntuple(j -> q[o+(i-1)*7+j], 7), M); o += 7M
    if C
        τC, θC, WCC = tau_of(q[o+1]), q[o+2], q[o+3]; o += 3
        WiC = ntuple(i -> q[o+i], M); o += M
        WCi = ntuple(i -> q[o+i], M); o += M
        UC = ntuple(j -> q[o+j], 4)
    else
        τC, θC, WCC, WiC, WCi, UC = 1.0, 0.0, 0.0, ntuple(_ -> 0.0, M), ntuple(_ -> 0.0, M), ntuple(_ -> 0.0, 4)
    end
    NCircuit{M,C}(τ, θ, Ws, Wc, U, τC, θC, WCC, WiC, WCi, UC)
end

"""Where each block of the free vector sits, for reading a fit (`:logτ`, `:θ`, `:Ws`, … as index ranges)."""
function layout(N)
    M = pairs_of(N); o = 2M
    L = Dict{Symbol,Any}(:logτ => 1:2:2M, :θ => 2:2:2M, :Ws => o+1:o+M^2, :Wc => o+M^2+1:o+2M^2, :U => o+2M^2+1:o+2M^2+7M)
    o += 2M^2 + 7M
    if has_centre(N)
        L[:logτC], L[:θC], L[:WCC] = o + 1, o + 2, o + 3
        L[:WiC], L[:WCi], L[:UC] = o+4:o+3+M, o+4+M:o+3+2M, o+4+2M:o+7+2M
    end
    L
end
logtau_index(N) = (L = layout(N); has_centre(N) ? [collect(L[:logτ]); L[:logτC]] : collect(L[:logτ]))

# One Euler step of the nodes: every input summed in mirror pairs (L and R get the same pair sums in the same order).
@inline function node_step(c::NCircuit{M,C}, yL, yR, yC, oL, oR, oC, I, dt) where {M,C}
    yL2 = ntuple(Val(M)) do i
        a = 0.0
        for k in 1:M
            a += c.Ws[i][k] * oL[k] + c.Wc[i][k] * oR[k]
        end
        C && (a += c.WiC[i] * oC)
        u = c.U[i]
        for j in 1:3
            a += u[j] * I[j] + u[8-j] * I[8-j]
        end
        yL[i] + dt * (a + u[4] * I[4] - yL[i]) / c.τ[i]
    end
    yR2 = ntuple(Val(M)) do i
        a = 0.0
        for k in 1:M
            a += c.Wc[i][k] * oL[k] + c.Ws[i][k] * oR[k]
        end
        C && (a += c.WiC[i] * oC)
        u = c.U[i]
        for j in 1:3
            a += u[8-j] * I[j] + u[j] * I[8-j]
        end
        yR[i] + dt * (a + u[4] * I[4] - yR[i]) / c.τ[i]
    end
    yC2 = yC
    if C
        aC = c.WCC * oC
        for k in 1:M
            aC += c.WCi[k] * (oL[k] + oR[k])
        end
        for j in 1:3
            aC += c.UC[j] * (I[j] + I[8-j])
        end
        yC2 = yC + dt * (aC + c.UC[4] * I[4] - yC) / c.τC
    end
    yL2, yR2, yC2
end
@inline outputs(c::NCircuit{M}, yL, yR, yC) where {M} =
    (ntuple(i -> σ(yL[i] + c.θ[i]), Val(M)), ntuple(i -> σ(yR[i] + c.θ[i]), Val(M)), σ(yC + c.θC))

"""Final Δx of one trial (the two-node `final_dx` with N nodes); every node starts at y = 0. `record` keeps Δx per step."""
function final_dx(c::NCircuit{M,C}, line::Bool, d::Float64; dt=DT, record=nothing) where {M,C}
    p = VISUAL
    decay = exp(-dt / MASS)
    yL = yR = ntuple(_ -> 0.0, Val(M)); yC = 0.0
    v = 0.0
    ax, oy = d, p.start_height
    oL, oR, oC = outputs(c, yL, yR, yC)
    for s in 1:nsteps(dt)
        I = rays_out_of_reach(oy) ? ntuple(_ -> 0.0, 7) : ray_inputs7(ax, line, oy)
        yL, yR, yC = node_step(c, yL, yR, yC, oL, oR, oC, I, dt)
        oy -= dt * p.fall_speed
        oL, oR, oC = outputs(c, yL, yR, yC)
        F = p.velocity_gain * (oR[1] - oL[1])
        v = F + (v - F) * decay
        ax += dt * v
        if abs(ax) > p.width / 2
            ax, v = clamp(ax, -p.width / 2, p.width / 2), 0.0
        end
        record === nothing || (record[s] = ax)
    end
    ax
end

nfinal_dxs(q, N; dt=DT) = (c = ncircuit(q, N); [[final_dx(c, false, d; dt) for d in SIDE]; [final_dx(c, true, d; dt) for d in SIDE]])

"""Δx over the trial (one value per step) for one start."""
function npath(q, N, line, d; dt=DT)
    rec = zeros(nsteps(dt)); final_dx(ncircuit(q, N), line, Float64(d); dt, record=rec); rec
end

# ── nnode/rest_modes.jl: the rest-mode study (run by pipeline/fit_categorization.jl, step rest_modes) ─────────────────────────────
# Does the two-node action modes' split (latched / symmetric rest with nothing in view) hold for N = 3 … 8?
# Three looks at each evolved agent, and at the 41 two-node circuits (refit/out/final.jls) as the check:
#   attractors       every rest the nodes reach with nothing in view, from 162 starts (exact symmetry y = 0, the small
#                    nudge, 100 on/off corners, 60 uniform), run T_SETTLE then watched for T_WATCH: fixed or cycling,
#                    mirror-symmetric or not, mean motor force;
#   symmetric rests  every rest on the mirror plane y_L = y_R, by Newton from a grid of net inputs, with the Jacobian
#                    split into the even block (pair sums and C) and the odd block (pair differences; C drops out):
#                    odd block = T⁻¹(−I + (W_s − W_c) diag σ′). For two nodes it is the odd gain (W_s − W_c)σ′ of
#                    CategorizationStrategies.jl, latched ⇔ > 1;
#   trials           the motor force 10 time units after the object leaves view (the two-node visual tell: latched
#                    |v| 1.23–5.0, symmetric ≤ 0.20).

const T_SETTLE = 1000.0
const T_WATCH = 200.0
const F_LATCH = 0.05      # |mean F| above this: latched
const SPREAD = 1e-4       # output range over the watch above this: cycling
const ASYM = 1e-3         # max |y_L − y_R| over the watch above this: mirror-asymmetric

state_tuple(yL, yR, yC) = (yL..., yR..., yC)

"""Settle with nothing in view from (yL, yR, yC), then watch: mean and range of F, output range, mirror asymmetry,
period of the motor node L₁ (median gap between upward crossings of its mean; NaN if fixed)."""
function settle(c::NCircuit{M,C}, y0; dt=DT) where {M,C}
    yL, yR, yC = y0
    I0 = ntuple(_ -> 0.0, 7)
    for _ in 1:round(Int, T_SETTLE / dt)
        oL, oR, oC = outputs(c, yL, yR, yC)
        yL, yR, yC = node_step(c, yL, yR, yC, oL, oR, oC, I0, dt)
    end
    n = round(Int, T_WATCH / dt)
    Fs = zeros(n); m1 = zeros(n)
    lo = hi = state_tuple(outputs(c, yL, yR, yC)...)
    asym = 0.0
    for s in 1:n
        oL, oR, oC = outputs(c, yL, yR, yC)
        yL, yR, yC = node_step(c, yL, yR, yC, oL, oR, oC, I0, dt)
        oL, oR, oC = outputs(c, yL, yR, yC)
        z = state_tuple(oL, oR, oC)
        lo, hi = min.(lo, z), max.(hi, z)
        Fs[s] = VISUAL.velocity_gain * (oR[1] - oL[1]); m1[s] = oL[1]
        asym = max(asym, maximum(abs.(yL .- yR)))
    end
    spread = maximum(hi .- lo)
    period = NaN
    if spread > SPREAD
        μ1 = mean(m1)
        ups = [s for s in 2:n if m1[s-1] < μ1 <= m1[s]]
        length(ups) >= 3 && (period = median(diff(ups)) * dt)
    end
    meanF = mean(Fs)
    label = spread <= SPREAD ? (abs(meanF) > F_LATCH ? "L" : asym > ASYM ? "A" : "S") :
            (abs(meanF) > F_LATCH ? "L~" : asym > ASYM ? "W" : "S~")
    (; label, meanF, ampF=maximum(Fs) - minimum(Fs), spread, asym, period)
end

"""The 162 starts: y = 0 (exact symmetry, the trial start), the nudge of `nrest_force`, 100 corners (every node fully
on or off, y = −θ ± 10, as `nrest_latch`), 60 uniform (y = −θ + U(−6, 6) per node)."""
function rest_starts(c::NCircuit{M,C}; rng=MersenneTwister(1)) where {M,C}
    z = ntuple(_ -> 0.0, M)
    starts = Any[(z, z, 0.0), (ntuple(_ -> 0.1, M), ntuple(_ -> -0.1, M), 0.0)]
    corner(θ) = -θ + (rand(rng, Bool) ? 10.0 : -10.0)
    unif(θ) = -θ + 12 * rand(rng) - 6
    for _ in 1:100
        push!(starts, (ntuple(i -> corner(c.θ[i]), M), ntuple(i -> corner(c.θ[i]), M), corner(c.θC)))
    end
    for _ in 1:60
        push!(starts, (ntuple(i -> unif(c.θ[i]), M), ntuple(i -> unif(c.θ[i]), M), unif(c.θC)))
    end
    starts
end

"""Distinct attractors (label, sign of F, |mean F| and F range to 0.05) with how many starts reach each, and how many
of those were off the mirror plane (`n_off`; a rest reached only from mirror-exact starts, y = 0 among them, may be
unstable across the plane); `from_zero` is the one the trial start y = 0 reaches."""
function attractors(c)
    starts = rest_starts(c)
    res = [settle(c, y0) for y0 in starts]
    on_plane(y0) = y0[1] == y0[2]
    key(r) = (r.label, round(abs(r.meanF) / 0.05) * 0.05, round(r.ampF / 0.05) * 0.05)
    groups = Dict{Any,Vector{Int}}()
    for (k, r) in enumerate(res)
        push!(get!(groups, key(r), Int[]), k)
    end
    list = [(; label=kk[1], absF=kk[2], ampF=kk[3], n=length(ks), n_off=count(k -> !on_plane(starts[k]), ks),
                period=res[ks[1]].period, asym=maximum(res[k].asym for k in ks))
            for (kk, ks) in groups]
    sort!(list; by=x -> -x.n)
    (; list, from_zero=res[1], from_nudge=res[2])
end

# Rests on the mirror plane: x = (y₁ … y_M, y_C), y_Lᵢ = y_Rᵢ = yᵢ.
function plane_rhs(c::NCircuit{M,C}, x) where {M,C}
    o = [σ(x[k] + c.θ[k]) for k in 1:M]
    oC = C ? σ(x[M+1] + c.θC) : zero(eltype(x))
    r = [-x[i] + sum((c.Ws[i][k] + c.Wc[i][k]) * o[k] for k in 1:M) + (C ? c.WiC[i] * oC : 0.0) for i in 1:M]
    C ? [r; -x[M+1] + c.WCC * oC + 2 * sum(c.WCi[k] * o[k] for k in 1:M)] : r
end

"""Every rest on the mirror plane: damped Newton from a grid of net inputs u = y + θ ∈ {−8, −3, −1, 0, 1, 3, 8} per
plane coordinate, merged to 1e-6."""
function plane_rests(c::NCircuit{M,C}) where {M,C}
    θs = C ? [collect(c.θ); c.θC] : collect(c.θ)
    nd = length(θs); grid = (-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0)
    found = Vector{Vector{Float64}}()
    for idx in Iterators.product(ntuple(_ -> grid, nd)...)
        x = collect(idx) .- θs
        for _ in 1:100
            f = plane_rhs(c, x)
            maximum(abs, f) < 1e-11 && break
            J = ForwardDiff.jacobian(z -> plane_rhs(c, z), x)
            dx = try J \ f catch; break end
            step = 1.0
            while step > 1e-4 && maximum(abs, plane_rhs(c, x .- step .* dx)) > maximum(abs, f)
                step /= 2
            end
            x = x .- step .* dx
        end
        maximum(abs, plane_rhs(c, x)) < 1e-9 || continue
        any(y -> maximum(abs, y .- x) < 1e-6, found) || push!(found, x)
    end
    [plane_blocks(c, x) for x in found]
end

"""Even and odd Jacobian blocks at a mirror-plane rest, their eigenvalues, and the odd loop gain G = (W_s − W_c) diag σ′
(the two-node odd gain for M = 1); `motor_gain` is the motor pair's own (W_s₁₁ − W_c₁₁)σ′₁."""
function plane_blocks(c::NCircuit{M,C}, x) where {M,C}
    o = [σ(x[k] + c.θ[k]) for k in 1:M]; d = o .* (1 .- o)
    T = [c.τ...]
    G = [(c.Ws[i][k] - c.Wc[i][k]) * d[k] for i in 1:M, k in 1:M]
    Aodd = (-I(M) + G) ./ T
    S = [(c.Ws[i][k] + c.Wc[i][k]) * d[k] for i in 1:M, k in 1:M]
    if C
        oC = σ(x[M+1] + c.θC); dC = oC * (1 - oC)
        Aeven = [(-I(M) + S) ./ T  [c.WiC[i] * dC / c.τ[i] for i in 1:M];
                 [2 * c.WCi[k] * d[k] / c.τC for k in 1:M]'  (-1 + c.WCC * dC) / c.τC]
    else
        Aeven = (-I(M) + S) ./ T
    end
    λo, λe = eigvals(Aodd), eigvals(Aeven)
    lead_odd = λo[argmax(real.(λo))]
    vodd = eigen(Aodd).vectors[:, argmax(real.(λo))]
    (; x, o, oC=C ? σ(x[M+1] + c.θC) : NaN, λodd=λo, λeven=λe, lead_odd, lead_even=λe[argmax(real.(λe))],
        odd_gain=maximum(real, eigvals(G)), det_odd=det(I(M) - G), motor_gain=G[1, 1],
        motor_share=abs(vodd[1]) / norm(vodd))
end

"""Trial (same steps as `final_dx`) with the node state kept at every leave of view (the first unseen step after the
object was seen): (state, force 10 later in the trial or NaN, how long the object then stays out of view)."""
function trial_leaves(c::NCircuit{M,C}, line::Bool, d::Float64; dt=DT, after=10.0) where {M,C}
    p = VISUAL; decay = exp(-dt / MASS)
    yL = yR = ntuple(_ -> 0.0, Val(M)); yC = 0.0
    v = 0.0; ax, oy = d, p.start_height
    oL, oR, oC = outputs(c, yL, yR, yC)
    n = nsteps(dt); Fs = zeros(n); seen = falses(n); states = Vector{Any}(undef, n)
    for s in 1:n
        Iv = rays_out_of_reach(oy) ? ntuple(_ -> 0.0, 7) : ray_inputs7(ax, line, oy)
        seen[s] = any(>(0), Iv)
        states[s] = (yL, yR, yC)
        yL, yR, yC = node_step(c, yL, yR, yC, oL, oR, oC, Iv, dt)
        oy -= dt * p.fall_speed
        oL, oR, oC = outputs(c, yL, yR, yC)
        F = p.velocity_gain * (oR[1] - oL[1]); Fs[s] = F
        v = F + (v - F) * decay
        ax += dt * v
        if abs(ax) > p.width / 2
            ax, v = clamp(ax, -p.width / 2, p.width / 2), 0.0
        end
    end
    k = round(Int, after / dt); out = []
    for s in 2:n
        seen[s-1] && !seen[s] || continue
        back = findnext(seen, s)
        push!(out, (; state=states[s], F10=s + k <= n && !any(view(seen, s:s+k)) ? Fs[s+k] : NaN,
            out_for=((back === nothing ? n + 1 : back) - s) * dt, line))
    end
    out
end

"""From a leave state, nothing in view for good: |F| at 10, 30 and 100 later, the hold (time until |F| < 0.5 for good,
within 1000) and the rest it ends in (`settle`)."""
function after_leave(c::NCircuit{M,C}, y0; dt=DT) where {M,C}
    yL, yR, yC = y0
    I0 = ntuple(_ -> 0.0, 7)
    marks = Dict(round(Int, t / dt) => t for t in (10.0, 30.0, 100.0)); at = Dict{Float64,Float64}()
    last_strong = 0
    for s in 1:round(Int, 1000 / dt)
        oL, oR, oC = outputs(c, yL, yR, yC)
        yL, yR, yC = node_step(c, yL, yR, yC, oL, oR, oC, I0, dt)
        oL, oR, _ = outputs(c, yL, yR, yC)
        F = VISUAL.velocity_gain * (oR[1] - oL[1])
        abs(F) >= 0.5 && (last_strong = s)
        haskey(marks, s) && (at[marks[s]] = abs(F))
    end
    (; F10=at[10.0], F30=at[30.0], F100=at[100.0], hold=last_strong * dt, rest=settle(c, (yL, yR, yC); dt).label)
end

function leaves_study(c)
    lv = vcat([trial_leaves(c, line, Float64(d)) for line in (false, true) for d in SIDE]...)
    [(; l..., after_leave(c, l.state)...) for l in lv]
end

function study(q, N)
    c = ncircuit(q, N)
    (; att=attractors(c), plane=plane_rests(c), leaves=leaves_study(c))
end

fmtλ(z) = abs(imag(z)) < 1e-9 ? @sprintf("%+.3f", real(z)) : @sprintf("%+.3f±%.3fi", real(z), abs(imag(z)))
"""The long report of `study`'s results (printed; returned as a string)."""
function report(res)
    io = IOBuffer()
    println(io, "rest attractors with nothing in view (starts reaching it / 162): S symmetric fixed, L latched fixed, A asymmetric fixed with F ≈ 0,")
    println(io, "S~ symmetric cycle (F ≡ 0), W cycle with mean F ≈ 0 that swings F, L~ latched cycle; |F| mean, F range, period")
    println(io, "mirror-plane rests: stable / odd-unstable / even-unstable, leading odd and even eigenvalue, odd loop gain, motor pair's own gain, motor share of the odd mode")
    println(io, "trials: leaves of view held ≥ 10, median |F| 10 after leaving, share with |F| > 0.5\n")
    for r in res
        @printf(io, "%-8s %-9s %-6s  ", r.name, r.mode, r.family)
        println(io, join([@sprintf("%s %d%s (|F| %.2f, range %.2f%s)", a.label, a.n, a.n_off < a.n ? @sprintf(" [%d off plane]", a.n_off) : "",
                              a.absF, a.ampF, isnan(a.period) ? "" : @sprintf(", T %.2f", a.period))
                          for a in r.att.list], "; "), "   from y=0: ", r.att.from_zero.label)
        for p in r.plane
            st = real(p.lead_odd) < 0 && real(p.lead_even) < 0 ? "stable" : real(p.lead_odd) >= 0 ? "odd-unstable" : "even-unstable"
            @printf(io, "           plane rest %-13s odd %-18s even %-18s odd gain %6.3f  motor gain %6.3f  motor share %.2f\n",
                st, fmtλ(p.lead_odd), fmtλ(p.lead_even), p.odd_gain, p.motor_gain, p.motor_share)
        end
        L = r.leaves
        if isempty(L)
            println(io, "           leaves: none")
        else
            ends = Dict{String,Int}()
            for l in L
                ends[l.rest] = get(ends, l.rest, 0) + 1
            end
            inT = filter(!isnan, [abs(l.F10) for l in L])
            @printf(io, "           leaves %d (lines %d): in trial |F|>0.5 at +10 %s;  nothing in view: |F| at 10/30/100 median %.2f/%.2f/%.2f, hold median %.1f max %.1f, end in %s\n",
                length(L), count(l -> l.line, L), isempty(inT) ? "n/a" : @sprintf("%d/%d", count(>(0.5), inT), length(inT)),
                median(l.F10 for l in L), median(l.F30 for l in L), median(l.F100 for l in L), median(l.hold for l in L),
                maximum(l.hold for l in L), join(["$k $v" for (k, v) in sort(collect(ends))], ", "))
        end
    end
    s = String(take!(io)); print(s)
    s
end

"""Plane rest kind: stable, `pitchfork` (odd block has a real eigenvalue > 0), `odd Hopf` (only complex ones > 0),
`even` (odd block stable, even block not)."""
plane_kind(p) = real(p.lead_odd) > 0 ? (any(z -> real(z) > 0 && abs(imag(z)) < 1e-9, p.λodd) ? "pitchfork" : "odd Hopf") :
                real(p.lead_even) > 0 ? "even" : "stable"

"""Rest class from the attractors that are reached from off the mirror plane (and S only when a plane rest is stable):
symmetric (S only), latched (L or L~, no stable S), bistable (S and L/L~), wobble (W), with `+cycle` when the latched
attractor is L~."""
function rest_class(r)
    real_att = filter(a -> a.n_off > 0 && (a.label != "S" || any(p -> plane_kind(p) == "stable", r.plane)), r.att.list)
    labs = Set(a.label for a in real_att)
    lat = "L" in labs || "L~" in labs
    cls = "W" in labs ? (lat ? "wobble+latched" : "wobble") : lat ? ("S" in labs ? "bistable" : "latched") : "symmetric"
    ("L~" in labs) && (cls *= " (L~)")
    cls, labs
end

"""One line per agent: rest class, plane rests, where the leaves end, the hold (printed; returned as a string)."""
function tabulate(res)
    io = IOBuffer()
    println(io, "rest class: attractors reached from off the mirror plane (S counted only with a stable plane rest)")
    println(io, "plane: kind of every mirror-plane rest (pitchfork = real odd eigenvalue > 0, odd Hopf = complex only, even = even block unstable)")
    println(io, "leaves: where the trial's leave states go with nothing in view for good; hold = time |F| stays ≥ 0.5 (median, max)\n")
    @printf(io, "%-8s %-9s %-6s  %-24s %-34s %-22s %s\n", "agent", "old mode", "family", "rest class", "plane rests", "leaves end in", "hold med / max")
    for r in res
        cls, _ = rest_class(r)
        kinds = join([plane_kind(p) for p in r.plane], ", ")
        ends = Dict{String,Int}()
        foreach(l -> ends[l.rest] = get(ends, l.rest, 0) + 1, r.leaves)
        endstr = join(["$k $v" for (k, v) in sort(collect(ends))], ", ")
        holds = [l.hold for l in r.leaves]
        @printf(io, "%-8s %-9s %-6s  %-24s %-34s %-22s %s\n", r.name, r.mode, r.family, cls, kinds, endstr,
            isempty(holds) ? "-" : @sprintf("%.1f / %.1f", median(holds), maximum(holds)))
    end
    s = String(take!(io)); print(s)
    s
end

end # module NNodeModel
