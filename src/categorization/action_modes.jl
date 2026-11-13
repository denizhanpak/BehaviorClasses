# Each circuit's action mode and the order of the §7 figures, from CategorizationStrategies.jl's cells that
# te_work/category/cat_meta.jl evaluated (the rest-state solvers, REST, ACTION_MODES, RELIEF_PEAKS / RELIEF_ORDER),
# verbatim; the notebook's globals CIRCUITS and RELIEF are arguments here. The family rule is cat_meta.jl's.

"""Slope of the logistic function, σ′(x) = σ(x) (1 − σ(x))."""
dσ(x) = (s = σ(x); s * (1 - s))

"""The bias θ = u − S σ(u) of the circuit whose symmetric rest has net input `u` and whose even weight is `S`."""
bias_of(u, S) = u - S * σ(u)

"""
Whether the rest state with net inputs `x1`, `x2` of the circuit with weights `Ws`, `Wc` is stable: the Jacobian
W diag(σ′(x₁), σ′(x₂)) − I has a negative trace and a positive determinant (its eigenvalues are real).
"""
function is_stable_rest(Ws, Wc, x1, x2)
    s1, s2 = dσ(x1), dσ(x2)
    a, b, c, e = Ws * s1 - 1, Wc * s2, Wc * s1, Ws * s2 - 1
    a + e < 0 && a * e - b * c > 0
end

"""
    mean_input(θ, S, d, m0)

The mean net input m that solves m − S ō(m, d) = θ for the half-difference `d` (one solution when S ≤ 4), by
Newton's method from `m0`, kept inside the bracket θ + S·[0, 1] by bisection steps.
"""
function mean_input(θ, S, d, m0)
    lo, hi = θ + min(0.0, S) - 1e-9, θ + max(0.0, S) + 1e-9
    m = clamp(m0, lo, hi)
    for _ in 1:100
        f = m - S * (σ(m + d) + σ(m - d)) / 2 - θ
        abs(f) < 1e-13 && return m
        f > 0 ? (hi = m) : (lo = m)
        step = m - f / (1 - S * (dσ(m + d) + dσ(m - d)) / 2)
        m = lo < step < hi ? step : (lo + hi) / 2
        hi - lo < 1e-14 && return m
    end
    m
end

"""
    latched_pairs(θ, S, D; n = 1500)

The number of stable asymmetric rest states with d > 0 (each is one half of a mirror-image pair) of the circuit with
bias `θ`, even weight `S` and odd weight `D`, for S ≤ 4: the roots of G(d) = D δ(m(d), d) − d on 0 < d ≤ D/2,
found on `n` steps and refined by bisection, that pass `is_stable_rest`.
"""
function latched_pairs(θ, S, D; n=1500)
    D <= 0 && return 0
    Ws, Wc = (S + D) / 2, (S - D) / 2
    G(d, m) = D * (σ(m + d) - σ(m - d)) / 2 - d
    count = 0
    dp = D / 2 * 1e-7
    mp = mean_input(θ, S, dp, θ)
    gp = G(dp, mp)
    for k in 1:n
        d = D / 2 * k / n
        m = mean_input(θ, S, d, mp)
        g = G(d, m)
        if gp != 0 && gp * g <= 0
            lo, hi, glo, mlo = dp, d, gp, mp
            dm, mm = d, m
            for _ in 1:60
                dm = (lo + hi) / 2
                mm = mean_input(θ, S, dm, mlo)
                gm = G(dm, mm)
                glo * gm <= 0 ? (hi = dm) : (lo = dm; glo = gm; mlo = mm)
            end
            is_stable_rest(Ws, Wc, mm + dm, mm - dm) && (count += 1)
        end
        dp, mp, gp = d, m, g
    end
    count
end

"""
    newton_rests(θ, Ws, Wc; stop_at_latch = false)

The stable rest states (x_L, x_R) of the circuit with bias `θ` and weights `Ws`, `Wc`, by Newton's method (with step
halving) from the net inputs that each output pair of a 21 × 21 grid drives. With `stop_at_latch`, it stops at the
first asymmetric one.
"""
function newton_rests(θ, Ws, Wc; stop_at_latch=false)
    found = NTuple{2,Float64}[]
    residual(x1, x2) = (θ + Ws * σ(x1) + Wc * σ(x2) - x1, θ + Wc * σ(x1) + Ws * σ(x2) - x2)
    for a in 0:0.05:1, b in 0:0.05:1
        x1, x2 = θ + Ws * a + Wc * b, θ + Wc * a + Ws * b
        r1, r2 = residual(x1, x2)
        for _ in 1:100
            r0 = max(abs(r1), abs(r2))
            r0 < 1e-12 && break
            s1, s2 = dσ(x1), dσ(x2)
            A, B, C, E = Ws * s1 - 1, Wc * s2, Wc * s1, Ws * s2 - 1
            det = A * E - B * C
            δ1, δ2 = (E * r1 - B * r2) / det, (A * r2 - C * r1) / det
            t = 1.0
            n1, n2 = residual(x1 - δ1, x2 - δ2)
            while t > 1e-8 && !(max(abs(n1), abs(n2)) < r0)
                t /= 2
                n1, n2 = residual(x1 - t * δ1, x2 - t * δ2)
            end
            x1, x2, r1, r2 = x1 - t * δ1, x2 - t * δ2, n1, n2
        end
        (max(abs(r1), abs(r2)) < 1e-10 && is_stable_rest(Ws, Wc, x1, x2)) || continue
        any(f -> abs(f[1] - x1) + abs(f[2] - x2) < 1e-6, found) && continue
        push!(found, (x1, x2))
        stop_at_latch && abs(x1 - x2) > 1e-6 && break
    end
    found
end

"""Whether the circuit with bias `θ`, even weight `S` and odd weight `D` latches: `latched_pairs` for S ≤ 4, else `newton_rests`."""
function is_latched(θ, S, D)
    S <= 4 && return latched_pairs(θ, S, D) > 0
    any(x -> abs(x[1] - x[2]) > 1e-6, newton_rests(θ, (S + D) / 2, (S - D) / 2; stop_at_latch=true))
end

"""
    latch_threshold(u, S; Dmax = 200, steps = 24)

The latch threshold D_crit(u, S): the smallest odd weight at which the circuit whose symmetric rest has net input
`u` and whose even weight is `S` has a stable latched pair, by `steps` bisections on [0, `Dmax`]. `Inf` if the
circuit does not latch even at `Dmax`.
"""
function latch_threshold(u, S; Dmax=200.0, steps=24)
    θ = bias_of(u, S)
    is_latched(θ, S, Dmax) || return Inf
    lo, hi = 0.0, Dmax
    for _ in 1:steps
        mid = (lo + hi) / 2
        is_latched(θ, S, mid) ? (hi = mid) : (lo = mid)
    end
    hi
end

"""
The symmetric rest of each circuit of `circuits` with nothing in view (§1, the notebook's REST): its `name`, bias `θ`,
self and cross weights `Ws` and `Wc`, the even and odd weights `S` and `D`, the net input `u` = u* (the one solution of
u = θ + S σ(u)), the output `o` = σ(u*), and the odd and even gains `odd` = D σ′(u*) and `even` = S σ′(u*).
"""
rest_states(circuits=CIRCUITS) = map(circuits) do a
    θ, Ws, Wc = a.q[2], a.q[3], a.q[4]
    S, D = Ws + Wc, Ws - Wc
    f(u) = θ + S * σ(u) - u
    us = range(θ + min(0, S) - 1, θ + max(0, S) + 1; length=4001)   # every solution lies in θ + S·[0, 1]
    roots = Float64[]
    for k in 1:length(us)-1
        lo, hi = us[k], us[k+1]
        f(lo) * f(hi) <= 0 || continue
        for _ in 1:60
            mid = (lo + hi) / 2
            f(lo) * f(mid) <= 0 ? (hi = mid) : (lo = mid)
        end
        push!(roots, (lo + hi) / 2)
    end
    @assert length(roots) == 1 "$(a.name) has $(length(roots)) symmetric rest states"
    u = roots[1]
    slope = σ(u) * (1 - σ(u))
    (; name=a.short, θ, Ws, Wc, S, D, u, o=σ(u), odd=D * slope, even=S * slope)
end

"""
The action mode of each circuit (§2, the notebook's ACTION_MODES): its symmetric rest (`rest_states`), whether it is
`latched`, and through what `mechanism`: `:pitchfork` when its symmetric rest is unstable (odd gain above 1), `:fold`
when it latches although its symmetric rest is stable, else `:symmetric`. Also its latch threshold `dcrit` =
D_crit(u*, S) and its `margin` D − D_crit.
"""
action_modes(rest=rest_states()) = map(rest) do r
    latched = is_latched(r.θ, r.S, r.D)
    mechanism = !latched ? :symmetric : r.odd > 1 ? :pitchfork : :fold
    dcrit = latch_threshold(r.u, r.S)
    (; r..., latched, mechanism, dcrit, margin=r.D - dcrit)
end

"""
    relief_map(R, k; cond = 2)

The TE map of circuit `k` (§7, eccentricity × moment) in the relief cache `R`, given the flat drive (`cond` = 2) or
plain (1): blank (NaN) where fewer than 100 trials read relief on that ray or where the force varies by less than 1%
of its largest spread, and the estimator's small negative values (≈ −0.02 bits at zero, §7) set to 0.
"""
function relief_map(R, k; cond=2)
    Z = max.(R.te[k, cond, :, :], 0.0)
    live_force = R.sforce[k, :] .>= 0.01 * maximum(R.sforce[k, :])
    for e in axes(Z, 1), m in axes(Z, 2)
        (R.nrelief[k, e, m] >= R.minhit && live_force[m] && isfinite(R.te[k, cond, e, m])) || (Z[e, m] = NaN)
    end
    Z
end

"""A centred 3-point mean along the moments (±0.4 time units), over the finite values (the §7 notebook's)."""
function smooth3_finite(v)
    map(eachindex(v)) do i
        w = filter(isfinite, v[max(1, i - 1):min(end, i + 1)])
        isfinite(v[i]) && !isempty(w) ? sum(w) / length(w) : NaN
    end
end

"""
Where each circuit's relief TE peaks (§7, the notebook's RELIEF_PEAKS), from its map in the relief cache `R` smoothed
over 3 moments: the height `h`, the moment `t` and the eccentricity `ecc` of the largest TE given the flat drive, its
value `te`, the span of heights `span` where the TE (largest over the rays) stays above half its peak, and the same
peak `h_plain`, `te_plain` for the plain TE.
"""
relief_peaks(R, circuits=CIRCUITS) = map(eachindex(circuits)) do k
    peak(cond) = begin
        Z = reduce(vcat, [smooth3_finite(relief_map(R, k; cond)[e, :])' for e in 1:4])
        Zs = replace(Z, NaN => -Inf)
        i = argmax(Zs)
        top = vec(maximum(Zs; dims=1))
        above = findall(>=(Zs[i] / 2), top)
        (; h=R.h[i[2]], t=R.t[i[2]], ecc=R.eccentricity[i[1]], te=Zs[i], span=(R.h[maximum(above)], R.h[minimum(above)]))
    end
    p, q = peak(2), peak(1)
    (; name=circuits[k].short, p..., h_plain=q.h, te_plain=q.te)
end

"""The circuits in the order of the §7 figures (the notebook's RELIEF_ORDER): latched first, then symmetric; within each, by the height of the TE peak, highest first."""
relief_order(modes, peaks, circuits=CIRCUITS) =
    sortperm(collect(eachindex(circuits)); by=k -> (!modes[k].latched, -peaks[k].h))

"""A circuit's family from the signs of its mirror-odd ray drives (cat_meta.jl): edge tracker (P, or +−−), middle first (−++), else middle bringer."""
family(c) = let signs = join(((c.q[4+j] - c.q[12-j]) / 2 > 0 ? "+" : "−") for j in 1:3)
    c.short == "P" || signs == "+−−" ? :edge : signs == "−++" ? :first : :middle
end
