# The larger agents' runner and the two-node procedures (te_work/figures/nnode_common.jl, shared by the figures' larger-
# agent data and bootstrap stages), verbatim: `NNodeModel` as NM, §7's views (views.jl), `run_view_n`, the relief windows
# (the relief ablation's), what the position holds (the category TE's `mi_x`), the split, the odd loop gain and the sign row.

const NM = NNodeModel

const KS = 81:4:857                         # the moments of the category ensemble (category data stage)
const ENSEMBLE_STARTS = collect(0.1:0.1:50.0)
const NOISE, REPS, NSHUF = 0.05, 2, 10       # the category TE stage's
const SPLIT_BITS = 0.9
const WINDOW_CENTRES = 20.0:5.0:255.0       # the relief ablation's
const WINDOW_WIDTH = 20.0
const CATCH_STARTS = [-reverse(1.0:50.0); 1.0:50.0]
const NSUB = round(Int, VISUAL.dt / NM.DT)  # Euler steps between recorded positions (10)
const POSITIONS = 868
const HEIGHTS = VISUAL.start_height .- VISUAL.dt * VISUAL.fall_speed .* (KS .- 1)

"""
    run_view_n(c, d, view; outside = CIRCLE_VIEW, window = (-Inf, Inf), record = false)

`NM.final_dx` for the N-node circuit `c` from Δx = `d`, with the object read in `view` while its height is inside
`window` and in `outside` elsewhere (`view_lengths!`). Returns the final Δx, or with `record` Δx at the 868 positions.
"""
function run_view_n(c::NM.NCircuit{M,C}, d, view; outside=CIRCLE_VIEW, window=(-Inf, Inf), record=false) where {M,C}
    p, dt = VISUAL, NM.DT
    decay = exp(-dt / NM.MASS)
    yL = yR = ntuple(_ -> 0.0, Val(M)); yC = 0.0
    v, ax, oy = 0.0, Float64(d), p.start_height
    oL, oR, oC = NM.outputs(c, yL, yR, yC)
    L, Lf = zeros(7), zeros(7)
    xs = record ? zeros(POSITIONS) : nothing
    record && (xs[1] = ax)
    for s in 1:NM.nsteps(dt)
        view_lengths!(L, Lf, ax, oy, window[1] <= oy <= window[2] ? view : outside)
        I = ntuple(j -> p.input_gain * (p.max_ray - L[j]) / p.max_ray, 7)
        yL, yR, yC = NM.node_step(c, yL, yR, yC, oL, oR, oC, I, dt)
        oy -= dt * p.fall_speed
        oL, oR, oC = NM.outputs(c, yL, yR, yC)
        F = p.velocity_gain * (oR[1] - oL[1])
        v = F + (v - F) * decay
        ax += dt * v
        if abs(ax) > p.width / 2
            ax, v = clamp(ax, -p.width / 2, p.width / 2), 0.0
        end
        record && s % NSUB == 0 && (xs[s÷NSUB+1] = ax)
    end
    record ? xs : ax
end

"""Whether each circle trial (columns: `CATCH_STARTS`) is caught with the circle read flat inside each relief window (rows: `WINDOW_CENTRES`)."""
function relief_outcomes(c)
    caught = falses(length(WINDOW_CENTRES), length(CATCH_STARTS))
    Threads.@threads for w in eachindex(WINDOW_CENTRES)
        h = WINDOW_CENTRES[w]
        for (i, d) in enumerate(CATCH_STARTS)
            caught[w, i] = abs(run_view_n(c, d, FLAT_VIEW; window=(h - WINDOW_WIDTH / 2, h + WINDOW_WIDTH / 2))) < 30
        end
    end
    caught
end

"""The catch rate with the circle read flat inside each relief window (the relief ablation's `windows[k, 1, :]`)."""
relief_windows(c) = vec(count(relief_outcomes(c); dims=2)) ./ length(CATCH_STARTS)

"""
What the position holds about the category at each moment, I(C; x) in bits, as the category TE stage computes `mi_x`:
the same ensemble, noise, draws (the generator seeded by (name, moment), with the draws the category TE makes for its
other variables and its label shuffles skipped over in the same order) and estimator.
"""
function position_info(c, name)
    trials = [(s, d) for s in (:circle, :line) for d in ENSEMBLE_STARTS]
    N = length(trials)
    X = zeros(length(KS), N)
    Threads.@threads for n in 1:N
        s, d = trials[n]
        X[:, n] .= run_view_n(c, d, s == :circle ? CIRCLE_VIEW : LINE_VIEW; record=true)[KS]
    end
    C = first.(trials) .== :circle
    whole = max(std(X), 1e-12)
    mi = tmap(eachindex(KS)) do m
        rng = Xoshiro(hash((name, m)))
        acc = 0.0
        for r in 1:REPS
            x = reshape(X[m, :] ./ whole .+ NOISE .* randn(rng, N), 1, N)
            for _ in 1:8                                   # the category TE's draws for o_L, o_R, F, o_L⁺, o_R⁺, F⁺, F⁺02, F⁺10
                randn(rng, N)
            end
            acc += ross_mi_nd(x, C) / REPS
            r == 1 && foreach(_ -> shuffle(rng, C), 1:NSHUF)   # its label shuffles for the null
        end
        acc
    end
    mi, X
end
"""The split: the height where the position first holds `SPLIT_BITS` (0.9) bits about the shape; NaN if never."""
split_height(mi) = (s = findfirst(>=(SPLIT_BITS), mi); s === nothing ? NaN : HEIGHTS[s])

"""The odd loop gain at the symmetric rest: the even-stable mirror-plane rest's (the smaller where there are two), else the only plane rest's."""
function odd_gain(r)
    stable = [p for p in r.plane if real(p.lead_even) < 0]
    ps = isempty(stable) ? r.plane : stable
    minimum(p -> p.odd_gain, ps)
end

"""The signs of the odd parts a_j = (U_j − U_{8−j})/2 of a node's ray weights `U`, outer to inner pair."""
sign_row(U) = join(((U[j] - U[8-j]) / 2 > 0 ? "+" : "−") for j in 1:3)
