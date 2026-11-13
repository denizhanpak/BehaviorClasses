# Analyze the categorization agents fitted by pipeline/fit_categorization.jl, in three blocks, each drawing its figure:
#
#   1. Relief and category: when the circle's relief and the object's category get into the agent's behaviour
#      → figure Categorization2
#   2. The angle distribution: where each agent holds the circle's edge, and its hold rule by sign row
#      → figure Categorization3
#   3. Latched or symmetric: the latch threshold, and the edge law's damping against the odd loop gain
#      → figure Categorization4
#
#   julia -t 7 --project pipeline/analyze_categorization.jl            # every step still to do, and the figures
#   julia -t 7 --project pipeline/analyze_categorization.jl --list     # the steps and which are done
#   julia -t 7 --project pipeline/analyze_categorization.jl --force    # redo every step
#   julia -t 7 --project pipeline/analyze_categorization.jl --only category_te,category_cache
#
# Steps of block 1 (a step is skipped when its outputs exist; runtimes on 7 threads):
#   relief_data     per circuit, 1000 circles and 1000 lines: ray relief, flat drives, motor outputs     ≈ 7 min
#   relief_te       TE from each ray's relief to the motor force, moment by moment                       ≈ 15 min
#   relief_ablate   catch rates with the relief (or every cue) removed, whole trial or in height windows ≈ 20 min
#   relief_cache    → cache/categorization_strategies_relief                                             ≈ 1 min
#   category_data   per circuit, 500 circles and 500 lines (dt 0.01; ten circuits again at dt 0.001)     ≈ 4 min
#   category_te     what the motors and the position hold about the category, and the TE it drives      ≈ 30 min
#   category_meta   each circuit's action mode, family and relief windows                                seconds
#   category_cache  → cache/categorization_strategies_category                                           seconds
#   dstar           the deficit law's switch height against where the position learns the category      ≈ 1 min
#                   → cache/categorization_strategies_dstar
#   larger_agents   the two-node procedures run on the 60 agents of 3 … 8 nodes: split, relief windows,
#                   sign row, odd loop gain (for figures 2–4) → cache/categorization_figures_nnode       ≈ 15 min
#   bootstrap       bootstrap 95% intervals of every agent's split, relief window and damping b
#                   (for figures 2 and 4) → cache/categorization_figures_boot                            ≈ 55 min
# Blocks 2 and 3 compute from the fits and these caches as they draw (src/figures/Categorization3.jl, 4.jl).
#
# Inputs: data/inputs/ and fit_categorization.jl's caches and intermediate/nnode/rest_modes.jld2. Writes intermediate/relief/,
# intermediate/category/, intermediate/figures/ (≈ 1.6 GB of trial ensembles in all) and cache/ under PIPELINE_DATA
# (default data/). For a smoke test set PIPELINE_SUBSET (e.g. "A,K") and PIPELINE_DATA to a scratch directory.
using BehaviorClasses, LinearAlgebra, Statistics, Random, Printf, StatsBase, HypothesisTests
using Associations
import Associations: convert_to_cmi_estimator
import BehaviorClasses: WINDOW_CENTRES, WINDOW_WIDTH, HEIGHTS, SPLIT_BITS, CATCH_STARTS
check_output_target()
const OPTS = pipeline_options(ARGS)

const KS = 81:4:857            # the sampled positions: t = 8.0:0.4:85.6 (the circle comes into range at h ≈ 250, t ≈ 8)
const LAGS = (2, 5, 10)        # the force or outputs at k + 2, 5, 10 (λ = 0.2, 0.5, 1.0)
number(file) = parse(Int, match(r"circuit_(\d+)\.jld2$", file)[1])
jld2_files(dir) = sort(filter(endswith(".jld2"), readdir(dir; join=true)))

# ══ Block 1: relief and category ══════════════════════════════════════════════════════════════════════════════════════

"""
The trial ensembles for the relief TE. Per circuit, 1000 circles and 1000 lines from d = ±(0.1:0.1:50) (the ordinary
task, finer), sampled at the positions KS: ray j's relief rⱼ = I^flat_j − Iⱼ (7 rays), the flat drives
Φ_L = Σⱼ U_Lj I^flat_j and Φ_R, the motor outputs o_L, o_R, the force F at k + LAGS, and the number of rays that hit.
One file per circuit (Float32).
"""
function relief_data(outdir)
    starts = [-reverse(0.1:0.1:50.0); 0.1:0.1:50.0]
    function circuit_data(entry)
        c = circuit(entry.q)
        trials = [(s, d) for s in (:circle, :line) for d in starts]
        M, N = length(KS), length(trials)
        R = zeros(Float32, 7, M, N); Φ = zeros(Float32, 2, M, N); O = zeros(Float32, 2, M, N)
        Ff = zeros(Float32, length(LAGS), M, N); nhit = zeros(UInt8, M, N); xend = zeros(N)
        Threads.@threads for n in 1:N
            s, d = trials[n]
            tr = run_view(c, d, s == :circle ? CIRCLE_VIEW : LINE_VIEW; record=true)
            xend[n] = tr.x[end]
            o(k, i) = node_out(c, view(tr.Y, :, k), i)
            F(k) = VISUAL.velocity_gain * (o(k, c.R) - o(k, c.L))
            for (m, k) in enumerate(KS)
                for j in 1:7
                    R[j, m, n] = tr.If[j, k] - tr.I[j, k]
                end
                Φ[1, m, n] = dot(view(c.U, c.L, :), view(tr.If, :, k))
                Φ[2, m, n] = dot(view(c.U, c.R, :), view(tr.If, :, k))
                O[1, m, n], O[2, m, n] = o(k, c.L), o(k, c.R)
                for (l, lag) in enumerate(LAGS)
                    Ff[l, m, n] = F(k + lag)
                end
                nhit[m, n] = count(>(0), view(tr.I, :, k))
            end
        end
        (; name=entry.name, short=entry.short, t=VISUAL.dt .* (KS .- 1), h=VISUAL.start_height .- VISUAL.dt * VISUAL.fall_speed .* (KS .- 1),
            lags=VISUAL.dt .* LAGS, shape=first.(trials), d=last.(trials), xend, R, Φ, O, Ff, nhit)
    end
    for (k, entry) in enumerate(CIRCUITS)
        in_subset(entry.short) || continue
        D = circuit_data(entry)
        save_data(joinpath(outdir, "circuit_$(lpad(k, 2, '0')).jld2"), D)
        caught = count(i -> D.shape[i] == :circle && abs(D.xend[i]) < 30, eachindex(D.xend))
        avoided = count(i -> D.shape[i] == :line && abs(D.xend[i]) >= 30, eachindex(D.xend))
        println(rpad(entry.short, 5), " circles caught ", caught, "/1000, lines avoided ", avoided, "/1000"); flush(stdout)
    end
end

"""
Ensemble TE from each ray's relief to the motor force, at each moment, pooled over the 2000 trials of a circuit:
  plain: TE = I(F(t+λ); rⱼ(t) | o_L(t), o_R(t))
  flat:  TE = I(F(t+λ); rⱼ(t) | o_L(t), o_R(t), Φ_L(t), Φ_R(t))
with Associations.jl's MIDecomposition(TEShannon, KSG2) (k = 5) in its CMI form (unbiased at 0, about −20% at
0.4–0.8 bits at this dimension and N). Each variable is divided by its spread across the trials at that moment, and
noise 5% of that spread is added, averaged over 2 draws. A moment needs relief on at least 100 trials. λ = 0.5 for all
circuits, λ = 0.2, 0.5, 1.0 for A, C, D, F; rays 1–4 (all 7 for A, the mirror check).
"""
function relief_te(datadir, out_file)
    noise, reps, minhit = 0.05, 2, 100
    te = convert_to_cmi_estimator(MIDecomposition(TEShannon(; base=2), KSG2(MIShannon(; base=2); k=5)))
    sweep = ("A", "C", "D", "F")
    conds = (:plain, :flat)
    spread(x) = max(std(x), 1e-12)
    function te_job(D, m, j, cond, l, rng)
        r = Float64.(D.R[j, m, :])
        count(>(1e-9), r) < minhit && return NaN
        f = Float64.(D.Ff[l, m, :])
        std(f) < 1e-9 && return NaN
        cols = [Float64.(D.O[1, m, :]), Float64.(D.O[2, m, :])]
        cond == :flat && append!(cols, [Float64.(D.Φ[1, m, :]), Float64.(D.Φ[2, m, :])])
        sc(x) = x ./ spread(x)
        r, f, cols = sc(r), sc(f), sc.(cols)
        acc = 0.0
        for _ in 1:reps
            nz(x) = x .+ noise .* randn(rng, length(x))
            acc += association(te, StateSpaceSet(nz(f)), StateSpaceSet(nz(r)), StateSpaceSet(hcat(nz.(cols)...))) / reps
        end
        acc
    end
    # circuit_kk.jld2: kk is the circuit's number in CIRCUITS
    files = jld2_files(datadir)
    results = []
    t0 = time()
    for file in files
        k = number(file)
        in_subset(CIRCUITS[k].short) || continue
        D = load_data(file)
        M = length(D.t)
        lags = D.short in sweep ? (1, 2, 3) : (2,)
        rays = D.short == "A" ? (1:7) : (1:4)
        jobs = [(m, j, ci, l) for m in 1:M for j in rays for ci in 1:2 for l in lags]
        out = fill(NaN, 7, 2, 3, M)
        Threads.@threads :dynamic for q in jobs
            local m, j, ci, l = q
            out[j, ci, l, m] = te_job(D, m, j, conds[ci], l, MersenneTwister(hash((k, q))))
        end
        nr = [count(>(1e-9), view(D.R, j, m, :)) for j in 1:7, m in 1:M]
        sr = [std(view(D.R, j, m, :)) for j in 1:7, m in 1:M]
        sf = [std(view(D.Ff, l, m, :)) for l in 1:3, m in 1:M]
        push!(results, (; name=D.name, short=D.short, te=out, nrelief=nr, srelief=sr, sforce=sf))
        println(rpad(D.short, 5), " done at ", round(time() - t0), " s; peak flat TE (λ 0.5) ",
            round(maximum(x -> isnan(x) ? -Inf : x, out[1:4, 2, 2, :]); digits=2)); flush(stdout)
    end
    D = load_data(files[1])
    save_data(out_file, (; results, t=D.t, h=D.h, lags=D.lags, conds, noise, reps, minhit))
end

"""
The ablations over the 100 ordinary circle trials (d = ±1…±50) of every circuit: (a) the catch rate (final |Δx| < 30)
in each of the 8 VIEWS, for the whole trial; (b) the catch rate with the circle read flat (FLAT_VIEW, relief off) or as
the line (LINE_VIEW, all cues off) only while its height is inside a window of width 20 centred at 20:5:255.
"""
function relief_ablate(out_file)
    starts = [-reverse(SIDE); SIDE]
    caught_rate(c, view; kw...) = count(d -> abs(run_view(c, d, view; kw...)[end]) < 30, starts) / length(starts)
    factorial = zeros(length(CIRCUITS), length(VIEWS))
    windows = zeros(length(CIRCUITS), 2, length(WINDOW_CENTRES))
    todo = [k for k in eachindex(CIRCUITS) if in_subset(CIRCUITS[k].short)]
    Threads.@threads for k in todo
        c = circuit(CIRCUITS[k].q)
        for (v, view) in enumerate(VIEWS)
            factorial[k, v] = caught_rate(c, view)
        end
        for (w, h) in enumerate(WINDOW_CENTRES), (m, view) in enumerate((FLAT_VIEW, LINE_VIEW))
            windows[k, m, w] = caught_rate(c, view; window=(h - WINDOW_WIDTH / 2, h + WINDOW_WIDTH / 2))
        end
    end
    save_data(out_file, (; factorial, views=VIEWS, labels=view_label.(VIEWS), windows,
        window_views=(FLAT_VIEW, LINE_VIEW), centres=collect(WINDOW_CENTRES), width=WINDOW_WIDTH, names=[e.short for e in CIRCUITS]))
    println("views: ", view_label.(VIEWS))
    for k in todo
        println(rpad(CIRCUITS[k].short, 5), " ", join(lpad.(round.(Int, 100 .* factorial[k, :]), 4)),
            "   flat-window min ", round(Int, 100 * minimum(windows[k, 1, :])), " at h ", WINDOW_CENTRES[argmin(windows[k, 1, :])],
            "   line-window min ", round(Int, 100 * minimum(windows[k, 2, :])), " at h ", WINDOW_CENTRES[argmin(windows[k, 2, :])])
    end
end

"""
Merge the relief TE, the ablations and the ensembles' relief statistics into the notebook's relief cache. Rays are
ordered by eccentricity: 0°, 5°, 10°, 15° = rays 4, 3, 2, 1 (ray 8 − j is the mirror image of ray j).
"""
function relief_cache(te_file, ablate_file, datadir, out_file)
    T, A = load_data(te_file), load_data(ablate_file)
    files = jld2_files(datadir)
    rays = [4, 3, 2, 1]
    K, M = length(T.results), length(T.t)
    @assert [r.short for r in T.results] == A.names
    te = zeros(K, 2, 4, M)                     # [circuit, plain/flat, eccentricity, moment] at λ = 0.5
    sweep = Dict{String,Array{Float64,4}}()    # [plain/flat, eccentricity, λ, moment] for A, C, D, F
    mirror = nothing
    for (k, r) in enumerate(T.results)
        for (e, j) in enumerate(rays), c in 1:2
            te[k, c, e, :] .= r.te[j, c, 2, :]
        end
        if !all(isnan, r.te[rays, :, 1, :])
            sweep[r.short] = permutedims(r.te[rays, :, :, :], (2, 1, 3, 4))
        end
        r.short == "A" && (mirror = (; rays=r.te[1:3, :, 2, :], mirrors=r.te[7:-1:5, :, 2, :]))
    end
    nrelief = [T.results[k].nrelief[j, m] for k in 1:K, j in rays, m in 1:M]
    srelief = [T.results[k].srelief[j, m] for k in 1:K, j in rays, m in 1:M]
    sforce = [T.results[k].sforce[2, m] for k in 1:K, m in 1:M]
    # the relief that the circle trials see: the mean over circle trials and over the mirror pair, and the share of
    # circle trials in which the ray reads any relief
    mean_relief = zeros(K, 4, M); share_relief = zeros(K, 4, M)
    for (k, f) in enumerate(files)
        D = load_data(f)
        circ = findall(==(:circle), D.shape)
        for (e, j) in enumerate(rays), m in 1:M
            pair = j == 4 ? [4] : [j, 8 - j]
            v = [Float64(D.R[i, m, n]) for i in pair for n in circ]
            mean_relief[k, e, m] = mean(v)
            share_relief[k, e, m] = count(>(1e-9), v) / length(v)
        end
    end
    save_data(out_file, (; names=A.names, t=T.t, h=T.h, lag=T.lags[2], lags=T.lags, eccentricity=[0, 5, 10, 15], te,
        conds=T.conds, sweep, mirror, nrelief, srelief, sforce, mean_relief, share_relief, minhit=T.minhit, noise=T.noise,
        reps=T.reps, factorial=A.factorial, views=A.views, labels=A.labels, windows=A.windows, window_views=A.window_views,
        centres=A.centres, width=A.width))
end

"""
The trial ensembles for the category TE. Per circuit, 500 circles and 500 lines from d = 0.1:0.1:50 (one side; the
other side is the exact mirror image, `mirror_sums!`), at the positions KS: the motor outputs o_L, o_R at k and at
k + LAGS, the agent's offset x at k, and the number of rays that hit. Float64. All 41 circuits at the notebooks' Euler
step 0.01, and ten circuits (A D F G #16 P J K M Q) at 0.001, the step-size check.
"""
function category_data(dt, outdir)
    side_starts = collect(0.1:0.1:50.0)
    dt_check = ["A", "D", "F", "G", "#16", "P", "J", "K", "M", "Q"]
    function circuit_data(entry)
        c = circuit(entry.q)
        trials = [(s, d) for s in (:circle, :line) for d in side_starts]
        M, N = length(KS), length(trials)
        O = zeros(2, M, N)                      # o_L, o_R at k
        Of = zeros(2, length(LAGS), M, N)       # o_L, o_R at k + lag
        X = zeros(M, N); nhit = zeros(UInt8, M, N); xend = zeros(N)
        Threads.@threads :dynamic for n in 1:N
            s, d = trials[n]
            tr = run_view(c, d, s == :circle ? CIRCLE_VIEW : LINE_VIEW; record=true, dt)
            xend[n] = tr.x[end]
            o(k, i) = node_out(c, view(tr.Y, :, k), i)
            for (m, k) in enumerate(KS)
                O[1, m, n], O[2, m, n] = o(k, c.L), o(k, c.R)
                for (l, lag) in enumerate(LAGS)
                    Of[1, l, m, n], Of[2, l, m, n] = o(k + lag, c.L), o(k + lag, c.R)
                end
                X[m, n] = tr.x[k]
                nhit[m, n] = count(>(0), view(tr.I, :, k))
            end
        end
        (; name=entry.name, short=entry.short, dt, t=VISUAL.dt .* (KS .- 1),
            h=VISUAL.start_height .- VISUAL.dt * VISUAL.fall_speed .* (KS .- 1), lags=VISUAL.dt .* LAGS,
            gain=VISUAL.velocity_gain, shape=first.(trials), d=last.(trials), xend, O, Of, X, nhit)
    end
    only = dt == 0.01 ? nothing : dt_check
    println("dt ", dt, ", substeps ", round(Int, VISUAL.dt / dt), ", threads ", Threads.nthreads()); flush(stdout)
    for (k, entry) in enumerate(CIRCUITS)
        (only === nothing || entry.short in only) && in_subset(entry.short) || continue
        D = circuit_data(entry)
        save_data(joinpath(outdir, "circuit_$(lpad(k, 2, '0')).jld2"), D)
        caught = count(i -> D.shape[i] == :circle && abs(D.xend[i]) < 30, eachindex(D.xend))
        avoided = count(i -> D.shape[i] == :line && abs(D.xend[i]) >= 30, eachindex(D.xend))
        println(rpad(entry.short, 5), " circles caught ", caught, "/500, lines avoided ", avoided, "/500"); flush(stdout)
    end
end

"""
The category TE. At each moment the trials of a circuit (500 circles, 500 lines, the object starting on the agent's
left) are one ensemble. For each target Y (o_L, o_R and the force F = 5 (o_R − o_L)):
  holds   I(C; Y(t))           (and the circuit's state, I(C; o_L, o_R), and the body's position, I(C; x))
  driven  TE_{C→Y}(t) = I(Y(t+λ); C | Y(t)) = I(C; Y(t+λ), Y(t)) − I(C; Y(t))
by Ross's nearest-neighbour estimator (k = 5) on rows divided by their spread over the run, with noise 5% of it,
averaged over 2 draws; a null from 10 label shuffles. λ = 0.5 (and 0.2, 1.0 for F).
"""
function category_te(datadir, out_file)
    noise, reps, nshuf = 0.05, 2, 10
    quantities = (:mi_x, :mi_oL, :mi_oR, :mi_F, :mi_state, :te_oL, :te_oR, :te_F, :te_F_02, :te_F_10)
    nulls = (:mi_x, :mi_F, :mi_state, :te_oL, :te_oR, :te_F)
    # the quantities at one moment from noisy rows (each a 1 × N matrix) and labels C
    function moment(v, C)
        mi(rows...) = ross_mi_nd(vcat(rows...), C)
        m = Dict(:mi_x => mi(v.x), :mi_oL => mi(v.oL), :mi_oR => mi(v.oR), :mi_F => mi(v.F), :mi_state => mi(v.oL, v.oR))
        m[:te_oL] = mi(v.oL⁺, v.oL) - m[:mi_oL]
        m[:te_oR] = mi(v.oR⁺, v.oR) - m[:mi_oR]
        m[:te_F] = mi(v.F⁺, v.F) - m[:mi_F]
        m[:te_F_02] = mi(v.F⁺02, v.F) - m[:mi_F]
        m[:te_F_10] = mi(v.F⁺10, v.F) - m[:mi_F]
        m
    end
    # the null quantities: the same rows with the labels shuffled
    function moment_null(v, C)
        mi(rows...) = ross_mi_nd(vcat(rows...), C)
        a, b, f = mi(v.oL), mi(v.oR), mi(v.F)
        Dict(:mi_x => mi(v.x), :mi_F => f, :mi_state => mi(v.oL, v.oR), :te_oL => mi(v.oL⁺, v.oL) - a,
            :te_oR => mi(v.oR⁺, v.oR) - b, :te_F => mi(v.F⁺, v.F) - f)
    end
    function circuit_te(D)
        C = D.shape .== :circle
        M, N = length(D.t), length(C)
        raw = (x=D.X, oL=D.O[1, :, :], oR=D.O[2, :, :], F=D.gain .* (D.O[2, :, :] .- D.O[1, :, :]),
            oL⁺=D.Of[1, 2, :, :], oR⁺=D.Of[2, 2, :, :], F⁺=D.gain .* (D.Of[2, 2, :, :] .- D.Of[1, 2, :, :]),
            F⁺02=D.gain .* (D.Of[2, 1, :, :] .- D.Of[1, 1, :, :]), F⁺10=D.gain .* (D.Of[2, 3, :, :] .- D.Of[1, 3, :, :]))
        whole = map(A -> max(std(A), 1e-12), raw)                       # spread over the whole run (M × N)
        out = Dict(q => zeros(M) for q in quantities)
        null = Dict(q => zeros(nshuf, M) for q in nulls)
        per_moment = tmap(1:M) do m
            rng = Xoshiro(hash((D.short, m)))
            sums = Dict(q => 0.0 for q in quantities)
            nl = Dict(q => zeros(nshuf) for q in nulls)
            for r in 1:reps
                v = map(keys(raw)) do key
                    reshape(raw[key][m, :] ./ whole[key] .+ noise .* randn(rng, N), 1, N)
                end
                v = NamedTuple{keys(raw)}(v)
                res = moment(v, C)
                foreach(q -> sums[q] += res[q] / reps, quantities)
                if r == 1
                    for sh in 1:nshuf
                        rn = moment_null(v, shuffle(rng, C))
                        foreach(q -> nl[q][sh] = rn[q], nulls)
                    end
                end
            end
            (sums, nl)
        end
        for (m, (sums, nl)) in enumerate(per_moment)
            foreach(q -> out[q][m] = sums[q], quantities)
            foreach(q -> null[q][:, m] .= nl[q], nulls)
        end
        (; name=D.name, short=D.short, dt=D.dt, out, null)
    end
    results = []
    t0 = time()
    D = nothing
    for f in jld2_files(datadir)
        D = load_data(f)
        in_subset(D.short) || continue
        r = circuit_te(D)
        push!(results, r)
        pk = argmax(r.out[:te_F])
        floorF = quantile(vec(r.null[:te_F]), 0.99)
        println(rpad(r.short, 5), " TE→F peak ", round(r.out[:te_F][pk]; digits=3), " at h ", round(D.h[pk]; digits=1),
            " (null 99%: ", round(floorF; digits=3), "); state holds ", round(maximum(r.out[:mi_state]); digits=3),
            "  [", round(time() - t0), " s]")
        flush(stdout)
    end
    save_data(out_file, (; t=D.t, h=D.h, lags=D.lags, noise, reps, nshuf, results))
end

"""
Each circuit's action mode (latched or symmetric, and how) and family, by the rest-state solvers; its relief windows
(the relief cache's windows below 80%), the order of the relief figures and the colours.
"""
function category_meta(relief_file, out_file)
    R = load_data(relief_file)
    @assert R.names == [c.short for c in CIRCUITS]
    modes = action_modes(rest_states(CIRCUITS))
    order = relief_order(modes, relief_peaks(R, CIRCUITS), CIRCUITS)
    meta = [(; short=c.short, latched=modes[k].latched, mechanism=modes[k].mechanism, family=family(c),
        relief=R.centres[R.windows[k, 1, :] .< 0.8], circle=R.centres[R.windows[k, 2, :] .< 0.8]) for (k, c) in enumerate(CIRCUITS)]
    save_data(out_file, (; meta, order=[CIRCUITS[k].short for k in order], width=R.width, colors=(latched=COLORS.blue, symmetric=COLORS.orange)))
    println(count(m -> m.latched, meta), " latched; ", [m.short for m in meta if m.mechanism == :fold], " by folds; families ",
        Dict(f => count(m -> m.family == f, meta) for f in (:middle, :first, :edge)), "; colours ", COLORS.blue, " ", COLORS.orange)
end

"""Merge the category TE at dt 0.01 (all 41 circuits) and at dt 0.001 (ten circuits) into the notebook's category cache."""
function category_cache(te_file, te_dt_file, meta_file, out_file)
    A, B = load_data(te_file), load_data(te_dt_file)
    names = [r.short for r in A.results]
    meta = load_data(meta_file).meta
    @assert names == [m.short for m in meta] "circuit order differs from CIRCUITS"
    Q = (:te_F, :te_oL, :te_oR, :te_F_02, :te_F_10, :mi_x, :mi_state, :mi_F, :mi_oL, :mi_oR)
    stack_rows(rs, q) = permutedims(reduce(hcat, [r.out[q] for r in rs]))          # circuits × moments
    quantities = Dict(q => stack_rows(A.results, q) for q in Q)
    null99 = Dict(q => [quantile(vec(r.null[q]), 0.99) for r in A.results] for q in keys(A.results[1].null))
    dt = (; names=[r.short for r in B.results], dt=0.001, quantities=Dict(q => stack_rows(B.results, q) for q in Q))
    save_data(out_file, (; names, t=A.t, h=A.h, lags=A.lags, lag=A.lags[2], noise=A.noise, reps=A.reps, nshuf=A.nshuf,
        starts=0.1:0.1:50.0, quantities, null99, dt))
    println("null 99% of TE→F ", extrema(null99[:te_F]))
end

"""
The height at which each circuit's fitted deficit law (H = α₀ − eβ + Γ max(D − D*, 0)) switches its extra term,
against where the category passes into the position:
  h_c     = R_OBJ / D*: where a centred line's deficit reaches D*
  h_paths = along the circuit's own line trials (category_data, d = 0.1 … 50): the first sampled height with a ray on
            the line and D(Δx, h) > D*; the median over the trials where it comes on
  split   = where the position first holds 0.9 bits about the shape, I(C; x) ≥ 0.9
  TE peak = where the category's TE into the force peaks above the split
Rank correlations by group, written to intermediate/category/cat_te_dstar.txt, and the notebook's cache.
"""
function dstar(category_file, meta_file, datadir, out_file)
    fits = merge(load_data(cache_path("categorization_edge_law_all")), load_data(cache_path("categorization_edge_law_all_2")))
    C = load_data(category_file)
    M = load_data(meta_file); meta = Dict(m.short => m for m in M.meta)
    h = C.h
    smooth3(v) = [mean(v[max(1, i - 1):min(end, i + 1)]) for i in eachindex(v)]   # as the notebook's smooth3
    files = Dict(load_data(f).short => f for f in jld2_files(datadir))
    rows = map(C.names) do s
        k = findfirst(==(s), C.names)
        L = fits[s].deficit.L
        x, te, nodes = C.quantities[:mi_x][k, :], C.quantities[:te_F][k, :], C.quantities[:mi_state][k, :]
        sp = findfirst(>=(0.9), x)
        live = [h[m] < 235 && (sp === nothing || m < sp) for m in eachindex(h)]
        sm = [live[m] ? v : -Inf for (m, v) in enumerate(smooth3(te))]
        D = load_data(files[s])
        first_on(n) = findfirst(m -> D.nhit[m, n] > 0 && relief_deficit(D.X[m, n], D.h[m], :line) > L.Dstar, eachindex(D.h))
        on = [D.h[m] for m in (first_on(n) for n in findall(==(:line), D.shape)) if m !== nothing]
        hc = R_OBJ / L.Dstar
        # what the position holds at the D* height (smoothed, interpolated between moments; none above the line's range)
        sx, i = smooth3(x), findfirst(<=(hc), h)
        xD = hc >= 235 || i === nothing || i == 1 ? NaN : sx[i-1] + (sx[i] - sx[i-1]) * (hc - h[i-1]) / (h[i] - h[i-1])
        (; s, k, family=meta[s].family, latched=meta[s].latched, Dstar=L.Dstar, Γ=L.Γ, hc,
            hpaths=isempty(on) ? NaN : median(on), npaths=length(on), nlines=count(==(:line), D.shape),
            split=sp === nothing ? NaN : h[sp], hte=h[argmax(sm)], xD, x, te, nodes)
    end
    R = Dict(r.s => r for r in rows)
    # the comparison: rank correlations (Pearson on tied ranks, HypothesisTests) and median gaps
    function rank_r(a, b)
        ok = .!(isnan.(a) .| isnan.(b))
        t = CorrelationTest(tiedrank(a[ok]), tiedrank(b[ok]))
        (; n=count(ok), ρ=t.r, p=pvalue(t))
    end
    groups = [("all 41", r -> true), ("latched middle bringers", r -> r.latched && r.family == :middle),
        ("symmetric", r -> !r.latched), ("middle first", r -> r.family == :first), ("edge trackers", r -> r.family == :edge)]
    io = IOBuffer()
    @printf(io, "%-5s %-9s %-6s %7s %8s %7s %7s %5s %7s %7s %6s\n", "", "mode", "family", "D*", "Γ", "15/D*", "paths", "n",
        "split", "TE pk", "x@D*")
    for s in M.order
        haskey(R, s) || continue
        r = R[s]
        @printf(io, "%-5s %-9s %-6s %7.4f %8.3g %7.1f %7.1f %5d %7.1f %7.1f %6.2f\n", s, r.latched ? "latched" : "symmetric",
            r.family, r.Dstar, r.Γ, r.hc, r.hpaths, r.npaths, r.split, r.hte, r.xD)
    end
    println(io, "x@D*: bits the position holds at the D* height, I(C; x) smoothed over 3 moments (NaN: D* height above the line's range)")
    println(io)
    for (g, sel) in groups
        G = filter(sel, rows)
        xs = filter(!isnan, [r.xD for r in G])
        length(xs) >= 2 || continue
        @printf(io, "%-24s the position holds at the D* height: median %.2f bits, middle half %.2f … %.2f (%d of %d below h 235)\n",
            g, median(xs), quantile(xs, 0.25), quantile(xs, 0.75), length(xs), length(G))
        for (a, an) in ((:hc, "15/D*"), (:hpaths, "on paths")), (b, bn) in ((:split, "split"), (:hte, "TE peak"))
            av, bv = [getfield(r, a) for r in G], [getfield(r, b) for r in G]
            t = rank_r(av, bv)
            gap = filter(!isnan, av .- bv)
            @printf(io, "%-24s %-8s vs %-7s n %2d  Spearman ρ %5.2f  p %.2g  median (D* height − %s) %6.1f, middle half %6.1f … %6.1f\n",
                g, an, bn, t.n, t.ρ, t.p, bn, median(gap), quantile(gap, 0.25), quantile(gap, 0.75))
        end
    end
    report = String(take!(io))
    print(report)
    mkpath(intermediate("category"))
    write(intermediate("category", "cat_te_dstar.txt"), report)
    # the notebook's cache, in CIRCUITS order (C.names): the rank tests on the same splits and TE peaks as its own
    test_groups = [("all 41", r -> true), ("latched middle bringers", r -> r.latched && r.family == :middle),
        ("symmetric middle bringers", r -> !r.latched && r.family == :middle), ("middle first", r -> r.family == :first),
        ("edge trackers", r -> r.family == :edge)]
    nb_name = Dict(:hc => :h_star, :hpaths => :onset, :split => :split, :hte => :te_peak)     # the notebook's names
    tests = [(; group=g, height=nb_name[height], against=nb_name[against],
                 rank_r(getfield.(filter(sel, rows), height), getfield.(filter(sel, rows), against))...)
             for (g, sel) in test_groups for height in (:hc, :hpaths) for against in (:split, :hte)]
    save_data(out_file, (; names=[r.s for r in rows], Dstar=[r.Dstar for r in rows], Γ=[r.Γ for r in rows],
        onset=[r.hpaths for r in rows], n_on=[r.npaths for r in rows], n_lines=[r.nlines for r in rows],
        split=[r.split for r in rows], te_peak=[r.hte for r in rows], tests))
end

"""
The larger agents' data for the figures' panels on agents of 3 … 8 nodes (2D, 3D, 4D), for the 60 agents, at Euler
steps of 0.01, by the procedures used for the 41 two-node agents: the split (I(C; x) by Ross's estimator, as
category_te), the relief windows (as relief_ablate), the sign row of the ray weights, the kept edge's law and the odd
loop gain of the rest on the mirror plane (rest_modes). Checks the runner against NNodeModel's own trials, and the
procedures against the two-node caches (skip with `--no-check`).
"""
function larger_agents(rest_file, out_file; check_two_node=!("--no-check" in ARGS))
    t0 = time()
    fits = load_data(cache_path("categorization_edge_law_nnode"))
    summary = filter(a -> in_subset("N$(a.N) s$(a.seed)"), nnode_agents())
    rests = Dict(r.name => r for r in load_data(rest_file))
    # the runner against NNodeModel's own trials
    reproduce = map(summary) do a
        c = NM.ncircuit(a.q, a.N)
        maximum(abs(run_view_n(c, d, view) - NM.final_dx(c, line, d)) for d in 1.0:50.0 for (view, line) in ((CIRCLE_VIEW, false), (LINE_VIEW, true)))
    end
    @printf("runner vs NNodeModel's trials: largest |Δx| difference %.3g over the %d agents\n", maximum(reproduce; init=0.0), length(summary)); flush(stdout)
    agents = map(summary) do a
        name = "N$(a.N) s$(a.seed)"
        f = fits[name]
        c = NM.ncircuit(a.q, a.N)
        mi, _ = position_info(c, name)
        rates = relief_windows(c)
        i = argmin(rates)
        L = f.circle[f.e].L
        r = (; name, a.N, a.seed, f.family, f.mode, row=sign_row(c.U[1]), α0p=f.circle[1].L.α0, f.e, b=L.b, V=L.V, ε=L.ε,
            Dstar=f.deficit.L.Dstar, hstar=VISUAL.size / 2 / f.deficit.L.Dstar, mi_x=mi, split=split_height(mi),
            windows=rates, relief=collect(WINDOW_CENTRES)[rates.<0.8], worst=rates[i] < 0.8 ? WINDOW_CENTRES[i] : NaN,
            worst_rate=rates[i], odd_gain=odd_gain(rests[name]), plane_rests=length(rests[name].plane))
        @printf("%-7s %-6s %-9s row %s  α₀⁺ %5.1f  b %7.3f  h* %6.1f  split %6.1f  windows %-14s odd gain %6.3f  (%4.0f s)\n",
            name, r.family, r.mode, r.row, r.α0p, r.b, r.hstar, r.split, isempty(r.relief) ? "none" : "$(r.relief[1])–$(r.relief[end])",
            r.odd_gain, time() - t0); flush(stdout)
        r
    end
    two_node_gain = Dict(r.name => odd_gain(r) for r in values(rests) if r.N == 2)
    check = nothing
    if check_two_node
        final = Dict(f.short => f.q for f in refit_final())
        relief = load_data(cache_path("categorization_strategies_relief"))
        cat = load_data(cache_path("categorization_strategies_category"))
        rows = map(filter(k -> in_subset(relief.names[k]), eachindex(relief.names))) do k
            s = relief.names[k]
            c = NM.ncircuit(final[s], 2)
            rates = relief_windows(c)
            mi, _ = position_info(c, s)
            kc = findfirst(==(s), cat.names)
            (; s, windows=maximum(abs.(rates .- relief.windows[k, 1, :])), mi_x=maximum(abs.(mi .- cat.quantities[:mi_x][kc, :])),
                split=isequal(split_height(mi), (x = cat.quantities[:mi_x][kc, :]; t = findfirst(>=(SPLIT_BITS), x); t === nothing ? NaN : cat.h[t])))
        end
        if !isempty(rows)
            check = (; windows=maximum(r.windows for r in rows), mi_x=maximum(r.mi_x for r in rows),
                splits_equal=count(r.split for r in rows), n=length(rows))
            println("two-node check: relief windows largest difference ", check.windows, ", I(C; x) largest difference ",
                check.mi_x, ", splits equal ", check.splits_equal, " of ", check.n); flush(stdout)
        end
    end
    save_data(out_file, (; agents, two_node_gain, heights=HEIGHTS, centres=collect(WINDOW_CENTRES), width=WINDOW_WIDTH,
        split_bits=SPLIT_BITS, reproduce=maximum(reproduce; init=0.0), check))
end

"""
Bootstrap 95% intervals for the figures' per-agent points (2D and 4D), for every agent, the 41 two-node and the 60 of
3 … 8 nodes, each from its own trials: the split (the category ensemble resampled within each shape, 100 times, and the
split recomputed), the worst relief window (the 100 catch starts resampled, 2000 times), and the damping b with the
law's other constants held (the 50 circle paths resampled, 2000 times, b at the minimum of the summed losses on a grid
of log b refined by a parabola). Resumes from a partial file.
"""
function bootstrap(out_file)
    b_split, b_window, b_b = 100, 2000, 2000
    log_b = range(log(1e-3), log(100.0); length=161)
    noise, reps = BehaviorClasses.NOISE, BehaviorClasses.REPS
    partial = intermediate("figures", "boot.jld2.partial")
    interval(v) = (v = sort(filter(isfinite, v)); isempty(v) ? (NaN, NaN) : (v[max(1, ceil(Int, 0.025length(v)))], v[ceil(Int, 0.975length(v))]))
    # (the nested helpers use names of their own: a name shared with the enclosing function would be shared with it)
    # the split recomputed on the trials idx (columns of the positions X, labels C), as `position_info` and
    # `split_height` do, stopping at the first moment that holds 0.9 bits
    function split_on(X, C, idx, rng)
        Xb, Cb = X[:, idx], C[idx]
        N = length(idx)
        whole = max(std(Xb), 1e-12)
        for m in axes(Xb, 1)
            info = sum(1:reps) do _
                ross_mi_nd(reshape(Xb[m, :] ./ whole .+ noise .* randn(rng, N), 1, N), Cb)
            end / reps
            info >= SPLIT_BITS && return HEIGHTS[m]
        end
        NaN
    end
    # the worst relief window on the starts idx (columns of `caught`), as `relief_windows` and the notebook take it
    function worst_on(caught, idx)
        rates = vec(count(caught[:, idx]; dims=2)) ./ length(idx)
        i = argmin(rates)
        rates[i] < 0.8 ? WINDOW_CENTRES[i] : NaN
    end
    # b at the minimum of the summed per-start losses ℓ (grid × starts) over the starts idx
    function b_on(ℓ, idx)
        f = vec(sum(ℓ[:, idx]; dims=2))
        i = argmin(f)
        (i == 1 || i == length(f)) && return exp(log_b[i])
        a, b, c = f[i-1], f[i], f[i+1]
        den = a - 2b + c
        exp(log_b[i] + (den > 0 ? 0.5 * (a - c) / den : 0.0) * step(log_b))
    end
    # the agents: name, circuit, kept-edge circle law
    two = merge(load_data(cache_path("categorization_edge_law_all")), load_data(cache_path("categorization_edge_law_all_2")))
    large = load_data(cache_path("categorization_edge_law_nnode"))
    agents = [[(; name=f.short, N=2, c=NM.ncircuit(f.q, 2), L=two[f.short].circle[two[f.short].e].L) for f in refit_final()];
        [(; name="N$(a.N) s$(a.seed)", a.N, c=NM.ncircuit(a.q, a.N), L=large["N$(a.N) s$(a.seed)"].circle[large["N$(a.N) s$(a.seed)"].e].L)
         for a in nnode_agents()]]
    agents = filter(a -> in_subset(a.name), agents)
    results = isfile(partial) ? load_data(partial) : Dict{String,Any}()
    t0 = time()
    println("threads ", Threads.nthreads(), ", ", length(agents), " agents"); flush(stdout)
    for a in agents
        haskey(results, a.name) && hasproperty(results[a.name], :split_sd) && continue
        # split
        mi, X = position_info(a.c, a.name)
        C = [trues(500); falses(500)]
        split = split_height(mi)
        boots = fill(NaN, b_split)
        if isfinite(split)
            # each replicate its own generator: `rng_r`, not `rng`, which the loop body below also assigns, and a
            # name assigned in both would be one variable shared by the threads (the old boot_data.jl's race, which
            # made its split replicates irreproducible)
            Threads.@threads for r in 1:b_split
                rng_r = Xoshiro(hash((a.name, :split, r)))
                idx = [rand(rng_r, 1:500, 500); 500 .+ rand(rng_r, 1:500, 500)]
                boots[r] = split_on(X, C, idx, rng_r)
            end
        end
        # relief window
        caught = relief_outcomes(a.c)
        worst = worst_on(caught, 1:length(CATCH_STARTS))
        rng = Xoshiro(hash((a.name, :window)))
        wboots = [worst_on(caught, rand(rng, 1:length(CATCH_STARTS), length(CATCH_STARTS))) for _ in 1:b_window]
        # b, the others held
        paths = [run_view_n(a.c, Float64(d), CIRCLE_VIEW; record=true) for d in 1:50]
        ℓ = zeros(length(log_b), 50)
        Threads.@threads for i in eachindex(log_b)
            Lb = merge(a.L, (; b=exp(log_b[i])))
            for d in 1:50
                x = law_path(Lb, d)
                ℓ[i, d] = mean(abs2, x .- paths[d]) + 10.0 * max(abs(x[end]) - 25, 0.0)^2
            end
        end
        b_full = b_on(ℓ, 1:50)
        rng = Xoshiro(hash((a.name, :b)))
        bboots = [b_on(ℓ, rand(rng, 1:50, 50)) for _ in 1:b_b]
        sd(v) = (v = filter(isfinite, v); length(v) < 2 ? NaN : std(v))
        r = (; a.name, a.N, split, split_ci=interval(boots), split_defined=count(isfinite, boots) / b_split,
            worst, worst_ci=interval(wboots), worst_defined=count(isfinite, wboots) / b_window,
            b=a.L.b, b_full, b_ci=interval(bboots), split_sd=sd(boots), worst_sd=sd(wboots), b_sd=sd(bboots),
            split_boot=boots, worst_boot=wboots, b_boot=bboots)
        results[a.name] = r
        save_data(partial, results)
        @printf("%-7s split %6.1f [%6.1f, %6.1f] (%3.0f%%)  window %6.1f [%6.1f, %6.1f] (%3.0f%%)  b %8.3f grid %8.3f [%8.3f, %8.3f]  (%4.0f s)\n",
            a.name, split, r.split_ci..., 100r.split_defined, worst, r.worst_ci..., 100r.worst_defined, a.L.b, b_full, r.b_ci..., time() - t0)
        flush(stdout)
    end
    save_data(out_file, results)
    rm(partial; force=true)
end

# ── Run ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

const RELIEF_DATA = intermediate("relief", "data")
const RELIEF_TE = intermediate("relief", "te.jld2")
const ABLATE = intermediate("relief", "ablate.jld2")
const CATEGORY_DATA = intermediate("category", "data")
const CATEGORY_DATA_DT = intermediate("category", "data_dt0.001")
const CATEGORY_TE = intermediate("category", "cat_te.jld2")
const CATEGORY_TE_DT = intermediate("category", "cat_te_dt0001.jld2")
const META = intermediate("category", "cat_meta.jld2")
const REST = intermediate("nnode", "rest_modes.jld2")
const NCIRC = subset() === nothing ? 41 : 1          # a subset run's ensembles count as complete with one circuit

println("── Block 1: relief and category")
run_step(() -> relief_data(RELIEF_DATA), OPTS, "relief_data", has_files(RELIEF_DATA, NCIRC))
run_step(() -> relief_te(RELIEF_DATA, RELIEF_TE), OPTS, "relief_te", RELIEF_TE)
run_step(() -> relief_ablate(ABLATE), OPTS, "relief_ablate", ABLATE)
run_step(() -> relief_cache(RELIEF_TE, ABLATE, RELIEF_DATA, cache_path("categorization_strategies_relief")), OPTS,
    "relief_cache", cache_path("categorization_strategies_relief"))
run_step(() -> (category_data(0.01, CATEGORY_DATA); category_data(0.001, CATEGORY_DATA_DT)), OPTS, "category_data",
    has_files(CATEGORY_DATA, NCIRC), has_files(CATEGORY_DATA_DT, min(NCIRC, 10)))
run_step(() -> (category_te(CATEGORY_DATA, CATEGORY_TE); category_te(CATEGORY_DATA_DT, CATEGORY_TE_DT)), OPTS, "category_te",
    CATEGORY_TE, CATEGORY_TE_DT)
run_step(() -> category_meta(cache_path("categorization_strategies_relief"), META), OPTS, "category_meta", META)
run_step(() -> category_cache(CATEGORY_TE, CATEGORY_TE_DT, META, cache_path("categorization_strategies_category")), OPTS,
    "category_cache", cache_path("categorization_strategies_category"))
run_step(() -> dstar(cache_path("categorization_strategies_category"), META, CATEGORY_DATA, cache_path("categorization_strategies_dstar")),
    OPTS, "dstar", cache_path("categorization_strategies_dstar"))
run_step(() -> larger_agents(REST, cache_path("categorization_figures_nnode")), OPTS, "larger_agents", cache_path("categorization_figures_nnode"))
run_step(() -> bootstrap(cache_path("categorization_figures_boot")), OPTS, "bootstrap", cache_path("categorization_figures_boot"))
OPTS.list || draw_figure(2)

println("\n── Block 2: the angle distribution")
OPTS.list || draw_figure(3)

println("\n── Block 3: latched or symmetric")
OPTS.list || draw_figure(4)
