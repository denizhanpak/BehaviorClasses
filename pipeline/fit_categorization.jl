# Fit the categorization agents: the edge law (with its relief-deficit term) for the 41 two-node circuits and the 60
# evolved agents of 3 … 8 nodes, and the larger agents' rest modes. Then draws figure Categorization1.
#
#   julia -t 7 --project pipeline/fit_categorization.jl            # every step still to do, then the figure
#   julia -t 7 --project pipeline/fit_categorization.jl --list     # the steps and which are done
#   julia -t 7 --project pipeline/fit_categorization.jl --force    # redo every step
#   julia -t 7 --project pipeline/fit_categorization.jl --only fit_all,fit_larger
#
# Steps (inputs: data/inputs/; a step is skipped when its outputs exist):
#   Edge law on six representative circuits (A, D, K, J, F, P: one per family × action mode)
#     edges        does each circuit hold an edge of the circle at a fixed visual angle, and which?   seconds
#     readoff      the edge law's starting constants read off each circuit's held-force map          ≈ 1 min
#     fit_six      the law's four constants fitted to each circuit's circle paths (CMA-ES)            ≈ 5 min
#   Edge law on every agent
#     fit_all      all 41 two-node circuits: circles, then the deficit term on lines       ≈ 2 h  → 2 caches
#     rest_modes   the larger agents' rest states: latched or symmetric, as for two nodes    a few min
#     fit_larger   the 60 agents of 3 … 8 nodes, by fit_all's procedure                     ≈ 3.5 h → 1 cache
#   Figure Categorization1 (agent A and its edge law), from data/cache/.
# Runtimes are on 7 threads. Writes intermediate/edge_law/, intermediate/nnode/ and cache/categorization_edge_law_*.jld2
# under PIPELINE_DATA (default data/). For a smoke test on a few agents set PIPELINE_SUBSET (e.g. "A,K" or "N3 s1") and
# PIPELINE_DATA to a scratch directory. Then run pipeline/analyze_categorization.jl.
using BehaviorClasses, Statistics, Random, Printf, DelimitedFiles
import BehaviorClasses: H
check_output_target()
const OPTS = pipeline_options(ARGS)

# ── Edge law on the six ────────────────────────────────────────────────────────────────────────────────────────────────

"""
Edge law, step 1. For each of the six circuits, its circle paths from d = 1 … 50 (the other side is their mirror image),
and along them the bearings of the circle's two edges seen from the agent's centre: outer edge |φ| + β, inner edge
|φ| − β, φ = atan(Δx/h) the centre's bearing, sin β = 15/r its angular half-width. Prints medians and spreads by height.
"""
function edges(out)
    bands = [(210.0, 250.0), (170.0, 210.0), (130.0, 170.0), (90.0, 130.0), (60.0, 90.0), (15.0, 60.0)]
    results = Dict{String,Any}()
    for l in filter(in_subset, PICKS)
        k = index_of(l)
        c = circuit(CIRCUITS[k].q)
        paths = [run_circuit(c, :circle, Float64(d)) for d in 1:50]
        outer = fill(NaN, STEPS, 50)
        inner = fill(NaN, STEPS, 50)
        for (j, x) in enumerate(paths), i in 1:STEPS
            outer_points(x[i], :circle, H[i]) === nothing && continue
            φ, β = geometry(x[i], H[i])
            outer[i, j] = abs(φ) + β
            inner[i, j] = abs(φ) - β
        end
        results[l] = (; k, paths, outer, inner)
        m = get(MODES, CIRCUITS[k].name, ("?", "?", "?"))
        @printf("\n%s  (%s)  τ = %.2f   ends |Δx| median %.2f\n", l, join(m, " "), c.τ[1], median(abs(x[end]) for x in paths))
        println("  band        outer edge: median [10%, 90%]    inner edge: median [10%, 90%]   seen share")
        for (lo, hi) in bands
            rows = findall(i -> lo <= H[i] < hi, 1:STEPS)
            o = filter(!isnan, vec(outer[rows, :]))
            n = filter(!isnan, vec(inner[rows, :]))
            isempty(o) && (println(@sprintf("  h %3d–%3d   (not seen)", lo, hi)); continue)
            q(v) = (median(v), quantile(v, 0.1), quantile(v, 0.9))
            @printf("  h %3d–%3d   %6.2f [%6.2f, %6.2f]        %6.2f [%6.2f, %6.2f]        %.2f\n", lo, hi,
                q(o)..., q(n)..., length(o) / (length(rows) * 50))
        end
    end
    save_data(out, results)
end

"""
Edge law, step 2. Each circuit's held-force map under a circle (the force of its settled nodes with the agent and the
circle held still, `held_forces`; the mean of the two stable forces where the circuit latches), on the circle's left
(Δx > 0), as a function of the held edge's angle a = |φ| + eβ at each height of its hold. Reads off the starting
constants of the edge law: α₀ = the angle where the push turns from away (a < α₀) to toward (a > α₀), nearest the edge
angle the circuit's paths keep (step 1); V = the largest |push| within 5° of it; ε = 2 V H²/s from the slope s of the
push across it (mean push 0–3° inside minus 0–3° outside, per degree), H the hold bearing there.
"""
function readoff(edges_file, out_file)
    E = load_data(edges_file)
    band = Dict("A" => 100:5:200, "D" => 100:5:200, "K" => 160:5:210, "J" => 160:5:210, "F" => 100:5:190, "P" => 100:5:190)
    DX = 0.0:0.25:90.0
    # the circuit's median held edge angle along its paths at heights within 2.5 of h
    function path_edge(l, h)
        r = E[l]
        A = EDGE[l] == 1 ? r.outer : r.inner
        rows = findall(i -> abs(H[i] - h) <= 2.5, 1:STEPS)
        median(filter(!isnan, vec(A[rows, :])))
    end
    out = Dict{String,Any}()
    for l in filter(in_subset, PICKS)
        k, e = index_of(l), EDGE[l]
        c = circuit(CIRCUITS[k].q)
        rows = []
        for h in band[l]
            Fs = fetch.([Threads.@spawn(let (lo, hi) = held_forces(c, :circle, -Δx, Float64(h)); (lo + hi) / 2 end) for Δx in DX])
            geo = [geometry(Δx, Float64(h)) for Δx in DX]
            a = [g[1] + e * g[2] for g in geo]
            target = path_edge(l, h)
            # restoring crossings: away (> 0) just inside, toward (≤ 0) just outside, a increasing with Δx
            cross = Float64[]
            for i in 1:length(DX)-1
                (isnan(Fs[i]) || isnan(Fs[i+1])) && continue
                Fs[i] > 0 && Fs[i+1] <= 0 || continue
                push!(cross, a[i] + (a[i+1] - a[i]) * Fs[i] / (Fs[i] - Fs[i+1]))
            end
            if isempty(cross)
                # no away band at this height: the map pulls toward the centre everywhere (the hold is the centre, so
                # α₀ ≤ β here); record the pull's slope by the centre (mean push at φ 0–3° per degree) and V near it
                φs = [g[1] for g in geo]
                cen = [Fs[i] for i in eachindex(φs) if 0 < φs[i] <= 3 && !isnan(Fs[i])]
                nearc = [abs(Fs[i]) for i in eachindex(φs) if φs[i] <= 5 && !isnan(Fs[i])]
                push!(rows, (; h, α=NaN, target, V=maximum(nearc), s=-mean(cen) / 1.5, Hb=NaN, ε=NaN, ncross=0))
                continue
            end
            α = cross[argmin(abs.(cross .- target))]
            near = findall(x -> abs(x - α) <= 5 && !isnan(Fs[findfirst(==(x), a)]), a)
            V = maximum(abs.(Fs[near]))
            inside = [Fs[i] for i in eachindex(a) if α - 3 <= a[i] < α && !isnan(Fs[i])]
            outside = [Fs[i] for i in eachindex(a) if α < a[i] <= α + 3 && !isnan(Fs[i])]
            s = (mean(inside) - mean(outside)) / 3
            Hb = α - e * asind(R_OBJ / h)     # the hold bearing (β at φ ≈ 0; the correction is below 0.1°)
            push!(rows, (; h, α, target, V, s, Hb, ε=2V * Hb^2 / s, ncross=length(cross)))
        end
        crossed = filter(r -> !isnan(r.α), rows)
        if isempty(crossed)
            # the centre hold: start α₀ at the edge angle the paths keep, ε from the centre pull with |H| ≥ 1°
            α0 = median(r.target for r in rows)
            V = median(r.V for r in rows)
            ε = median(V * max((α0 - e * asind(R_OBJ / r.h))^2, 1.0) / max(abs(r.s), 1e-3) for r in rows)
            centre = true
        else
            α0, V, ε = median(r.α for r in crossed), median(r.V for r in crossed), median(r.ε for r in crossed)
            centre = false
        end
        out[l] = (; k, e, α0, V, ε, centre, rows)
        @printf("\n%s  e = %+d   α₀ = %.2f°  V = %.2f  ε = %.3g%s\n", l, e, α0, V, ε, centre ? "   (no away band: centre hold, α₀ from the paths)" : "")
        println("    h    α₀(h)  path edge   V      slope/°   H      ε      crossings")
        for r in rows
            @printf("  %4d  %6.2f  %6.2f   %6.2f  %7.3f  %6.2f  %8.3g   %d\n", r.h, r.α, r.target, r.V, r.s, r.Hb, r.ε, r.ncross)
        end
    end
    save_data(out_file, out)
end

"""
Edge law, step 3. For each of the six circuits, the edge law's four constants (α₀, V, ε, b; the edge e fixed from step
1) fitted to the circuit's circle paths from d = 1 … 50 with the circle loss (path error + catch penalty past 25,
λ = 10), the law stepping at dt 0.01. CMA-ES (with its stall stop): seed 1 from the read-off constants (step 2) with
b = 1, seeds 2–4 from the same point with other random streams. Also scores the read-off law itself (nothing fitted,
b = 1). Its fits are the CMA-ES starts of fit_all and fit_larger. Resumes from a partial file.
"""
function fit_six(readoff_file, out)
    R = load_data(readoff_file)
    partial = out * ".partial"
    scale = [2.0, 0.5, 1.0, 0.5]      # α₀ (degrees), log V, log ε, log b
    nseeds, maxgen, stall = 4, 300, 60
    results = isfile(partial) ? load_data(partial) : Dict{String,Any}()
    for l in filter(in_subset, PICKS)
        haskey(results, l) && continue
        t0 = time()
        k, e, r = index_of(l), EDGE[l], R[l]
        paths = circle_paths(k)
        z0 = [r.α0, log(r.V), log(r.ε), 0.0]
        run_of(L) = d -> law_path(L, d)
        loss(z) = circle_loss(run_of(edge_params(z, e)), paths)
        L0 = edge_params(z0, e)
        base = (L=L0, loss=loss(z0), R2=path_R2(run_of(L0), paths))
        runs = [cmaes(loss, z0, scale; σ0=0.5, λ=24, maxgen, rng=MersenneTwister(i), stall) for i in 1:nseeds]
        best = runs[argmin([q.f for q in runs])]
        L = edge_params(best.p, e)
        run = run_of(L)
        ends = [run(d)[end] for d in 1:50]
        results[l] = (; k, name=CIRCUITS[k].name, e, z=best.p, L, loss=best.f, R2=path_R2(run, paths),
            ends, circuit_ends=[x[end] for x in paths], readoff=base, seeds=[(f=q.f, gens=q.gens) for q in runs])
        save_data(partial, results)
        @printf("%s  read-off: α₀ %.2f V %.2f ε %.3g b 1 → loss %.2f R² %.3f |  fit: α₀ %.2f V %.2f ε %.3g b %.3g → loss %.2f R² %.3f  (%.0f s; seeds %s)\n",
            l, L0.α0, L0.V, L0.ε, base.loss, base.R2, L.α0, L.V, L.ε, L.b, best.f, results[l].R2, time() - t0,
            join([@sprintf("%.2f/%d", q.f, q.gens) for q in runs], " "))
        flush(stdout)
    end
    save_data(out, results)
    rm(partial; force=true)
end

# ── Edge law on every agent ────────────────────────────────────────────────────────────────────────────────────────────

"""The one-line report of an agent's two-stage fit."""
function report_fit(label, mode, family, f, lpaths, t0)
    c, d = f.circle[f.e], f.deficit
    @printf("%-7s %-9s %-6s e %+d (circle loss %.2f vs %.2f) α₀ %.2f V %.3g ε %.3g b %.3g R² %.3f | lines: Γ=0 R² %.3f → Γ %.4g D* %.4f R² %.3f avoided %d/50 (circuit %d/50) | circle shift %.1e  (%.0f s)\n",
        label, mode, family, f.e, f.circle[1].loss, f.circle[-1].loss, c.L.α0, c.L.V, c.L.ε, c.L.b, c.R2, f.gamma0.R2,
        d.L.Γ, d.L.Dstar, d.lines.R2, count(x -> abs(x) >= 35, d.lines.ends), count(x -> abs(x[end]) >= 35, lpaths),
        d.circles.shift, time() - t0)
    flush(stdout)
end

"""
The edge law with the relief-deficit term for all 41 two-node circuits (user, 2026-10-21: "fit it to the other agents
as well"). For each circuit, two stages (`fit_edge_and_deficit`):
1. circles: the four constants (α₀, V, ε, b) for each edge e = ±1, CMA-ES from four starts: the six fits of that edge
   (outer A D K J; inner F P, and A K's constants with e = −1); the edge with the lower circle loss is kept;
2. lines: Γ and D* with that circle law frozen (`fit_deficit`: a grid of (log Γ, D*), then CMA-ES from its three best
   cells at least 0.05 apart in D*).
The circuits are split alternately into two caches (the old run's two processes), which the notebook merges.
"""
function fit_all(six_file)
    starts = edge_starts(load_data(six_file))
    # circuits in CIRCUITS order, the six first so their check comes early
    order_all = sortperm(1:length(CIRCUITS); by=k -> (MODES[CIRCUITS[k].name][1] in PICKS ? 0 : 1, k))
    for part in (1, 2)
        out = cache_path(part == 2 ? "categorization_edge_law_all_2" : "categorization_edge_law_all")
        partial = intermediate("edge_law", "all_fit_part$(part).jld2.partial")
        results = isfile(partial) ? load_data(partial) : Dict{String,Any}()
        for k in order_all[part:2:end]
            name = CIRCUITS[k].name
            short, mode, family = MODES[name]
            (haskey(results, short) || !in_subset(short)) && continue
            t0 = time()
            cpaths, lpaths = circle_paths(k), line_paths(k)
            f = fit_edge_and_deficit(cpaths, lpaths, starts)
            results[short] = (; k, name, short, mode, family, f.e, f.circle, gamma0=f.gamma0, f.deficit,
                circuit_circle_ends=[x[end] for x in cpaths], circuit_line_ends=[x[end] for x in lpaths])
            save_data(partial, results)
            report_fit(short, mode, family, f, lpaths, t0)
        end
        save_data(out, results)
        rm(partial; force=true)
        println("part ", part, ": wrote ", out, " (", length(results), " circuits)")
    end
end

"""
Does the two-node action modes' split (latched / symmetric rest with nothing in view) hold for N = 3 … 8? Each evolved
agent, and the 41 two-node circuits as the check, by `NNodeModel.study`: every rest the nodes reach with nothing in view
(from 162 starts), every rest on the mirror plane by Newton, and their stability. Writes the table beside the file.
"""
function rest_modes(out)
    modes = readdlm(inputs_dir("modes.tsv"), '\t', String)
    known = Dict(modes[k, 2] => (modes[k, 3], modes[k, 4]) for k in axes(modes, 1))
    jobs = [[(; N=2, name=f.short, q=f.q, mode=known[f.short][1], family=known[f.short][2]) for f in refit_final()];
            [(; r.N, name="N$(r.N) s$(r.seed)", r.q, r.mode, r.family) for r in nnode_agents()]]
    jobs = filter(j -> in_subset(j.name), jobs)
    res = Vector{Any}(undef, length(jobs))
    done = Threads.Atomic{Int}(0)
    Threads.@threads :dynamic for k in eachindex(jobs)
        j = jobs[k]
        res[k] = (; j..., NNodeModel.study(j.q, j.N)...)
        Threads.atomic_add!(done, 1)
        done[] % 10 == 0 && println("  ", done[], " / ", length(jobs)); flush(stdout)
    end
    save_data(out, res)
    write(joinpath(dirname(out), "rest_modes.txt"), NNodeModel.report(res))
    write(joinpath(dirname(out), "rest_modes_table.txt"), NNodeModel.tabulate(res))
end

"""
The edge law with the relief-deficit term for the 60 agents of 3 … 8 nodes (user 2026-10-22: "fit the edge law to all
the more than two node agents"). fit_all's procedure unchanged, only the paths come from `NNodeModel` (dt 0.01, the 868
positions 0.1 apart). Checks first that for N = 2 the N-node paths are the two-node circuits' own, for all 41.
"""
function fit_larger(six_file)
    starts = edge_starts(load_data(six_file))
    out = cache_path("categorization_edge_law_nnode")
    partial = intermediate("edge_law", "nnode_fit.jld2.partial")
    agents = nnode_agents()
    sub = round(Int, VISUAL.dt / NM.DT)
    # an agent's path at the 868 positions of `run_circuit` (d, then every 10th Euler step of 0.01)
    npath868(q, N, line, d) = [Float64(d); NM.npath(q, N, line, d)[sub:sub:end]]
    shape_paths(a, line) = [npath868(a.q, a.N, line, d) for d in 1:50]
    worst = maximum(k -> maximum(d -> maximum(abs, npath868(CIRCUITS[k].q, 2, false, d) .-
        run_circuit(circuit(CIRCUITS[k].q), :circle, Float64(d))), (1, 25, 50)), eachindex(CIRCUITS))
    @printf("N = 2 check: nmodel paths vs run_circuit, largest |Δx| difference %.1e\n", worst)
    worst < 1e-9 || error("nmodel and run_circuit disagree")
    results = isfile(partial) ? load_data(partial) : Dict{String,Any}()
    for (k, a) in enumerate(agents)
        short = "N$(a.N) s$(a.seed)"
        (haskey(results, short) || !in_subset(short)) && continue
        t0 = time()
        cpaths, lpaths = shape_paths(a, false), shape_paths(a, true)
        f = fit_edge_and_deficit(cpaths, lpaths, starts)
        results[short] = (; k, short, a.N, a.seed, a.mode, a.family, f.e, f.circle, gamma0=f.gamma0, f.deficit,
            circuit_circle_ends=[x[end] for x in cpaths], circuit_line_ends=[x[end] for x in lpaths])
        save_data(partial, results)
        report_fit(short, a.mode, a.family, f, lpaths, t0)
    end
    save_data(out, results)
    rm(partial; force=true)
    println("wrote ", out, " (", length(results), " agents)")
end

# ── Run ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

const EDGES = intermediate("edge_law", "edges.jld2")
const READOFF = intermediate("edge_law", "readoff.jld2")
const SIX = intermediate("edge_law", "six_fits.jld2")
const REST = intermediate("nnode", "rest_modes.jld2")

run_step(() -> edges(EDGES), OPTS, "edges", EDGES)
run_step(() -> readoff(EDGES, READOFF), OPTS, "readoff", READOFF)
run_step(() -> fit_six(READOFF, SIX), OPTS, "fit_six", SIX)
run_step(() -> fit_all(SIX), OPTS, "fit_all", cache_path("categorization_edge_law_all"), cache_path("categorization_edge_law_all_2"))
run_step(() -> rest_modes(REST), OPTS, "rest_modes", REST)
run_step(() -> fit_larger(SIX), OPTS, "fit_larger", cache_path("categorization_edge_law_nnode"))
OPTS.list || draw_figure(1)
