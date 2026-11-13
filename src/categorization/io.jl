# Where the pipeline reads and writes, and the root inputs (data/inputs/).
#
#   PIPELINE_DATA    the data root the stages write to (default: the repository's data/). Intermediates go to
#                    <root>/intermediate/, the notebook's caches to <root>/cache/.
#   PIPELINE_SUBSET  a comma-separated list of circuit shorts ("A,#16") or agent names ("N3 s1") to restrict a stage to,
#                    for smoke tests. A subset run must write somewhere other than the repository's data/ (set
#                    PIPELINE_DATA), so partial results never land in data/cache.
#   PIPELINE_ROSS_THREADS  see `ross_mi_nd`: the thread count its work partition is cut for (default: this session's).

"""The repository's root directory."""
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

"""The root inputs: what the pipeline starts from and does not rebuild."""
inputs_dir(parts...) = joinpath(REPO, "data", "inputs", parts...)

"""The data root the stages write to (`PIPELINE_DATA`, default the repository's data/)."""
data_dir() = abspath(get(ENV, "PIPELINE_DATA", joinpath(REPO, "data")))

"""A path under `<data root>/intermediate/`."""
intermediate(parts...) = joinpath(data_dir(), "intermediate", parts...)

"""The notebook cache `name` (without extension) under `<data root>/cache/`."""
cache_path(name) = joinpath(data_dir(), "cache", name * ".jld2")

"""The names of `PIPELINE_SUBSET`, or `nothing` for the whole stage."""
function subset()
    s = strip(get(ENV, "PIPELINE_SUBSET", ""))
    isempty(s) ? nothing : [String(strip(x)) for x in split(s, ',') if !isempty(strip(x))]
end
"""Whether `name` is in `PIPELINE_SUBSET` (always, without one)."""
in_subset(name) = (s = subset(); s === nothing || name in s)

"""Refuses a subset run that would write into the repository's data/ (the notebook's caches)."""
function check_output_target()
    if subset() !== nothing && data_dir() == abspath(joinpath(REPO, "data"))
        error("PIPELINE_SUBSET is set: set PIPELINE_DATA to a scratch directory so partial results stay out of data/")
    end
    nothing
end

"""Writes `x` as a JLD2 file with the one key \"data\" (through a temporary file, so a killed run leaves no half file)."""
function save_data(path, x)
    mkpath(dirname(path))
    tmp = path * ".part"
    JLD2.jldsave(tmp; data=x)
    mv(tmp, path; force=true)
    path
end
"""Reads the key \"data\" of a JLD2 file."""
load_data(path) = JLD2.load(path, "data")

# ---------------------------------------------------------------------------------------------------------------------
# The root inputs

const CircuitEntry = @NamedTuple{name::String, letter::String, short::String, round::Int64, search::String, q::Vector{Float64}}

"""
The 41 two-node circuits (data/inputs/two_node_circuits.jld2), as the `CIRCUITS` cell of
ActiveCategorizationTwoNodeSearch.jl and CategorizationStrategies.jl has them (the two cells are identical): each has
its `name` (as on the circuit-search page), its `letter` (A to Q, else empty), a `short` label for the figures, its
`round`, the `search` it came from, and its free vector `q` (`circuit`: log τ, θ, W_s, W_c, U₁ … U₇), the refitted one
(data/inputs/refit_final.jld2). In the order of the searches, which numbers the per-circuit files.
"""
const CIRCUITS = CircuitEntry[]

"""Each circuit's short label, action mode and family by its name (data/inputs/modes.tsv), as the scripts read it."""
const MODES = Dict{SubString{String},NTuple{3,SubString{String}}}()

function load_inputs!()
    empty!(CIRCUITS)
    append!(CIRCUITS, load_data(inputs_dir("two_node_circuits.jld2")))
    empty!(MODES)
    merge!(MODES, Dict(l[1] => (l[2], l[3], l[4]) for l in split.(readlines(inputs_dir("modes.tsv")), '\t')))
    nothing
end

const _REFIT = Ref{Any}(nothing)
const _AGENTS = Ref{Any}(nothing)

"""
The refit of the 41 two-node circuits at Euler step 0.01 (data/inputs/refit_final.jld2, the refit stage's final
output, not rebuilt here): name, short, letter, the old vector `q_old`, the refit's `q_refit`, the final `q` and
whether a guard run replaced the refit (`guarded`). Its order is the refit's, not `CIRCUITS`'.
"""
refit_final() = (_REFIT[] === nothing && (_REFIT[] = load_data(inputs_dir("refit_final.jld2"))); _REFIT[])

"""
The 60 evolved agents of 3 … 8 nodes (data/inputs/nnode_agents.jld2: the `agents` of the evolution's summary, not
rebuilt here): N, seed, free vector q (`NNodeModel.ncircuit`), their task scores, action `mode` and `family`.
"""
nnode_agents() = (_AGENTS[] === nothing && (_AGENTS[] = load_data(inputs_dir("nnode_agents.jld2"))); _AGENTS[])

# ---------------------------------------------------------------------------------------------------------------------
# Steps of the two pipeline scripts (pipeline/fit_categorization.jl, pipeline/analyze_categorization.jl)

"""
The command-line options of a pipeline script: `--force` (redo steps whose outputs exist), `--only a,b` (run only
these steps), `--list` (print the steps and whether each is done, run nothing).
"""
function pipeline_options(args)
    i = findfirst(==("--only"), args)
    only = i === nothing ? String[] : String.(split(args[i+1], ','))
    (; force="--force" in args, list="--list" in args, only)
end

"""
    run_step(f, opts, name, outputs...)

Run the step `name` (the function `f`) unless every one of its `outputs` exists (paths, or zero-argument functions that
return whether that output is there), `opts.force` is set, or `opts.only` names other steps. Prints what it does and how
long it took. With `opts.list`, only reports.
"""
function run_step(f, opts, name, outputs...)
    there(o) = o isa Function ? o() : isfile(o)
    done = !isempty(outputs) && all(there, outputs)
    if opts.list
        println(rpad(name, 22), done ? "done" : "to do"); return nothing
    end
    isempty(opts.only) || name in opts.only || return nothing
    if done && !opts.force && isempty(opts.only)
        println("[skip] ", name, " (outputs exist; --force to redo)"); return nothing
    end
    println("\n[run]  ", name); flush(stdout)
    t0 = time()
    f()
    println("[done] ", name, " in ", round(Int, time() - t0), " s"); flush(stdout)
    nothing
end

"""Whether a directory holds at least `n` `.jld2` files (a per-circuit ensemble is complete)."""
has_files(dir, n) = () -> isdir(dir) && count(endswith(".jld2"), readdir(dir)) >= n

"""
    draw_figure(k)

Draw and save the Categorization figure `k` (1–4) from the caches in data/cache/, by loading its module in
src/figures/ (the code the Categorization notebook shows). Skipped when the run writes elsewhere (PIPELINE_DATA) or is a subset run.
"""
function draw_figure(k)
    if subset() !== nothing || data_dir() != abspath(joinpath(REPO, "data"))
        println("[skip] figure Categorization$k (the figures read data/cache/; this run writes elsewhere)"); return nothing
    end
    dir = joinpath(REPO, "src", "figures")
    isdefined(Main, :CategorizationCommon) || Base.include(Main, joinpath(dir, "CategorizationCommon.jl"))
    # figures 2–4 also use the larger agents' data and the bootstrap (analyze_categorization.jl's caches)
    k >= 2 && !isdefined(Main, :CategorizationLarger) && Base.include(Main, joinpath(dir, "CategorizationLarger.jl"))
    Base.include(Main, joinpath(dir, "Categorization$k.jl"))
    println("[figure] figures/Categorization$k.png and .pdf")
    nothing
end
