# The larger agents' data and the bootstrap intervals the Categorization figures 2–4 use (`LARGER`, `BOOT`; data/cache/,
# written by pipeline/analyze_categorization.jl).
# The code of notebooks/Categorization.jl, which loads these modules and explains each panel; run there or by
# pipeline/analyze_categorization.jl (Categorization2–4) and pipeline/fit_categorization.jl (Categorization1).
module CategorizationLarger

using Markdown
using MathematicalSystems
import HybridSystems, LazySets
using OrdinaryDiffEqTsit5
using Plots
using LinearAlgebra
import JLD2
import KernelDensity, StatsBase, Roots, Random
using ..CategorizationCommon
using ..BehaviorClasses.BehaviorTools, ..BehaviorClasses.AgentModels

# ── cell 71b805f6 ────────────────────────────────────────────────────────────
begin
    """The brain sizes of the larger agents (`data/inputs/nnode_agents.jld2`: 10 evolved for each size)."""
    const LARGER_SIZES = 3:8

    """
    The 60 agents of 3 … 8 nodes (`pipeline/analyze_categorization.jl`, step larger_agents, cached): for each, its `name`, size `N`, `family`,
    action `mode`, the motor pair's sign `row`, from the edge-law fit (`pipeline/fit_categorization.jl`, step fit_larger) `α0p`, the kept
    edge's damping `b` and `hstar` = 15/D*, and, run by the two-node procedures, what the position holds `mi_x` (at
    `heights`), the `split`, the catch rate in each relief window `windows` (at `centres`), the windows that count
    `relief` and the most damaging `worst` (NaN if none counts), and the odd loop gain at the symmetric rest `odd_gain`
    (`pipeline/fit_categorization.jl`, step rest_modes). Also `two_node_gain`, the same gain for the 41 two-node agents, and the checks: the runner
    reproduces `src/categorization/nnode_model.jl`'s trials (`reproduce`, the largest difference in Δx) and, run on the 41 two-node agents,
    their relief windows and position information in the caches §2 reads (`check`).
    """
    LARGER = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "cache", "categorization_figures_nnode.jld2"), "data")
end

# ── cell 5b900caa ────────────────────────────────────────────────────────────
"""
Bootstrap resamples for every agent, two-node and larger (`pipeline/analyze_categorization.jl`, step bootstrap, cached, keyed by name), with
their standard deviations and 95% intervals: its split (`split_sd`, `split_ci`, the category ensemble resampled within
each shape, 100 times), its most damaging relief window (`worst_sd`, `worst_ci`, the 100 circle starts resampled, 2000
times, over the resamples in which a window counts; `worst_defined` their share) and its damping b (`b_sd`, `b_ci`, the
50 circle starts resampled, 2000 times, b refitted with the law's other constants held).
"""
BOOT = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "cache", "categorization_figures_boot.jld2"), "data")

# Export every name defined here, so the analysis scripts and the notebook see them all.
eval(Expr(:export, filter(n -> !startswith(string(n), '#') && n ∉ (:eval, :include, nameof(@__MODULE__)),
    names(@__MODULE__; all=true))...))

end
