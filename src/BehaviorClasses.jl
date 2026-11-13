"""
    BehaviorClasses

The code behind the paper's notebooks. `BehaviorTools` and `AgentModels` are the notebooks' shared modules (the notebooks
include those two files directly). The rest is the categorization pipeline (pipeline/), which rebuilds the
Categorization notebook's caches (data/cache/) from the root inputs (data/inputs/):

- `two_node.jl`: the two-node circuits' simulator (`Circuit`, `circuit`, `run_circuit`, `held_forces`);
- `io.jl`: where the stages read and write, and the root inputs (`CIRCUITS`, `MODES`, `refit_final`, `nnode_agents`);
- `cmaes.jl`: the CMA-ES of the edge-law fits;
- `edge_law.jl`: the view's outermost points, the edge law, the relief deficit and their two-stage fit;
- `views.jl`: the §7 views (`view_lengths!`, `run_view`);
- `ross_mi.jl`: Ross's nearest-neighbour MI with a discrete label (`ross_mi_nd`);
- `action_modes.jl`: each circuit's rest, action mode and family, and the order of the §7 figures;
- `nnode_model.jl`: `NNodeModel`, the 3 … 8-node agents' simulator and rest-mode study;
- `nnode_views.jl`: the larger agents' runner through the §7 views and the two-node procedures run on them.
"""
module BehaviorClasses

using LinearAlgebra, Statistics, Random, Printf
import JLD2

include("BehaviorTools.jl")
include("AgentModels.jl")
using .BehaviorTools, .AgentModels

export BehaviorTools, AgentModels
# the notebooks' shared names, so a script needs only `using BehaviorClasses`
export COLORS, INK, INK_SECONDARY, INK_MUTED, MODEL_COLORS, figure_style!, save_figure
export VISUAL, σ, ray_length, ray_angles

include(joinpath("categorization", "two_node.jl"))
include(joinpath("categorization", "io.jl"))
include(joinpath("categorization", "cmaes.jl"))
include(joinpath("categorization", "edge_law.jl"))
include(joinpath("categorization", "views.jl"))
include(joinpath("categorization", "ross_mi.jl"))
include(joinpath("categorization", "action_modes.jl"))
include(joinpath("categorization", "nnode_model.jl"))
include(joinpath("categorization", "nnode_views.jl"))

# io
export REPO, inputs_dir, data_dir, intermediate, cache_path, subset, in_subset, check_output_target, save_data, load_data,
    pipeline_options, run_step, has_files, draw_figure
export CIRCUITS, MODES, refit_final, nnode_agents
# two-node circuits
export Circuit, circuit, nnodes, node_out, mirror_sums!, run_circuit, stable_forces, held_forces, STEPS, SIDE
# edge law
export cmaes, outer_points, R_OBJ, PICKS, EDGE, index_of, geometry, line_geometry, in_view, run_edge_law, law_path,
    edge_params, circle_paths, line_paths, circle_loss, line_loss, path_R2, relief_deficit, deficit_path, fit_deficit,
    EDGE_SCALE, edge_starts, fit_edge_and_deficit
# views, estimator, action modes
export VIEWS, CIRCLE_VIEW, LINE_VIEW, FLAT_VIEW, view_label, view_lengths!, run_view
export tmap, ross_mi_nd, ross_cmi, ROSS_PARTITION_THREADS
export rest_states, action_modes, relief_peaks, relief_order, family
# larger agents
export NNodeModel, NM, run_view_n, relief_outcomes, relief_windows, position_info, split_height, odd_gain, sign_row

function __init__()
    load_inputs!()
    n = tryparse(Int, get(ENV, "PIPELINE_ROSS_THREADS", ""))
    n === nothing || (ROSS_PARTITION_THREADS[] = n)
    nothing
end

end # module BehaviorClasses
