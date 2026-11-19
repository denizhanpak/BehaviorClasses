# BehaviorClasses

Code and data for the figures of the *Behavior classes* chapter. It has three examples of adaptive behaviour, each with
a sensorimotor model and a simpler law that describes it:

| Example | Models | Figures |
|:--|:--|:--|
| **Approach** | Braitenberg's vehicle and Fajen & Warren's steering law | `Approach1`, `Approach2` |
| **Limb oscillation** | Bingham's action–perception oscillator and Beer's one-legged walker | `Oscillator1`–`Oscillator5` |
| **Categorization** | evolved CTRNN agents that catch circles and avoid lines (Beer 2003), and the edge law that describes them | `Categorization1`–`Categorization4` |

plus `Models`, the three models side by side. All figures are in [`figures/`](figures/) as PNG and PDF.

## Quick start: the figures

Install Julia 1.13 (e.g. with [juliaup](https://github.com/JuliaLang/juliaup)), then from the repository's root:

```sh
julia -e 'using Pkg; Pkg.add("Pluto")'                 # once, in your default environment
julia --project -e 'using Pkg; Pkg.instantiate()'       # once: the repository's environment
julia --threads 8 scripts/run_notebooks.jl              # runs the three notebooks, saves figures/ (≈ 8 min)
julia --project scripts/models_figure.jl                # figures/Models.png and .pdf
```

To read the notebooks, open them in Pluto (`julia -e 'using Pluto; Pluto.run()'`, then open
`notebooks/Approach.jl`, `notebooks/Oscillator.jl` or `notebooks/Categorization.jl`). Each notebook explains its
figures panel by panel and carries its own package environment. Approach and Oscillator compute everything they show.
Categorization reads the results of the categorization pipeline from `data/cache/`.

## Rebuilding the categorization analysis

The caches in `data/cache/` are what the pipeline writes. To rebuild them from the agents in `data/inputs/`:

```sh
julia -t 7 --project pipeline/fit_categorization.jl       # ≈ 5.7 h, then draws Categorization1
julia -t 7 --project pipeline/analyze_categorization.jl   # ≈ 2.7 h, then draws Categorization2–4
```

**`fit_categorization.jl`** fits the *edge law* to every agent. Under a falling circle, an agent holds one edge of the
circle at a fixed visual angle; a relief-deficit term makes it turn away from lines.

| Step | What it does | Time (7 threads) |
|:--|:--|:--|
| `edges` | for six representative circuits: which edge of the circle each holds, and at what angle | seconds |
| `readoff` | the law's starting constants, read off each circuit's held-force map | ≈ 1 min |
| `fit_six` | the law's constants fitted to the six circuits' paths (CMA-ES) | ≈ 5 min |
| `fit_all` | all 41 two-node circuits: the circle law, then the deficit term on lines | ≈ 2 h |
| `rest_modes` | the rest states (latched or symmetric) of the 60 agents of 3–8 nodes | minutes |
| `fit_larger` | the 60 larger agents, by `fit_all`'s procedure | ≈ 3.5 h |

**`analyze_categorization.jl`** has three blocks, each ending with its figure.

1. **Relief and category** (`Categorization2`): when the circle's relief and the object's category get into behaviour.
   Steps `relief_data`, `relief_te`, `relief_ablate`, `relief_cache` (≈ 45 min), `category_data`, `category_te`,
   `category_meta`, `category_cache`, `dstar` (≈ 35 min), then the two-node procedures run on the larger agents
   (`larger_agents`, ≈ 15 min) and bootstrap intervals for every agent (`bootstrap`, ≈ 55 min).
2. **The angle distribution** (`Categorization3`): where each agent holds the edge, and its hold rule by sign row.
3. **Latched or symmetric** (`Categorization4`): the latch threshold, and the law's damping against the odd loop gain.

Blocks 2 and 3 compute from the fits and the caches as they draw (`src/figures/Categorization3.jl`, `…4.jl`).

Both scripts skip steps whose outputs exist. Useful flags:

- `--list`: show the steps and whether each is done.
- `--force`: redo everything.
- `--only fit_all,fit_larger`: run only some steps.

Long steps resume from a partial file if interrupted. For a quick check on a few agents, set a subset and a scratch
data directory; the scripts refuse to write a partial run into `data/`:

```sh
PIPELINE_SUBSET="A,N3 s1" PIPELINE_DATA=/tmp/bc julia -t 7 --project pipeline/fit_categorization.jl --only edges,readoff
```

The analysis writes about 1.6 GB of trial ensembles to `data/intermediate/`, which git ignores.

## Layout

```
notebooks/   the three Pluto notebooks (Approach, Oscillator, Categorization)
src/         BehaviorTools.jl, AgentModels.jl   shared by the notebooks: running systems, figure style, models
             BehaviorClasses.jl, categorization/ the package the pipeline uses: simulators, edge law, estimators
             figures/                           the Categorization figures' code, shared by the notebook and pipeline
pipeline/    fit_categorization.jl, analyze_categorization.jl
scripts/     run_notebooks.jl, models_figure.jl
data/inputs/ what the pipeline starts from (not rebuilt here):
               two_node_circuits.jld2   the 41 evolved two-node circuits (their refit parameters)
               refit_final.jld2         the refit of those circuits at Euler step 0.01
               nnode_agents.jld2        the 60 evolved agents of 3–8 nodes (10 per size)
               modes.tsv                each two-node circuit's label, action mode and family
data/cache/  the pipeline's results that the Categorization notebook reads
figures/     every figure, PNG and PDF
```

Data files are JLD2 with one key, `"data"`, holding plain Julia values (NamedTuples, Dicts, arrays):
`using JLD2; JLD2.load("data/cache/categorization_strategies_dstar.jld2", "data")`.

The evolutionary searches that produced the agents, and the refit of the two-node circuits, are not part of this
repository; their results are the inputs above. The notebooks' text and some code comments name the development
notebooks the figures were distilled from (e.g. `CategorizationEdgeLaw.jl`, `ApproachBehavior.jl`); those are not
included either.

## License

MIT; see [LICENSE](LICENSE).
