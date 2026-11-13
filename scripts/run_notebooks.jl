# Run the figure notebooks headless in Pluto: every cell runs, the figures are saved to figures/, and any cell that
# errors is reported. Each notebook carries its own package environment, which Pluto installs on the first run.
#
#   julia --threads 8 scripts/run_notebooks.jl                     # all three notebooks
#   julia --threads 8 scripts/run_notebooks.jl Approach Oscillator  # some of them
#
# Needs Pluto in the environment you start Julia with (e.g. `julia -e 'using Pkg; Pkg.add("Pluto")'` once, in the
# default environment). Add --save to write the notebooks back after the run (refreshes their embedded Manifest).
import Pluto

const NOTEBOOKS = joinpath(@__DIR__, "..", "notebooks")
save = "--save" in ARGS
names = filter(!=("--save"), ARGS)
names = isempty(names) ? ["Approach", "Oscillator", "Categorization"] : names

session = Pluto.ServerSession()
session.options.server.disable_writing_notebook_files = !save
session.options.compiler.threads = string(Threads.nthreads())

failures = 0
for name in names
    path = abspath(joinpath(NOTEBOOKS, name * ".jl"))
    t = time()
    println("running ", name, " …"); flush(stdout)
    nb = Pluto.SessionActions.open(session, path; run_async=false)
    bad = [c for c in nb.cells if c.errored]
    println(name, ": ", length(nb.cells), " cells, ", length(bad), " errored, ", round(Int, time() - t), " s")
    for c in bad
        println("  CELL: ", first(replace(c.code, "\n" => " "), 100))
        println("  ERROR: ", first(replace(string(get(c.output.body, :msg, c.output.body)), "\n" => " "), 500))
    end
    global failures += length(bad)
    Pluto.SessionActions.shutdown(session, nb; async=false)
end
println(failures == 0 ? "ALL CLEAN" : "FAILURES: $failures")
exit(failures == 0 ? 0 : 1)
