using Pkg; Pkg.activate(@__DIR__)

using PGLearn, PGLib, PowerModels, JuMP, DiffOpt, MadDiff, MadNLP
using MathOptInterface; const MOI = MathOptInterface
using Printf

PowerModels.silence()

n_runs = 10
warmup_case = "pglib_opf_case5_pjm"
case = "pglib_opf_case1354_pegase"
warmup_network = make_basic_network(pglib(warmup_case))
warmup_data = PGLearn.OPFData(warmup_network)
network = make_basic_network(pglib(case))
data = PGLearn.OPFData(network)
println("Case: $case ($(data.N) buses, $(data.G) gens, $(data.L) loads)")

function build_and_solve(data, optimizer)
    opf, pd_params, qd_params = PGLearn.build_opf(PGLearn.ACOPFParam, data, optimizer)
    model = opf.model
    set_silent(model)
    optimize!(model)
    return model
end

function run_reverse_sensitivity!(model, pg_vars)
    DiffOpt.empty_input_sensitivities!(model)
    # Set dL/dx = 1 for all pg variables (generator outputs)
    for pg in pg_vars
        MOI.set(model, DiffOpt.ReverseVariablePrimal(), pg, 1.0)
    end
    t = @elapsed DiffOpt.reverse_differentiate!(model)
    return t
end

for (name, optimizer) in [
    ("MadDiff (reuse ReducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer), MadDiff.MADDIFF_SKIP_KKT_REFACTORIZATION => true)),
    ("MadDiff (fresh UnreducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer), MadDiff.MADDIFF_KKTSYSTEM => MadNLP.SparseUnreducedKKTSystem)),
    ("DiffOpt", () -> DiffOpt.diff_optimizer(MadNLP.Optimizer)),
]
    @info "Warmup $name"
    model= build_and_solve(warmup_data, optimizer)
    run_reverse_sensitivity!(model, model[:pg])
    @info "Warmup $name done"
end


for (name, optimizer) in [
    ("MadDiff (reuse ReducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer), MadDiff.MADDIFF_SKIP_KKT_REFACTORIZATION => true)),
    ("MadDiff (fresh UnreducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer), MadDiff.MADDIFF_KKTSYSTEM => MadNLP.SparseUnreducedKKTSystem)),
    ("DiffOpt", () -> DiffOpt.diff_optimizer(MadNLP.Optimizer)),
]
    model = build_and_solve(data, optimizer)
    println("Starting benchmark $name (reverse mode)...")
    
    times = Float64[]
    for run in 1:n_runs
        t = run_reverse_sensitivity!(model, model[:pg])
        push!(times, t)
        @printf("  Run %d: %.3f ms\n", run, t * 1000)
    end
    avg_time = sum(times) / n_runs * 1000  # ms
    min_time = minimum(times) * 1000

    println("\n$name: Reverse mode timing results (1 VJP per run, $n_runs runs):")
    println("="^50)
    @printf("Avg:     %9.3f ms\n", avg_time)
    @printf("Min:     %9.3f ms\n", min_time)
    println("="^50)
end
