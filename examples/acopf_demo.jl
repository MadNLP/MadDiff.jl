using Pkg; Pkg.activate(@__DIR__)

using PGLearn, PGLib, PowerModels, JuMP, DiffOpt, MadDiff, MadNLP
using MathOptInterface; const MOI = MathOptInterface
using Printf

PowerModels.silence()

CONFIGS = [
    ("MadDiff (reuse ReducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer),
        MadDiff.MADDIFF_SKIP_KKT_REFACTORIZATION => true)),
    ("MadDiff (fresh UnreducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer),
        MadDiff.MADDIFF_KKTSYSTEM => MadNLP.SparseUnreducedKKTSystem)),
    ("DiffOpt", () -> DiffOpt.diff_optimizer(MadNLP.Optimizer)),
]

n_runs = 10
case = "pglib_opf_case1354_pegase"
network = make_basic_network(pglib(case))
data = PGLearn.OPFData(network)
println("Case: $case ($(data.N) buses, $(data.G) gens, $(data.L) loads)")

function build_and_solve(data, optimizer)
    model = PGLearn.build_opf(PGLearn.ACOPFParam, data, optimizer)[1].model
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

function run_benchmark(name, optimizer, data; warmup=false)
    model = build_and_solve(data, optimizer)

    times = Float64[]
    warmup || @info "$name"
    for run in 1:(warmup ? 1 : n_runs)
        t = run_reverse_sensitivity!(model, model[:pg])
        push!(times, t)
        warmup || @printf("  Run %d: %.3f ms\n", run, t * 1000)
    end

    warmup || @info "$name ($n_runs VJPs):" ms_avg=sum(times) / n_runs * 1000 ms_min=minimum(times) * 1000 ms_max=maximum(times) * 1000
end

@info "Starting warmup..."
warmup_case = "pglib_opf_case5_pjm"
warmup_network = make_basic_network(pglib(warmup_case))
warmup_data = PGLearn.OPFData(warmup_network)
for (name, optimizer) in CONFIGS
    run_benchmark(name, optimizer, warmup_data, warmup=true)
end

@info "Starting benchmark..."
for (name, optimizer) in CONFIGS
    run_benchmark(name, optimizer, data)
end