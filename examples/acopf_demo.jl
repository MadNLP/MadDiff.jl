using Pkg; Pkg.activate(@__DIR__)

using PGLearn, PGLib, PowerModels, JuMP, DiffOpt, MadDiff, MadNLP
using MathOptInterface; const MOI = MathOptInterface
using Printf

PowerModels.silence()

case = "pglib_opf_case1354_pegase"
network = make_basic_network(pglib(case))
data = PGLearn.OPFData(network)
println("Case: $case ($(data.N) buses, $(data.G) gens, $(data.L) loads)")

function build_and_solve(data, make_optimizer)
    opf, pd_params, qd_params = PGLearn.build_opf(PGLearn.ACOPFParam, data, make_optimizer)
    model = opf.model
    set_silent(model)
    optimize!(model)
    return model, pd_params, qd_params, opf
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

function benchmark_sensitivities(data; n_runs = 10)
    make_optimizer = MadDiff.diff_optimizer(MadNLP.Optimizer)
    model, pd_params, qd_params, opf = build_and_solve(data, make_optimizer)

    pg_vars = model[:pg]

    # warmup
    run_reverse_sensitivity!(model, pg_vars)

    times = Float64[]

    println("Starting benchmark (reverse mode, 1 VJP per run)...")
    for run in 1:n_runs
        t = run_reverse_sensitivity!(model, pg_vars)
        push!(times, t)
        @printf("  Run %d: %.3f ms\n", run, t * 1000)
    end

    avg_time = sum(times) / n_runs * 1000  # ms
    min_time = minimum(times) * 1000

    println("\nTiming Results (reverse mode, $n_runs runs):")
    println("="^50)
    @printf("Avg:     %9.3f ms\n", avg_time)
    @printf("Min:     %9.3f ms\n", min_time)

    return (avg = avg_time, min = min_time)
end

result = benchmark_sensitivities(data; n_runs = 10)
