using PGLearn, PGLib, PowerModels, JuMP, DiffOpt, MadDiff, MadNLP
using MathOptInterface; const MOI = MathOptInterface
using Printf, Random, Test

ENV["JULIA_TEST_FAILFAST"] = "true"

PowerModels.silence()

CONFIGS = [
    ("MadDiff (reuse ReducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer),
        MadDiff.MADDIFF_SKIP_KKT_REFACTORIZATION => true)),
    ("MadDiff (fresh UnreducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer),
        MadDiff.MADDIFF_KKTSYSTEM => MadNLP.SparseUnreducedKKTSystem)),
    ("DiffOpt", () -> DiffOpt.diff_optimizer(MadNLP.Optimizer)),
]

seed = 42
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

function run_reverse_sensitivity!(model, pg_vars; run=1)
    rng = Xoshiro(seed)
    DiffOpt.empty_input_sensitivities!(model)
    # Set dL/dx = 1 for all pg variables (generator outputs)
    for pg in pg_vars
        MOI.set(model, DiffOpt.ReverseVariablePrimal(), pg, randn(rng))
    end
    t = @elapsed DiffOpt.reverse_differentiate!(model)
    results = []
    for pd in model[:pd]
        push!(results, DiffOpt.get_reverse_parameter(model, pd))
    end
    return t, results
end

function run_benchmark(name, optimizer, data; warmup=false)
    model = build_and_solve(data, optimizer)

    times = Float64[]
    warmup || @info "$name"
    results = []
    for run in 1:(warmup ? 2 : n_runs)
        t, result = run_reverse_sensitivity!(model, model[:pg]; run)
        push!(times, t)
        warmup || @printf("  Run %d: %.3f ms\n", run, t * 1000)
        push!(results, result)
    end

    warmup || @info "$name:" ms_avg=sum(times) / n_runs * 1000 ms_min=minimum(times) * 1000 ms_max=maximum(times) * 1000

    return times, results
end

@info "Starting warmup..."
warmup_case = "pglib_opf_case5_pjm"
warmup_network = make_basic_network(pglib(warmup_case))
warmup_data = PGLearn.OPFData(warmup_network)
println("Case: $warmup_case ($(warmup_data.N) buses, $(warmup_data.G) gens, $(warmup_data.L) loads)")

results = []
for (name, optimizer) in CONFIGS
    times, result = run_benchmark(name, optimizer, warmup_data, warmup=true)
    push!(results, result)
end
@testset "Warmup" begin
    for i in axes(results, 1)  # for each config i
        (i > 1) && for j in axes(results[i], 1)  # for each run
            for k in axes(results[i][j], 1)  # for each parameter
                @test results[i-1][j][k] ≈ results[i][j][k]
            end
        end
    end
end

@info "Starting benchmark..."
results = []
for (name, optimizer) in CONFIGS
    times, result = run_benchmark(name, optimizer, data)
    push!(results, result)
end
@testset "Benchmark" begin
    for i in axes(results, 1)  # for each config i
        (i > 1) && for j in axes(results[i], 1)  # for each run
            for k in axes(results[i][j], 1)  # for each parameter
                @test results[i-1][j][k] ≈ results[i][j][k] atol=5e-4
            end
        end
    end
end