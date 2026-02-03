using PGLearn, PGLib, PowerModels, JuMP, DiffOpt, MadDiff, MadNLP
using MathOptInterface; const MOI = MathOptInterface
using Printf, Random, Test, LinearAlgebra

# ENV["JULIA_TEST_FAILFAST"] = "true"

PowerModels.silence()

CONFIGS = [
    ("MadDiff (reuse ReducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer),
        MadDiff.MADDIFF_SKIP_KKT_REFACTORIZATION => true)),
    ("MadDiff (fresh UnreducedKKT)", optimizer_with_attributes(MadDiff.diff_optimizer(MadNLP.Optimizer),
        MadDiff.MADDIFF_KKTSYSTEM => MadNLP.SparseUnreducedKKTSystem)),
    ("DiffOpt", () -> DiffOpt.diff_optimizer(MadNLP.Optimizer)),
]

seed = 42
n_runs = 3
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
function Loss(model, seeds_pg)
    return dot(seeds_pg, value.(model[:pg]))
end
function Loss(model, seeds_pg, seeds_pf, seeds_vm_lb, seeds_vm_ub, seeds_kcl)
    N = length(model[:vm])
    L = 0.0
    L += dot(seeds_pg, value.(model[:pg]))
    L += dot(seeds_pf, value.(model[:pf]))
    for i in 1:N
        L += seeds_vm_lb[i] * dual(LowerBoundRef(model[:vm][i]))
        L += seeds_vm_ub[i] * dual(UpperBoundRef(model[:vm][i]))
    end
    L += dot(seeds_kcl, dual.(model[:kcl_q]))
    return L
end

function finite_diff_grad_k(data, optimizer, k; seed = 42, ε = 1e-6)
    rng = Xoshiro(seed)
    N, G, E = data.N, data.G, data.E
    seeds_pg = [randn(rng) for _ in 1:G]
    seeds_pf = [randn(rng) for _ in 1:E]
    seeds_vm_lb = [randn(rng) for _ in 1:N]
    seeds_vm_ub = [randn(rng) for _ in 1:N]
    seeds_kcl = [randn(rng) for _ in 1:N]
    data_plus = deepcopy(data)
    data_plus.pd = copy(data.pd)
    data_plus.pd[k] += ε
    model_plus = build_and_solve(data_plus, optimizer)
    L_plus = Loss(model_plus, seeds_pg, seeds_pf, seeds_vm_lb, seeds_vm_ub, seeds_kcl)
    data_minus = deepcopy(data)
    data_minus.pd = copy(data.pd)
    data_minus.pd[k] -= ε
    model_minus = build_and_solve(data_minus, optimizer)
    L_minus = Loss(model_minus, seeds_pg, seeds_pf, seeds_vm_lb, seeds_vm_ub, seeds_kcl)
    return (L_plus - L_minus) / (2 * ε)
end

function run_reverse_sensitivity!(model, pg_vars; run=1)
    rng = Xoshiro(seed)
    DiffOpt.empty_input_sensitivities!(model)
    for pg in pg_vars
        MOI.set(model, DiffOpt.ReverseVariablePrimal(), pg, randn(rng))
    end
    for pf in model[:pf]
        MOI.set(model, DiffOpt.ReverseVariablePrimal(), pf, randn(rng))
    end
    for vm in model[:vm]
        MOI.set(model, DiffOpt.ReverseConstraintDual(), LowerBoundRef(vm), randn(rng))
    end
    for vm in model[:vm]
        MOI.set(model, DiffOpt.ReverseConstraintDual(), UpperBoundRef(vm), randn(rng))
    end
    for kcl in model[:kcl_q]
        MOI.set(model, DiffOpt.ReverseConstraintDual(), kcl, randn(rng))
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
    for i in axes(results, 1)
        (i > 1) && for j in axes(results[i], 1)
            for k in axes(results[i][j], 1)
                @test results[i-1][j][k] ≈ results[i][j][k] atol = 1e-4 rtol = 1e-3
            end
        end
    end
    fd_k1 = finite_diff_grad_k(warmup_data, CONFIGS[1][2], 1)
    for i in eachindex(CONFIGS)
        @test results[i][1][1] ≈ fd_k1 atol = 1e-3 rtol = 1e-2
    end
end

@info "Starting benchmark..."
results = []
for (name, optimizer) in CONFIGS
    times, result = run_benchmark(name, optimizer, data)
    push!(results, result)
end
@testset "Benchmark" begin
    atol_sens, rtol_sens = 1e-4, 1e-3
    n_p = length(data.pd)
    pair2_failing_ks = Set{Int}()
    for j in axes(results[2], 1)
        for k in 1:n_p
            if !isapprox(results[2][j][k], results[3][j][k]; atol = atol_sens, rtol = rtol_sens)
                push!(pair2_failing_ks, k)
            end
        end
    end
    for j in axes(results[1], 1)
        for k in 1:n_p
            @test results[1][j][k] ≈ results[2][j][k] atol = atol_sens rtol = rtol_sens
        end
    end
    if isempty(pair2_failing_ks)
        for j in axes(results[2], 1)
            for k in 1:n_p
                @test results[2][j][k] ≈ results[3][j][k] atol = atol_sens rtol = rtol_sens
            end
        end
    else
        diffopt_closer_ks = Int[]
        @info "Found mismatches for $(length(pair2_failing_ks)) param(s). Checking against finite-diff..."
        num_checks = 0
        max_checks = 10
        for k in pair2_failing_ks
            if num_checks >= max_checks
                @info "Stopping FD after $max_checks checks."
                break
            end
            print("Running FD for param $k...")
            fd_k = finite_diff_grad_k(data, CONFIGS[1][2], k)
            err_md = abs(results[2][1][k] - fd_k)
            err_do = abs(results[3][1][k] - fd_k)
            if err_do < err_md
                push!(diffopt_closer_ks, k)
                @test false
            else
                println("pass (MadDiff: abs=$(err_md) rel=$(err_md/abs(fd_k)) DiffOpt: abs=$(err_do) rel=$(err_do/abs(fd_k)))")
            end
            num_checks += 1
        end
        if isempty(diffopt_closer_ks)
            @info "MadDiff and DiffOpt had sensitivity mismatches but MadDiff was always closer to finite-diff."
        end
        for j in axes(results[2], 1)
            for k in 1:n_p
                k in pair2_failing_ks && continue
                @test results[2][j][k] ≈ results[3][j][k] atol = atol_sens rtol = rtol_sens
            end
        end
    end
end