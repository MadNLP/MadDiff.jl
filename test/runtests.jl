using Test, Random, LinearAlgebra
using MadDiff
using MadNLP, MadIPM, MadNCL, HybridKKT
using NLPModels, CUDA, MadNLPGPU, MadNLPTests, QuadraticModels, ExaModels
using JuMP, DiffOpt, MathOptInterface
const MOI = MathOptInterface

const HAS_CUDA = CUDA.functional()

include("helpers.jl")

@testset "adjoint_{solve,mul}!" begin
for (KKTSystem, Callback) in [
    (MadNLP.SparseKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseUnreducedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.ScaledSparseKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseCondensedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.DenseKKTSystem, MadNLP.DenseCallback),
    (MadNLP.DenseCondensedKKTSystem, MadNLP.DenseCallback),
    (MadIPM.NormalKKTSystem, MadNLP.SparseCallback),
    (MadNCL.K1sAuglagKKTSystem, MadNLP.SparseCallback),
    (MadNCL.K2rAuglagKKTSystem, MadNLP.SparseCallback),
    (HybridKKT.HybridCondensedKKTSystem, MadNLP.SparseCallback),
    # TODO: BFGS support
    # TODO: Argos.jl (MixedAuglagKKTSystem, BieglerKKTSystem)
    # TODO: CompressedSensingIPM.jl (FFTKKTSystem, GondzioKKTSystem)
]
    @testset "$(KKTSystem)" begin
        _run_adjoint_tests(KKTSystem, Callback)
    end
end
end

@testset "README" begin
@testset "DiffOpt API" begin
    model = MadDiff.diff_model(MadNLP.Optimizer)
    set_silent(model)
    @variable(model, x)
    @variable(model, p in MOI.Parameter(1.0))
    @constraint(model, x >= 2p)  # can use explicit parameters
    @objective(model, Min, x^2)
    optimize!(model)

    DiffOpt.empty_input_sensitivities!(model)
    MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), MOI.Parameter(1.0))
    DiffOpt.forward_differentiate!(model)
    dx_mad = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x)

    DiffOpt.empty_input_sensitivities!(model)
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)
    DiffOpt.reverse_differentiate!(model)
    dp_mad = MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)).value

    model_ref = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
    MOI.set(model_ref, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
    set_silent(model_ref)
    @variable(model_ref, x_ref)
    @variable(model_ref, p_ref in MOI.Parameter(1.0))
    @constraint(model_ref, x_ref >= 2p_ref)
    @objective(model_ref, Min, x_ref^2)
    optimize!(model_ref)

    DiffOpt.empty_input_sensitivities!(model_ref)
    MOI.set(model_ref, DiffOpt.ForwardConstraintSet(), ParameterRef(p_ref), MOI.Parameter(1.0))
    DiffOpt.forward_differentiate!(model_ref)
    dx_ref = MOI.get(model_ref, DiffOpt.ForwardVariablePrimal(), x_ref)

    DiffOpt.empty_input_sensitivities!(model_ref)
    MOI.set(model_ref, DiffOpt.ReverseVariablePrimal(), x_ref, 1.0)
    DiffOpt.reverse_differentiate!(model_ref)
    dp_ref = MOI.get(model_ref, DiffOpt.ReverseConstraintSet(), ParameterRef(p_ref)).value

    @test isapprox(dx_mad, dx_ref; atol=1e-6)
    @test isapprox(dp_mad, dp_ref; atol=1e-6)

    @testset "empty_input_sensitivities!" begin
        model = MadDiff.diff_model(MadNLP.Optimizer)
        set_silent(model)
        @variable(model, x)
        @variable(model, p1 in MOI.Parameter(1.0))
        @variable(model, p2 in MOI.Parameter(1.0))
        @constraint(model, x >= p1 + 2p2)
        @objective(model, Min, x^2)
        optimize!(model)

        model_ref = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
        MOI.set(model_ref, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
        set_silent(model_ref)
        @variable(model_ref, x_ref)
        @variable(model_ref, p1_ref in MOI.Parameter(1.0))
        @variable(model_ref, p2_ref in MOI.Parameter(1.0))
        @constraint(model_ref, x_ref >= p1_ref + 2p2_ref)
        @objective(model_ref, Min, x_ref^2)
        optimize!(model_ref)

        function _forward_dx(model, x, param, dp)
            DiffOpt.empty_input_sensitivities!(model)
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(param), MOI.Parameter(dp))
            DiffOpt.forward_differentiate!(model)
            return MOI.get(model, DiffOpt.ForwardVariablePrimal(), x)
        end

        dx1_mad = _forward_dx(model, x, p1, 1.0)
        dx2_mad = _forward_dx(model, x, p1, 2.0)

        dx1_ref = _forward_dx(model_ref, x_ref, p1_ref, 1.0)
        dx2_ref = _forward_dx(model_ref, x_ref, p1_ref, 2.0)

        @test isapprox(dx1_mad, dx1_ref; atol=1e-6)
        @test isapprox(dx2_mad, dx2_ref; atol=1e-6)
    end
end
end

include("problems.jl")

const SKIP_PROBLEMS = Set([
    "qp_multi_con",  # dual degeneracy
    "qp_mixed",  # primal degeneracy
    "nlp_trig",  # TODO: investigate (condensed)
])

include("test_jacobian.jl")

const DX_TOL = 1e-6
const DY_TOL = 1e-3  # TODO: investigate
_kkt_config(; name, madnlp_opts = (;), maddiff_opts = (;), dx_tol = DX_TOL, dy_tol = DY_TOL, skip_equality = false) = (name = name, madnlp_opts = madnlp_opts, maddiff_opts = maddiff_opts, dx_tol = dx_tol, dy_tol = dy_tol, skip_equality = skip_equality)
const KKT_CONFIGS = [
    _kkt_config(name = "Default (SparseKKT)"),
    _kkt_config(
        name = "Default Skip (SparseKKT)",
        maddiff_opts = (; skip_kkt_refactorization = true),
        dx_tol = 5e-4,
        dy_tol = 5e-3,
    ),
    _kkt_config(
        name = "SparseCondensedKKT",
        madnlp_opts = (; kkt_system = MadNLP.SparseCondensedKKTSystem, bound_relax_factor = 1e-6),
        dx_tol = 5e-4,
        dy_tol = 5e-3,
        skip_equality = true,
    ),  # /!\ reduced tolerances for condensed
    _kkt_config(
        name = "SparseUnreducedKKT",
        madnlp_opts = (; kkt_system = MadNLP.SparseUnreducedKKTSystem),
    ),
    _kkt_config(
        name = "ScaledSparseKKT",
        madnlp_opts = (; kkt_system = MadNLP.ScaledSparseKKTSystem),
    ),
    _kkt_config(
        name = "DenseKKT",
        madnlp_opts = (; kkt_system = MadNLP.DenseKKTSystem, linear_solver = MadNLP.LapackCPUSolver),
    ),
    _kkt_config(
        name = "DenseCondensedKKT",
        madnlp_opts = (; kkt_system = MadNLP.DenseCondensedKKTSystem, linear_solver = MadNLP.LapackCPUSolver),
    ),
    _kkt_config(
        name = "NormalKKT",
        madnlp_opts = (; kkt_system = MadIPM.NormalKKTSystem, linear_solver = MadNLP.LapackCPUSolver),
    ),
    _kkt_config(
        name = "HybridCondensedKKT",
        madnlp_opts = (; kkt_system = HybridKKT.HybridCondensedKKTSystem),
        dx_tol = 5e-4,
        dy_tol = 5e-3,
    ),  # /!\ reduced tolerances for condensed
    # TODO: test MadIPM.Optimizer, MadNCL.Optimizer
]

if HAS_CUDA
    push!(
        KKT_CONFIGS,
        _kkt_config(
            name = "cuDSS (SparseCondensedKKT)",
            madnlp_opts = (
                linear_solver = CUDSSSolver,
                cudss_ir = 3,
                bound_relax_factor = 1e-7,
                tol = 1e-7,
            ),
            dx_tol = 1e-4,
            dy_tol = 1e-2,
        ),
    )
end

const FV_CONFIGS = [
    ("Default", nothing),
    ("MakeParameter", MadNLP.MakeParameter),
    ("RelaxBound", MadNLP.RelaxBound),
]

for cfg in KKT_CONFIGS
kkt_name = cfg.name
madnlp_opts_base = cfg.madnlp_opts
maddiff_opts = cfg.maddiff_opts
dx_tol = cfg.dx_tol
dy_tol = cfg.dy_tol
skip_equality = cfg.skip_equality
for (fv_name, fv_handler) in FV_CONFIGS
@testset "$kkt_name × $fv_name" begin
Random.seed!(42)
for prob_name in keys(PROBLEMS)
    prob_name in SKIP_PROBLEMS && continue
    build, n_params, has_equality, is_lp = get_problem(prob_name)
    kkt_system = get(madnlp_opts_base, :kkt_system, nothing)
    kkt_system === MadIPM.NormalKKTSystem && !is_lp && continue
    skip_equality && has_equality && continue

    # TODO: investigate
    prob_name == "small_coef" && kkt_system === MadNLP.SparseCondensedKKTSystem && continue

    # tolerance adjustments for misbehaving cases
    dx_atol = prob_name == "small_coef" ? 0.0 : dx_tol  # use rtol for small_coef
    dy_atol = prob_name == "small_coef" ? 0.0 : dy_tol
    rtol = prob_name == "small_coef" ? 1e-4 : 0.0

    @testset "$prob_name" begin
    madnlp_opts = fv_handler === nothing ?
        madnlp_opts_base :
        merge(madnlp_opts_base, (; fixed_variable_treatment = fv_handler))

    n_x, n_con, n_lb, n_ub = get_problem_dims(build; madnlp_opts)

    # Forward mode tests
    @testset "Forward $prob_name" begin
        rng = MersenneTwister(hash((kkt_name, fv_name, prob_name)))
        for pidx in 1:n_params
        dp = randn(rng)
        (dx_mad, dy_mad, dzl_mad, dzu_mad, dobj_mad) = run_maddiff(build; param_idx = pidx, dp = dp, madnlp_opts, maddiff_opts)
        (dx_diff, dy_diff, dzl_diff, dzu_diff, dobj_diff) = run_diffopt(build; param_idx = pidx, dp = dp, madnlp_opts)

        for (g1, g2) in zip(dx_mad, dx_diff)
            @test isapprox(g1, g2; atol=dx_atol, rtol)
        end
        for (g1, g2) in zip(dy_mad, dy_diff)
            @test isapprox(g1, g2; atol=dy_atol, rtol)
        end
        for (g1, g2) in zip(dzl_mad, dzl_diff)
            @test isapprox(g1, g2; atol=dy_atol, rtol)
        end
        for (g1, g2) in zip(dzu_mad, dzu_diff)
            @test isapprox(g1, g2; atol=dy_atol, rtol)
        end
        @test isapprox(dobj_mad, dobj_diff; atol=dy_atol, rtol)
        end
    end

    # Reverse mode tests
    @testset "Reverse $prob_name" begin
        rng = MersenneTwister(hash((kkt_name, fv_name, prob_name)))
        dL_dx = randn(rng, n_x)
        dL_dy = randn(rng, n_con)
        dL_dzl = randn(rng, n_lb)
        dL_dzu = randn(rng, n_ub)
        dobj = randn(rng)

        grad_mad = run_maddiff_reverse(build; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj, madnlp_opts, maddiff_opts)
        grad_diff = run_diffopt_reverse(build; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj, madnlp_opts)
        for (g1, g2) in zip(grad_mad, grad_diff)
            @test isapprox(g1, g2; atol=dy_atol, rtol)
        end
    end

    # Consistency tests: jac/jact transposes each other, jvp/vjp are adjoint
    @testset "Consistency $prob_name" begin
        run_maddiff_consistency(build; madnlp_opts, maddiff_opts, atol = dy_atol, rtol)
    end
end
end
end
end
end

@testset "MOI attributes" begin
    model = MadDiff.diff_model(MadNLP.Optimizer)
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(1.0))
    @constraint(model, c, x >= p)
    @objective(model, Min, x^2)
    optimize!(model)
    DiffOpt.empty_input_sensitivities!(model)
    MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), MOI.Parameter(1.0))
    DiffOpt.forward_differentiate!(model)
    @test solver_name(model) == "MadNLP"
    diff_time = MOI.get(model, DiffOpt.DifferentiateTimeSec())
    @test isfinite(diff_time)
    @test diff_time >= 0.0
end
