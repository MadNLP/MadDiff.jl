using Test, Random, LinearAlgebra
using MadDiff
using MadNLP, MadIPM, MadNCL, HybridKKT
using NLPModels, CUDA, MadNLPGPU, MadNLPTests, QuadraticModels
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
    # TODO: Argos.jl (MixedAuglagKKTSystem, BieglerKKTSystem)
    # TODO: CompressedSensingIPM.jl (FFTKKTSystem, CondensedFFTKKTSystem)
]
    # TODO: test LBFGS
    @testset "$(KKTSystem)" begin
        _run_adjoint_tests(KKTSystem, Callback)
    end
end
end

@testset "README" begin
@testset "NLPModels API" begin
    using NLPModelsJuMP

    model = Model()
    @variable(model, x)
    @constraint(model, x >= 2.0)  # the RHS is 2*p
    @objective(model, Min, x^2)

    nlp = MathOptNLPModel(model)
    solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    MadNLP.solve!(solver)

    sens = MadDiff.MadDiffSolver(solver, config=MadDiffConfig(skip_kkt_refactorization=true))

    dlcon_dp = [2.0;;]  # ∂lcon/∂p (n_con × n_p matrix)
    Δp = [1.0]  # parameter perturbation direction
    fwd = MadDiff.forward_differentiate!(sens; dlcon_dp=dlcon_dp * Δp)

    dL_dx = [1.0]  # gradient of loss w.r.t. x*
    vjp = MadDiff.make_param_pullback(dlcon_dp=dlcon_dp)
    sens_rev = MadDiff.MadDiffSolver(solver; param_pullback=vjp, n_p=1)
    rev = MadDiff.reverse_differentiate!(sens_rev; dL_dx)

    @test dot(dL_dx, fwd.dx) ≈ dot(rev.grad_p, Δp) atol=1e-4  # FIXME: precision issue when not refactorizing

    sens2 = MadDiff.MadDiffSolver(solver; config=MadDiffConfig(kkt_system=MadNLP.SparseUnreducedKKTSystem, reuse_kkt=false))
    dlcon_dp2 = [2.0;;]  # ∂lcon/∂p (n_con × n_p matrix)
    Δp2 = [1.0]  # parameter perturbation direction
    fwd2 = MadDiff.forward_differentiate!(sens; dlcon_dp=dlcon_dp2 * Δp2)

    dL_dx2 = [1.0]  # gradient of loss w.r.t. x*
    vjp2 = MadDiff.make_param_pullback(dlcon_dp=dlcon_dp2)
    sens_rev2 = MadDiff.MadDiffSolver(solver; param_pullback=vjp2, n_p=1)
    rev2 = MadDiff.reverse_differentiate!(sens_rev2; dL_dx=dL_dx2)

    @test dot(dL_dx2, fwd2.dx) ≈ dot(rev2.grad_p, Δp2)
    @test dot(dL_dx, fwd.dx) ≈ dot(dL_dx2, fwd2.dx)  atol=1e-4  # FIXME: precision issue when not refactorizing
end

@testset "DiffOpt API" begin
    model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
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
        model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
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

const DX_TOL = 1e-6
const DY_TOL = 1e-3  # TODO: investigate
# (name, opts, dx_tol, dy_tol, skip_equality)
const KKT_CONFIGS = [
    ("SparseKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.SparseKKTSystem), DX_TOL, DY_TOL, false),
    ("SparseCondensedKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.SparseCondensedKKTSystem, :bound_relax_factor => 1e-6), 5e-4, 5e-3, true),  # /!\ reduced tolerances for condensed
    ("SparseUnreducedKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.SparseUnreducedKKTSystem), DX_TOL, DY_TOL, false),
    ("ScaledSparseKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.ScaledSparseKKTSystem), DX_TOL, DY_TOL, false),
    ("DenseKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.DenseKKTSystem, :linear_solver => MadNLP.LapackCPUSolver), DX_TOL, DY_TOL, false),
    ("DenseCondensedKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.DenseCondensedKKTSystem, :linear_solver => MadNLP.LapackCPUSolver), DX_TOL, DY_TOL, false),
    ("NormalKKT", Dict{Symbol, Any}(:kkt_system => MadIPM.NormalKKTSystem, :linear_solver => MadNLP.LapackCPUSolver), DX_TOL, DY_TOL, false),
    # TODO: implement MadNCL.Optimizer so we can test K1s and K2r here
    ("HybridCondensedKKT", Dict{Symbol, Any}(:kkt_system => HybridKKT.HybridCondensedKKTSystem), 5e-4, 5e-3, false),  # /!\ reduced tolerances for condensed
]

if HAS_CUDA
    @info "Testing with CUDSS (reduced tolerances)"
    push!(KKT_CONFIGS, ("CUDA", Dict{Symbol, Any}(:linear_solver => CUDSSSolver, :tol => 1e-8), 1e-4, 1e-2, false))
end

const FV_CONFIGS = [
    ("Default", nothing),
    ("MakeParameter", MadNLP.MakeParameter),
    ("RelaxBound", MadNLP.RelaxBound),
]

const SKIP_PROBLEMS = Set([
    "qp_multi_con",  # dual degeneracy
    "qp_mixed",  # primal degeneracy
    "nlp_trig",  # TODO: investigate (condensed)
])

for (kkt_name, kkt_opts, dx_tol, dy_tol, skip_equality) in KKT_CONFIGS
for (fv_name, fv_handler) in FV_CONFIGS
@testset "$kkt_name × $fv_name" begin
Random.seed!(42)
for prob_name in keys(PROBLEMS)
    prob_name in SKIP_PROBLEMS && continue
    build, n_params, has_equality, is_lp = get_problem(prob_name)
    kkt_system = get(kkt_opts, :kkt_system, nothing)
    kkt_system === MadIPM.NormalKKTSystem && !is_lp && continue
    skip_equality && has_equality && continue

    # TODO: investigate
    prob_name == "small_coef" && kkt_system === MadNLP.SparseCondensedKKTSystem && continue

    # tolerance adjustments for misbehaving cases
    dx_atol = prob_name == "small_coef" ? 0.0 : dx_tol  # use rtol for small_coef
    dy_atol = prob_name == "small_coef" ? 0.0 : dy_tol
    rtol = prob_name == "small_coef" ? 1e-4 : 0.0

    @testset "$prob_name" begin
    opts = copy(kkt_opts)
    if fv_handler !== nothing
        opts[:fixed_variable_treatment] = fv_handler
    end

    n_x, n_con, n_lb, n_ub = get_problem_dims(build; opts...)

    # Forward mode tests
    @testset "Forward $prob_name" begin
        rng = MersenneTwister(hash((kkt_name, fv_name, prob_name)))
        for pidx in 1:n_params
        dp = randn(rng)
        (dx_mad, dy_mad, dzl_mad, dzu_mad) = run_maddiff(build; param_idx=pidx, dp=dp, opts...)
        (dx_diff, dy_diff, dzl_diff, dzu_diff) = run_diffopt(build; param_idx=pidx, dp=dp, opts...)

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
        end
    end

    # Reverse mode tests
    @testset "Reverse $prob_name" begin
        rng = MersenneTwister(hash((kkt_name, fv_name, prob_name)))
        dL_dx = randn(rng, n_x)
        dL_dy = randn(rng, n_con)
        dL_dzl = randn(rng, n_lb)
        dL_dzu = randn(rng, n_ub)

        grad_mad = run_maddiff_reverse(build; dL_dx, dL_dy, dL_dzl, dL_dzu, opts...)
        grad_diff = run_diffopt_reverse(build; dL_dx, dL_dy, dL_dzl, dL_dzu, opts...)
        for (g1, g2) in zip(grad_mad, grad_diff)
            @test isapprox(g1, g2; atol=dy_atol, rtol)
        end
    end
end
end
end
end
end

if HAS_CUDA
    include("cuda.jl")
end

#=
@testset "MOI" begin
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MadDiff.diff_optimizer(MadNLP.Optimizer)(),
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol=1e-4,
            rtol=1e-4,
            infeasible_status=MOI.LOCALLY_INFEASIBLE,
            optimal_status=MOI.LOCALLY_SOLVED,
            exclude=Any[
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
            ]
        );
        exclude = [
            # MadNLP reaches maximum number of iterations instead
            # of returning infeasibility certificate.
            r"test_linear_DUAL_INFEASIBLE.*",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            # Symbolic exception in Mumps
            "test_solve_VariableIndex_ConstraintDual_",
            # Tests excluded on purpose
            # - Excluded because Hessian information is needed
            "test_nonlinear_hs071_hessian_vector_product",
            # - Excluded because Hessian information is needed
            "test_nonlinear_invalid",
            #  - Excluded because this test is optional
            "test_model_ScalarFunctionConstantNotZero",
            # Throw an error: "Unable to query the dual of a variable
            # bound that was reformulated using `ZerosBridge`."
            "test_linear_VectorAffineFunction_empty_row",
            "test_conic_linear_VectorOfVariables_2",
            # TODO: investigate why it is breaking.
            "test_nonlinear_expression_hs109",
        ]
    )
end
=#
