using Test
using MadDiff
using MadNLP
using JuMP
using DiffOpt
using LinearAlgebra
using MathOptInterface
using FiniteDiff
const MOI = MathOptInterface

@testset "README examples" begin
    @testset "NLPModels API" begin
        using NLPModelsJuMP

        model = Model()
        @variable(model, x)
        @constraint(model, x >= 2.0)  # the RHS is 2*p
        @objective(model, Min, x^2)

        nlp = MathOptNLPModel(model)
        solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR)
        MadNLP.solve!(solver)

        sens = MadDiff.MadDiffSolver(solver)

        Dp_lcon = [2.0;;]  # ∂lcon/∂p (n_con × n_params matrix)
        Δp = [1.0]  # parameter perturbation direction
        fwd = MadDiff.forward_differentiate!(sens; Dp_lcon=Dp_lcon * Δp)

        dL_dx = [1.0]  # gradient of loss w.r.t. x*
        vjp = MadDiff.make_param_pullback(Dp_lcon=Dp_lcon)
        sens_rev = MadDiff.MadDiffSolver(solver; param_pullback=vjp, n_params=1)
        rev = MadDiff.reverse_differentiate!(sens_rev; dL_dx)

        @test dot(dL_dx, fwd.dx) ≈ dot(rev.grad_p, Δp)

        sens2 = MadDiff.MadDiffSolver(solver; config=MadDiffConfig(kkt_system=MadNLP.SparseUnreducedKKTSystem, regularization=:inertia, reuse_kkt=false))
        Dp_lcon2 = [2.0;;]  # ∂lcon/∂p (n_con × n_params matrix)
        Δp2 = [1.0]  # parameter perturbation direction
        fwd2 = MadDiff.forward_differentiate!(sens; Dp_lcon=Dp_lcon2 * Δp2)

        dL_dx2 = [1.0]  # gradient of loss w.r.t. x*
        vjp2 = MadDiff.make_param_pullback(Dp_lcon=Dp_lcon2)
        sens_rev2 = MadDiff.MadDiffSolver(solver; param_pullback=vjp2, n_params=1)
        rev2 = MadDiff.reverse_differentiate!(sens_rev2; dL_dx=dL_dx2)

        @test dot(dL_dx2, fwd2.dx) ≈ dot(rev2.grad_p, Δp2)
        @test dot(dL_dx, fwd.dx) ≈ dot(dL_dx2, fwd2.dx)
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
    end
end

include("problems.jl")
include("helpers.jl")

const DX_TOL = 1e-6
const Dλ_TOL = 1e-3  # TODO: investigate
# (name, opts, dx_tol, dλ_tol, skip_equality)
const KKT_CONFIGS = [
    ("SparseKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.SparseKKTSystem), DX_TOL, Dλ_TOL, false),
    ("SparseCondensedKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.SparseCondensedKKTSystem, :bound_relax_factor => 1e-6), 5e-4, 5e-3, true),  # /!\ reduced tolerances for condensed
    ("SparseUnreducedKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.SparseUnreducedKKTSystem), DX_TOL, Dλ_TOL, false),
    ("ScaledSparseKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.ScaledSparseKKTSystem), DX_TOL, Dλ_TOL, false),
    ("DenseKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.DenseKKTSystem, :linear_solver => MadNLP.LapackCPUSolver), DX_TOL, Dλ_TOL, false),
    ("DenseCondensedKKT", Dict{Symbol, Any}(:kkt_system => MadNLP.DenseCondensedKKTSystem, :linear_solver => MadNLP.LapackCPUSolver), DX_TOL, Dλ_TOL, false),
]

const FV_CONFIGS = [
    ("Default", nothing),
    ("MakeParameter", MadNLP.MakeParameter),
    ("RelaxBound", MadNLP.RelaxBound),
]

const FORWARD_DP_VALUES = [1.0, 0.1, -1.7]

const SKIP_PROBLEMS = Set([
    "qp_multi_con",  # dual degeneracy
    "qp_mixed",  # primal degeneracy
    "nlp_trig",  # TODO: investigate (condensed)
])
const SKIP_FINITEDIFF = Set([  # TODO: investigate
    "diffopt_nlp_1",
    "bound_two_params",
    "diffopt_model_1",
    "diffopt_nlp_6",
    "diffopt_sipopt_multi",
    # "qp_max_basic",  # fails on github CI
    # "nlp_max_exp",   # fails on github CI
    # "nlp_max_quad",  # fails on github CI
])

@testset "Compare with DiffOpt.jl and FiniteDiff" begin
    for (kkt_name, kkt_opts, dx_tol, dλ_tol, skip_equality) in KKT_CONFIGS
        for (fv_name, fv_handler) in FV_CONFIGS
            @testset "$kkt_name × $fv_name" begin
                for prob_name in keys(PROBLEMS)
                    prob_name in SKIP_PROBLEMS && continue
                    build, n_params, has_equality = PROBLEMS[prob_name]
                    skip_equality && has_equality && continue

                    # tolerance adjustments for misbehaving cases
                    dx_atol = if prob_name == "small_coef"
                        0.0  # use rtol (sensitivity values are ~1000)
                    elseif prob_name == "qp_mixed"
                        1e-4
                    elseif prob_name in ("qp_max_basic", "nlp_max_exp", "nlp_max_quad")
                        dx_tol*10  # TODO: investigate why it fails on CI
                    else
                        dx_tol
                    end
                    dλ_atol = if prob_name == "small_coef"
                        0.0
                    elseif prob_name in ("qp_max_basic", "nlp_max_exp", "nlp_max_quad")
                        dλ_tol*10  # TODO: investigate why it fails on CI
                    else
                        dλ_tol
                    end
                    rtol = prob_name == "small_coef" ? 1e-4 : 0.0

                    @testset "$prob_name" begin
                        opts = copy(kkt_opts)
                        if fv_handler !== nothing
                            opts[:fixed_variable_treatment] = fv_handler
                        end

                        n_x, n_con, n_lb, n_ub = get_problem_dims(build)

                        # Forward mode tests
                        @testset "Forward $prob_name" begin
                            for pidx in 1:n_params
                                for dp in FORWARD_DP_VALUES
                                    (dx_mad, dλ_mad, dzl_mad, dzu_mad) = run_maddiff(build; param_idx=pidx, dp=dp, opts...)
                                    (dx_diff, dλ_diff, dzl_diff, dzu_diff) = run_diffopt(build; param_idx=pidx, dp=dp)

                                    @test all(isapprox.(dx_mad, dx_diff; atol=dx_atol, rtol))
                                    @test all(isapprox.(dλ_mad, dλ_diff; atol=dλ_atol, rtol))
                                    @test all(isapprox.(dzl_mad, dzl_diff; atol=dλ_atol, rtol))
                                    @test all(isapprox.(dzu_mad, dzu_diff; atol=dλ_atol, rtol))

                                    if !(prob_name in SKIP_FINITEDIFF)
                                        (dx_fd, dλ_fd, dzl_fd, dzu_fd) = run_finitediff_forward(build; param_idx=pidx, dp=dp)
                                        @test all(isapprox.(dx_mad, dx_fd; atol=dx_atol, rtol=rtol))
                                        @test all(isapprox.(dλ_mad, dλ_fd; atol=dλ_atol, rtol=rtol))
                                        @test all(isapprox.(dzl_mad, dzl_fd; atol=dλ_atol, rtol=rtol))
                                        @test all(isapprox.(dzu_mad, dzu_fd; atol=dλ_atol, rtol=rtol))
                                    end
                                end
                            end
                        end

                        # Reverse mode tests
                        @testset "Reverse $prob_name" begin
                            grad_mad = run_maddiff_reverse(build; opts...)
                            grad_diff = run_diffopt_reverse(build)
                            @test all(isapprox.(grad_mad, grad_diff; atol=dλ_atol, rtol))

                            if !(prob_name in SKIP_FINITEDIFF)
                                dL_dx_ones = ones(n_x)
                                dL_dλ_ones = ones(n_con)
                                dL_dzl_ones = ones(n_lb)
                                dL_dzu_ones = ones(n_ub)
                                grad_fd = run_finitediff_reverse(build, dL_dx_ones, dL_dλ_ones, dL_dzl_ones, dL_dzu_ones)
                                grad_mad_ones = run_maddiff_reverse(build; dL_dx=dL_dx_ones, dL_dλ=dL_dλ_ones, dL_dzl=dL_dzl_ones, dL_dzu=dL_dzu_ones, opts...)
                                @test all(isapprox.(grad_mad_ones, grad_fd; atol=dλ_atol, rtol=rtol))
                            end

                            dL_dx = [(-1.0)^i * (0.5 + i * 0.1) for i in 1:n_x]
                            dL_dλ = [(-1.0)^i * (0.3 + i * 0.2) for i in 1:n_con]
                            dL_dzl = [(-1.0)^i * (0.7 + i * 0.1) for i in 1:n_lb]
                            dL_dzu = [(-1.0)^i * (0.4 + i * 0.15) for i in 1:n_ub]

                            grad_mad_nu = run_maddiff_reverse(build; dL_dx, dL_dλ, dL_dzl, dL_dzu, opts...)
                            grad_diff_nu = run_diffopt_reverse(build; dL_dx, dL_dλ, dL_dzl, dL_dzu)
                            @test all(isapprox.(grad_mad_nu, grad_diff_nu; atol=dλ_atol, rtol=rtol))

                            if !(prob_name in SKIP_FINITEDIFF)
                                grad_fd_nu = run_finitediff_reverse(build, dL_dx, dL_dλ, dL_dzl, dL_dzu)
                                @test all(isapprox.(grad_mad_nu, grad_fd_nu; atol=dλ_atol, rtol=rtol))
                            end
                        end
                    end
                end
            end
        end
    end
end


# TODO:
# reset_sensitivity_cache!,
# make_param_pullback,_fwd_dual_rhs!,_fwd_dual_bound_con! cases (provide not all Dp, no dL_dx, etc.)
# _process_reverse_dual_input! for interval, equalto
# (one-shot) nlpmodels API
# DiffOpt.DifferentiateTimeSec
# reverse without param_pullback
# CUDA


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
