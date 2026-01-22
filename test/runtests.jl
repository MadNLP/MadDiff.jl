using Test
using MadDiff
using MadNLP
using JuMP
using DiffOpt
using LinearAlgebra
using MathOptInterface
using FiniteDiff
using CUDA
using MadNLPGPU
const MOI = MathOptInterface

ENV["JULIA_TEST_FAILFAST"] = "true"

const CUDA_FUNCTIONAL = CUDA.functional()

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

        dlcon_dp = [2.0;;]  # ∂lcon/∂p (n_con × n_p matrix)
        Δp = [1.0]  # parameter perturbation direction
        fwd = MadDiff.forward_differentiate!(sens; dlcon_dp=dlcon_dp * Δp)

        dL_dx = [1.0]  # gradient of loss w.r.t. x*
        vjp = MadDiff.make_param_pullback(dlcon_dp=dlcon_dp)
        sens_rev = MadDiff.MadDiffSolver(solver; param_pullback=vjp, n_p=1)
        rev = MadDiff.reverse_differentiate!(sens_rev; dL_dx)

        @test dot(dL_dx, fwd.dx) ≈ dot(rev.grad_p, Δp)

        sens2 = MadDiff.MadDiffSolver(solver; config=MadDiffConfig(kkt_system=MadNLP.SparseUnreducedKKTSystem, regularization=:inertia, reuse_kkt=false))
        dlcon_dp2 = [2.0;;]  # ∂lcon/∂p (n_con × n_p matrix)
        Δp2 = [1.0]  # parameter perturbation direction
        fwd2 = MadDiff.forward_differentiate!(sens; dlcon_dp=dlcon_dp2 * Δp2)

        dL_dx2 = [1.0]  # gradient of loss w.r.t. x*
        vjp2 = MadDiff.make_param_pullback(dlcon_dp=dlcon_dp2)
        sens_rev2 = MadDiff.MadDiffSolver(solver; param_pullback=vjp2, n_p=1)
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

if CUDA_FUNCTIONAL
    @info "Testing with CUDSS (reduced tolerances)"
    push!(KKT_CONFIGS, ("CUDA", Dict{Symbol, Any}(:linear_solver => CUDSSSolver, :tol => 1e-8), 1e-4, 1e-2, false))
end

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
                    dx_atol = prob_name == "small_coef" ? 0.0 : dx_tol  # use rtol for small_coef
                    dλ_atol = prob_name == "small_coef" ? 0.0 : dλ_tol
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


@testset "MaxSense" begin
    @testset "NLP Equality" begin
        build = function (m)
            @variable(m, x >= 0.1, start = 0.5)
            @variable(m, p in MOI.Parameter(2.0))
            @constraint(m, p * x == 1.0)
            @objective(m, Max, -(x - p)^2)
            return [x], p
        end

        (dx_mad, dλ_mad, dzl_mad, dzu_mad) = run_maddiff(build; dp = 1.0)
        (dx_diff, dλ_diff, dzl_diff, dzu_diff) = run_diffopt(build; dp = 1.0)

        @test all(isapprox.(dx_mad, dx_diff; atol = DX_TOL))
        @test all(isapprox.(dλ_mad, dλ_diff; atol = Dλ_TOL))
        @test all(isapprox.(dzl_mad, dzl_diff; atol = Dλ_TOL))
        @test all(isapprox.(dzu_mad, dzu_diff; atol = Dλ_TOL))

        grad_mad = run_maddiff_reverse(build)
        grad_diff = run_diffopt_reverse(build)
        @test all(isapprox.(grad_mad, grad_diff; atol = Dλ_TOL))
    end

    @testset "Active Bound" begin
        build = function (m)
            @variable(m, x >= 0.0, start = 0.1)
            @variable(m, p in MOI.Parameter(-1.0))
            @objective(m, Max, -(x - p)^2)
            return [x], p
        end

        (dx_mad, dλ_mad, dzl_mad, dzu_mad) = run_maddiff(build; dp = 1.0)
        (dx_diff, dλ_diff, dzl_diff, dzu_diff) = run_diffopt(build; dp = 1.0)

        @test all(isapprox.(dx_mad, dx_diff; atol = DX_TOL))
        @test all(isapprox.(dλ_mad, dλ_diff; atol = Dλ_TOL))
        @test all(isapprox.(dzl_mad, dzl_diff; atol = Dλ_TOL))
        @test all(isapprox.(dzu_mad, dzu_diff; atol = Dλ_TOL))

        grad_mad = run_maddiff_reverse(build)
        grad_diff = run_diffopt_reverse(build)
        @test all(isapprox.(grad_mad, grad_diff; atol = Dλ_TOL))
    end
end


# TODO:
# reset_sensitivity_cache!,
# make_param_pullback,_fwd_dual_rhs!,_fwd_dual_bound_con! cases (provide not all Dp, no dL_dx, etc.)
# _process_reverse_dual_input! for interval, equalto
# (one-shot) nlpmodels API
# DiffOpt.DifferentiateTimeSec
# reverse without param_pullback

if CUDA_FUNCTIONAL
    include("cuda.jl")
end

@testset "Fixed variable with parameter-dependent bounds" begin
    @testset "Forward mode - compare with DiffOpt" begin
        # Problem: min x^2 + y^2 s.t. x + y >= 3, y == p (equality constraint fixes y to p)
        # When p = 1.5: y* = 1.5, x* = 1.5 (inequality constraint active)
        model_ref = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
        MOI.set(model_ref, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
        set_silent(model_ref)
        @variable(model_ref, x_ref >= 0, start=1.5)
        @variable(model_ref, y_ref, start=1.5)
        @variable(model_ref, p_ref in MOI.Parameter(1.5))
        @constraint(model_ref, y_ref == p_ref)
        @constraint(model_ref, x_ref + y_ref >= 3)
        @objective(model_ref, Min, x_ref^2 + y_ref^2)
        optimize!(model_ref)

        DiffOpt.empty_input_sensitivities!(model_ref)
        MOI.set(model_ref, DiffOpt.ForwardConstraintSet(), ParameterRef(p_ref), MOI.Parameter(1.0))
        DiffOpt.forward_differentiate!(model_ref)
        dx_ref = MOI.get(model_ref, DiffOpt.ForwardVariablePrimal(), x_ref)
        dy_ref = MOI.get(model_ref, DiffOpt.ForwardVariablePrimal(), y_ref)

        model_mad = Model(MadDiff.diff_optimizer(MadNLP.Optimizer; fixed_variable_treatment=MadNLP.MakeParameter))
        set_silent(model_mad)
        @variable(model_mad, x_mad >= 0, start=1.5)
        @variable(model_mad, y_mad, start=1.5)
        @variable(model_mad, p_mad in MOI.Parameter(1.5))
        @constraint(model_mad, y_mad == p_mad)
        @constraint(model_mad, x_mad + y_mad >= 3)
        @objective(model_mad, Min, x_mad^2 + y_mad^2)
        optimize!(model_mad)

        DiffOpt.empty_input_sensitivities!(model_mad)
        MOI.set(model_mad, DiffOpt.ForwardConstraintSet(), ParameterRef(p_mad), MOI.Parameter(1.0))
        DiffOpt.forward_differentiate!(model_mad)
        dx_mad = MOI.get(model_mad, DiffOpt.ForwardVariablePrimal(), x_mad)
        dy_mad = MOI.get(model_mad, DiffOpt.ForwardVariablePrimal(), y_mad)

        @test isapprox(dx_mad, dx_ref; atol=1e-6)
        @test isapprox(dy_mad, dy_ref; atol=1e-6)
        @test isapprox(dy_mad, 1.0; atol=1e-6)  # dy/dp = 1
        @test isapprox(dx_mad, -1.0; atol=1e-6)  # dx/dp = -1 (from constraint)
    end

    @testset "Reverse mode - compare with DiffOpt" begin
        model_ref = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
        MOI.set(model_ref, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
        set_silent(model_ref)
        @variable(model_ref, x_ref >= 0, start=1.5)
        @variable(model_ref, y_ref, start=1.5)
        @variable(model_ref, p_ref in MOI.Parameter(1.5))
        @constraint(model_ref, y_ref == p_ref)
        @constraint(model_ref, x_ref + y_ref >= 3)
        @objective(model_ref, Min, x_ref^2 + y_ref^2)
        optimize!(model_ref)

        DiffOpt.empty_input_sensitivities!(model_ref)
        MOI.set(model_ref, DiffOpt.ReverseVariablePrimal(), x_ref, 1.0)
        MOI.set(model_ref, DiffOpt.ReverseVariablePrimal(), y_ref, 1.0)
        DiffOpt.reverse_differentiate!(model_ref)
        dp_ref = MOI.get(model_ref, DiffOpt.ReverseConstraintSet(), ParameterRef(p_ref)).value

        model_mad = Model(MadDiff.diff_optimizer(MadNLP.Optimizer; fixed_variable_treatment=MadNLP.MakeParameter))
        set_silent(model_mad)
        @variable(model_mad, x_mad >= 0, start=1.5)
        @variable(model_mad, y_mad, start=1.5)
        @variable(model_mad, p_mad in MOI.Parameter(1.5))
        @constraint(model_mad, y_mad == p_mad)
        @constraint(model_mad, x_mad + y_mad >= 3)
        @objective(model_mad, Min, x_mad^2 + y_mad^2)
        optimize!(model_mad)

        DiffOpt.empty_input_sensitivities!(model_mad)
        MOI.set(model_mad, DiffOpt.ReverseVariablePrimal(), x_mad, 1.0)
        MOI.set(model_mad, DiffOpt.ReverseVariablePrimal(), y_mad, 1.0)
        DiffOpt.reverse_differentiate!(model_mad)
        dp_mad = MOI.get(model_mad, DiffOpt.ReverseConstraintSet(), ParameterRef(p_mad)).value

        @test isapprox(dp_mad, dp_ref; atol=1e-6)
    end

    @testset "Multiple fixed variables - compare with DiffOpt" begin
        # min x^2 + y^2 + z^2 s.t. x + y + z >= 6, y == p, z == 2p

        model_ref = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
        MOI.set(model_ref, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
        set_silent(model_ref)
        @variable(model_ref, x_ref >= 0, start=2.0)
        @variable(model_ref, y_ref, start=1.0)
        @variable(model_ref, z_ref, start=2.0)
        @variable(model_ref, p_ref in MOI.Parameter(1.0))
        @constraint(model_ref, y_ref == p_ref)
        @constraint(model_ref, z_ref == 2 * p_ref)
        @constraint(model_ref, x_ref + y_ref + z_ref >= 6)
        @objective(model_ref, Min, x_ref^2 + y_ref^2 + z_ref^2)
        optimize!(model_ref)

        DiffOpt.empty_input_sensitivities!(model_ref)
        MOI.set(model_ref, DiffOpt.ForwardConstraintSet(), ParameterRef(p_ref), MOI.Parameter(1.0))
        DiffOpt.forward_differentiate!(model_ref)
        dx_ref = MOI.get(model_ref, DiffOpt.ForwardVariablePrimal(), x_ref)
        dy_ref = MOI.get(model_ref, DiffOpt.ForwardVariablePrimal(), y_ref)
        dz_ref = MOI.get(model_ref, DiffOpt.ForwardVariablePrimal(), z_ref)

        model_mad = Model(MadDiff.diff_optimizer(MadNLP.Optimizer; fixed_variable_treatment=MadNLP.MakeParameter))
        set_silent(model_mad)
        @variable(model_mad, x_mad >= 0, start=2.0)
        @variable(model_mad, y_mad, start=1.0)
        @variable(model_mad, z_mad, start=2.0)
        @variable(model_mad, p_mad in MOI.Parameter(1.0))
        @constraint(model_mad, y_mad == p_mad)
        @constraint(model_mad, z_mad == 2 * p_mad)
        @constraint(model_mad, x_mad + y_mad + z_mad >= 6)
        @objective(model_mad, Min, x_mad^2 + y_mad^2 + z_mad^2)
        optimize!(model_mad)

        DiffOpt.empty_input_sensitivities!(model_mad)
        MOI.set(model_mad, DiffOpt.ForwardConstraintSet(), ParameterRef(p_mad), MOI.Parameter(1.0))
        DiffOpt.forward_differentiate!(model_mad)
        dx_mad = MOI.get(model_mad, DiffOpt.ForwardVariablePrimal(), x_mad)
        dy_mad = MOI.get(model_mad, DiffOpt.ForwardVariablePrimal(), y_mad)
        dz_mad = MOI.get(model_mad, DiffOpt.ForwardVariablePrimal(), z_mad)

        @test isapprox(dx_mad, dx_ref; atol=1e-6)
        @test isapprox(dy_mad, dy_ref; atol=1e-6)
        @test isapprox(dz_mad, dz_ref; atol=1e-6)
        @test isapprox(dy_mad, 1.0; atol=1e-6)  # dy/dp = 1
        @test isapprox(dz_mad, 2.0; atol=1e-6)  # dz/dp = 2
    end
end

@testset "Single variable seed reverse mode" begin
    @testset "Single seed - linear constraint" begin
        function build_single_seed(m)
            @variable(m, x, start=0.5)
            @variable(m, y, start=0.5)
            @variable(m, z, start=1.0)
            @variable(m, p in MOI.Parameter(1.0))
            @variable(m, q in MOI.Parameter(1.0))
            @constraint(m, x + y == p)
            @constraint(m, z == q)
            @objective(m, Min, x^2 + y^2 + z^2)
            return [x, y, z], [p, q]
        end

        model_mad = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
        set_silent(model_mad)
        vars_mad, params_mad = build_single_seed(model_mad)
        optimize!(model_mad)

        DiffOpt.empty_input_sensitivities!(model_mad)
        MOI.set(model_mad, DiffOpt.ReverseVariablePrimal(), vars_mad[1], 1.0)  # only x
        DiffOpt.reverse_differentiate!(model_mad)
        grad_p_mad = MOI.get(model_mad, DiffOpt.ReverseConstraintSet(), ParameterRef(params_mad[1])).value
        grad_q_mad = MOI.get(model_mad, DiffOpt.ReverseConstraintSet(), ParameterRef(params_mad[2])).value

        model_diff = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
        MOI.set(model_diff, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
        set_silent(model_diff)
        vars_diff, params_diff = build_single_seed(model_diff)
        optimize!(model_diff)

        DiffOpt.empty_input_sensitivities!(model_diff)
        MOI.set(model_diff, DiffOpt.ReverseVariablePrimal(), vars_diff[1], 1.0)  # only x
        DiffOpt.reverse_differentiate!(model_diff)
        grad_p_diff = MOI.get(model_diff, DiffOpt.ReverseConstraintSet(), ParameterRef(params_diff[1])).value
        grad_q_diff = MOI.get(model_diff, DiffOpt.ReverseConstraintSet(), ParameterRef(params_diff[2])).value

        function solve_and_get_x(p_val, q_val)
            m = Model(MadNLP.Optimizer)
            set_silent(m)
            @variable(m, x, start=0.5)
            @variable(m, y, start=0.5)
            @variable(m, z, start=1.0)
            @constraint(m, x + y == p_val)
            @constraint(m, z == q_val)
            @objective(m, Min, x^2 + y^2 + z^2)
            optimize!(m)
            return value(x)
        end

        eps = 1e-6
        x_base = solve_and_get_x(1.0, 1.0)
        grad_p_fd = (solve_and_get_x(1.0 + eps, 1.0) - x_base) / eps
        grad_q_fd = (solve_and_get_x(1.0, 1.0 + eps) - x_base) / eps

        @test isapprox(grad_p_mad, grad_p_fd; atol=1e-4)
        @test isapprox(grad_q_mad, grad_q_fd; atol=1e-4)
        @test isapprox(grad_p_diff, grad_p_fd; atol=1e-4)
        @test isapprox(grad_q_diff, grad_q_fd; atol=1e-4)
        @test isapprox(grad_p_mad, grad_p_diff; atol=1e-4)
        @test isapprox(grad_q_mad, grad_q_diff; atol=1e-4)
    end

    @testset "Single seed - NLP with multiple parameters" begin
        function build_nlp_multi_param(m)
            @variable(m, x[1:3] >= 0, start=1.0)
            @variable(m, p[1:3] in MOI.Parameter.([1.0, 0.5, 0.3]))
            @constraint(m, x[1] + x[2] == p[1] + p[2])
            @constraint(m, x[2] + x[3] == p[2] + p[3])
            @objective(m, Min, sum(x[i]^2 for i in 1:3))
            return x, p
        end

        model_mad = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
        set_silent(model_mad)
        vars_mad, params_mad = build_nlp_multi_param(model_mad)
        optimize!(model_mad)

        DiffOpt.empty_input_sensitivities!(model_mad)
        MOI.set(model_mad, DiffOpt.ReverseVariablePrimal(), vars_mad[1], 1.0)
        DiffOpt.reverse_differentiate!(model_mad)
        grad_mad = [MOI.get(model_mad, DiffOpt.ReverseConstraintSet(), ParameterRef(params_mad[i])).value for i in 1:3]

        model_diff = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
        MOI.set(model_diff, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
        set_silent(model_diff)
        vars_diff, params_diff = build_nlp_multi_param(model_diff)
        optimize!(model_diff)

        DiffOpt.empty_input_sensitivities!(model_diff)
        MOI.set(model_diff, DiffOpt.ReverseVariablePrimal(), vars_diff[1], 1.0)
        DiffOpt.reverse_differentiate!(model_diff)
        grad_diff = [MOI.get(model_diff, DiffOpt.ReverseConstraintSet(), ParameterRef(params_diff[i])).value for i in 1:3]

        function solve_and_get_x1(p_vals)
            m = Model(MadNLP.Optimizer)
            set_silent(m)
            @variable(m, x[1:3] >= 0, start=1.0)
            @constraint(m, x[1] + x[2] == p_vals[1] + p_vals[2])
            @constraint(m, x[2] + x[3] == p_vals[2] + p_vals[3])
            @objective(m, Min, sum(x[i]^2 for i in 1:3))
            optimize!(m)
            return value(x[1])
        end

        eps = 1e-6
        p_base = [1.0, 0.5, 0.3]
        x1_base = solve_and_get_x1(p_base)
        grad_fd = [(solve_and_get_x1(p_base .+ eps .* (1:3 .== i)) - x1_base) / eps for i in 1:3]

        for i in 1:3
            @test isapprox(grad_mad[i], grad_fd[i]; atol=1e-3)
            @test isapprox(grad_diff[i], grad_fd[i]; atol=1e-3)
        end
        @test all(isapprox.(grad_mad, grad_diff; atol=1e-4))
    end
end

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
