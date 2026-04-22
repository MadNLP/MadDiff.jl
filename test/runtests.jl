using Test, Random, LinearAlgebra
using MadDiff
using MadNLP, MadIPM, HybridKKT
using NLPModels, CUDA, MadNLPGPU, MadNLPTests, QuadraticModels, ExaModels
using JuMP, DiffOpt, MathOptInterface
const MOI = MathOptInterface

const HAS_CUDA = CUDA.functional()

include("helpers.jl")
include("problems.jl")
include("test_diff.jl")

@testset "adjoint_{solve,mul}!" begin
for (KKTSystem, Callback) in [
    (MadNLP.SparseKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseUnreducedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.ScaledSparseKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseCondensedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.DenseKKTSystem, MadNLP.DenseCallback),
    (MadNLP.DenseCondensedKKTSystem, MadNLP.DenseCallback),
    (MadIPM.NormalKKTSystem, MadNLP.SparseCallback),
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

const SKIP_PROBLEMS = Set([
    "qp_multi_con",  # dual degeneracy
    "qp_mixed",  # primal degeneracy
    "nlp_trig",  # TODO: investigate (condensed)
])

include("test_exa.jl")

# Default tolerances. Comparisons use `isapprox(g1, g2; atol, rtol)` so the
# same pair works whether a sensitivity is O(1) or O(1e6). Problems where
# DiffOpt's own round-off exceeds these (e.g. `small_coef`) are compared
# against an analytic reference from `ANALYTIC_SENSITIVITIES` instead.
#
# A handful of KKT configs genuinely lose accuracy compared to the
# default and override these — see individual entries below.
const DX_ATOL = 1e-6
const DY_ATOL = 1e-3
const RTOL    = 1e-4

_kkt_config(; name, madnlp_opts = (;), maddiff_opts = (;),
             dx_atol = DX_ATOL, dy_atol = DY_ATOL, rtol = RTOL,
             skip_equality = false) =
    (; name, madnlp_opts, maddiff_opts, dx_atol, dy_atol, rtol, skip_equality)

const KKT_CONFIGS = [
    _kkt_config(name = "Default (SparseKKT)"),
    _kkt_config(
        name = "Default Skip (SparseKKT)",
        maddiff_opts = (; skip_kkt_refactorization = true),
        # `skip_kkt_refactorization` reuses the solver's final factor, which
        # was built before the last few iterations tightened μ — residuals
        # still have an O(μ) smear that IR can only partly clean up.
        dx_atol = 5e-4,
        dy_atol = 5e-3,
    ),
    _kkt_config(
        name = "SparseCondensedKKT",
        madnlp_opts = (; kkt_system = MadNLP.SparseCondensedKKTSystem, bound_relax_factor = 1e-6),
        # Condensed eliminates the slacks, so K_cond = H + JᵀΣJ with
        # Σ ≈ λ/s. At the test's `bound_relax_factor = 1e-6` the active
        # slacks sit near 1e-6, making K_cond's condition number
        # ~1e10–1e12. Both MadDiff *and* DiffOpt lose 3–4 digits on this
        # path; the augmented-system configs above are unaffected.
        dx_atol = 5e-4,
        dy_atol = 5e-3,
        rtol = 5e-3,
        skip_equality = true,
    ),
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
        # Same condensed-conditioning story as `SparseCondensedKKT`.
        dx_atol = 5e-4,
        dy_atol = 5e-3,
        rtol = 5e-3,
    ),
    # TODO: test MadIPM.Optimizer,
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
            # Tighter `bound_relax_factor` than the CPU condensed config
            # above recovers one digit, but Σ still dominates conditioning.
            dx_atol = 1e-4,
            dy_atol = 1e-2,
            rtol = 1e-3,
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
dx_atol = cfg.dx_atol
dy_atol = cfg.dy_atol
rtol    = cfg.rtol
skip_equality = cfg.skip_equality
for (fv_name, fv_handler) in FV_CONFIGS
@testset "$kkt_name × $fv_name" begin
Random.seed!(42)
for prob_name in sort!(collect(keys(PROBLEMS)))
    prob_name in SKIP_PROBLEMS && continue
    build, n_params, has_equality, is_lp = get_problem(prob_name)
    kkt_system = get(madnlp_opts_base, :kkt_system, nothing)
    kkt_system === MadIPM.NormalKKTSystem && !is_lp && continue
    skip_equality && has_equality && continue

    # Reference source: closed form (if registered — see `ANALYTIC_SENSITIVITIES`
    # in problems.jl) otherwise DiffOpt.
    analytic = get(ANALYTIC_SENSITIVITIES, prob_name, nothing)

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
        (dx_ref, dy_ref, dzl_ref, dzu_ref, dobj_ref) = if analytic !== nothing
            col = analytic(pidx)
            (col.dx .* dp, col.dy .* dp, col.dzl .* dp, col.dzu .* dp, col.dobj * dp)
        else
            run_diffopt(build; param_idx = pidx, dp = dp, madnlp_opts)
        end

        for (g1, g2) in zip(dx_mad, dx_ref)
            @test isapprox(g1, g2; atol=dx_atol, rtol=rtol)
        end
        for (g1, g2) in zip(dy_mad, dy_ref)
            @test isapprox(g1, g2; atol=dy_atol, rtol=rtol)
        end
        for (g1, g2) in zip(dzl_mad, dzl_ref)
            @test isapprox(g1, g2; atol=dy_atol, rtol=rtol)
        end
        for (g1, g2) in zip(dzu_mad, dzu_ref)
            @test isapprox(g1, g2; atol=dy_atol, rtol=rtol)
        end
        @test isapprox(dobj_mad, dobj_ref; atol=dy_atol, rtol=rtol)
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
        grad_ref = if analytic !== nothing
            [let c = analytic(j)
                 dot(dL_dx, c.dx) + dot(dL_dy, c.dy) +
                 dot(dL_dzl, c.dzl[1:length(dL_dzl)]) +
                 dot(dL_dzu, c.dzu[1:length(dL_dzu)]) +
                 dobj * c.dobj
             end for j in 1:n_params]
        else
            run_diffopt_reverse(build; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj, madnlp_opts)
        end
        for (g1, g2) in zip(grad_mad, grad_ref)
            @test isapprox(g1, g2; atol=dy_atol, rtol=rtol)
        end
    end

    # Consistency tests: jac/jact transposes each other, jvp/vjp are adjoint
    @testset "Consistency $prob_name" begin
        run_maddiff_consistency(build; madnlp_opts, maddiff_opts, atol=dy_atol, rtol=rtol)
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
