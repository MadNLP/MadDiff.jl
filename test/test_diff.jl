# =============================================================================
# ChainRulesCore integration
# -----------------------------------------------------------------------------
# Validates the `differentiable_solve` / `SolverSpec` / `Components` surface
# and its `rrule`. Every numerical check compares the rrule's pullback
# against a direct call to the already-validated
# `MadDiff.vector_jacobian_product!` on the same solved MadDiffSolver, so we
# don't need FiniteDifferences or DiffOpt here — only the wiring.
# =============================================================================

using ChainRulesCore

# ---------- API surface ----------

@testset "differentiable_solve API" begin
    spec = MadDiff.SolverSpec(_ -> error("forward never called"))
    @test spec.kind === :build
    @test MadDiff.SolverSpec(_ -> nothing; kind = :update).kind === :update

    # Only the documented component names are accepted, and dups rejected.
    for bad in ((:not_a_thing,), (:x, :x))
        @test_throws ArgumentError MadDiff.differentiable_solve(spec; components = bad)
    end

    @test MadDiff.Components((:x, :y, :obj)) isa MadDiff.Components{(:x, :y, :obj)}

    # Passing legal components yields a callable; no forward is run yet.
    @test MadDiff.differentiable_solve(spec; components = (:x,)) isa Function

    # ChainRulesCore is loaded ⇒ the extension is active.
    @test Base.get_extension(MadDiff, :MadDiffChainRulesCoreExt) !== nothing
end

# ---------- end-to-end: solve → differentiable_solve → pullback ----------
# Uses the existing problem library (scalar QPs via MadDiff.diff_model) to
# drive a real solve, then checks three invariants of the rrule:
#   (a) the forward returns only the components the user asked for,
#   (b) the pullback's `grad_p` equals `vector_jacobian_product!(sens; ...)`
#       with the same seeds,
#   (c) components not listed in `Components` are silently dropped from both
#       the forward output and the adjoint back-solve.

function _solve_and_spec(build_model)
    model = _maddiff_model(MadNLP.Optimizer, (;), (;))
    set_silent(model)
    build_model(model)
    optimize!(model)
    solver = _solver_from_model(model)
    obj    = JuMP.objective_value(model)
    # Forward returns the already-solved solver regardless of `p` — we're
    # exercising the rrule's wiring, not retracing the solve.
    spec = MadDiff.SolverSpec(p -> (solver, obj))
    return spec, solver, obj
end

@testset "differentiable_solve forward" begin
    for pname in ("qp_ineq_only", "qp_eq_only", "qp_mixed")
        @testset "$(pname)" begin
            build, nparam, _ = PROBLEMS[pname]
            spec, solver, obj = _solve_and_spec(build)

            for comps in ((:x,), (:x, :y), (:x, :y, :zl, :zu, :obj))
                fn = MadDiff.differentiable_solve(spec; components = comps)
                sol = fn(zeros(nparam))
                @test Set(propertynames(sol)) == Set(comps)
                :x  in comps && @test sol.x  == MadNLP.variable(solver.x)
                :obj in comps && @test sol.obj == obj
            end
        end
    end
end

@testset "differentiable_solve pullback matches VJP" begin
    atol = 1e-8
    build, nparam, _ = PROBLEMS["qp_mixed"]
    spec, solver, _ = _solve_and_spec(build)
    sens = MadDiff.MadDiffSolver(solver)

    n_x, n_con = NLPModels.get_nvar(solver.nlp), NLPModels.get_ncon(solver.nlp)
    p0 = zeros(nparam)

    # Seed each component individually; the pullback's gradient must match a
    # direct `vector_jacobian_product!` call with the same seed.
    for (field, dim, kw) in (
        (:x,  n_x, :dL_dx),
        (:y,  n_con, :dL_dy),
        (:zl, n_x, :dL_dzl),
        (:zu, n_x, :dL_dzu),
    )
        comps = (field,)
        fn = MadDiff.differentiable_solve(spec; components = comps)
        _, pullback = ChainRulesCore.rrule(MadDiff._solve_forward,
                                            spec, MadDiff.Components(comps), p0)
        seed = zeros(dim); seed[1] = 1.0
        Δ = NamedTuple{(field,)}((seed,))
        _, _, _, grad_p = pullback(Δ)
        expected = MadDiff.vector_jacobian_product!(sens; (kw => seed,)...).grad_p
        @test isapprox(grad_p, expected; atol)
    end

    # `dobj` is a scalar seed (not an array), same check.
    fn = MadDiff.differentiable_solve(spec; components = (:obj,))
    _, pullback = ChainRulesCore.rrule(MadDiff._solve_forward,
                                        spec, MadDiff.Components((:obj,)), p0)
    _, _, _, grad_p = pullback((obj = 1.0,))
    expected = MadDiff.vector_jacobian_product!(sens; dobj = 1.0).grad_p
    @test isapprox(grad_p, expected; atol)
end

@testset "differentiable_solve component elision" begin
    build, nparam, _ = PROBLEMS["qp_mixed"]
    spec, solver, _ = _solve_and_spec(build)
    p0 = zeros(nparam)

    # Seeds for components the caller didn't list must be ignored entirely
    # (they get `nothing` in the adjoint, producing `grad_p = 0`).
    fn = MadDiff.differentiable_solve(spec; components = (:x,))
    _, pullback = ChainRulesCore.rrule(MadDiff._solve_forward,
                                        spec, MadDiff.Components((:x,)), p0)
    # Seed only on `y` (not requested) — pullback must short-circuit to zero.
    Δ = (y = ones(NLPModels.get_ncon(solver.nlp)),)
    _, _, _, grad_p = pullback(Δ)
    @test grad_p === ChainRulesCore.ZeroTangent()
end

# ---------- `jacobian_vector_product!` / `vector_jacobian_product!` rrules ----

@testset "JVP rrule" begin
    atol = 1e-8
    build, nparam, _ = PROBLEMS["qp_mixed"]
    _, solver, _ = _solve_and_spec(build)
    sens = MadDiff.MadDiffSolver(solver)

    Δp = [1.0]
    result, pullback = ChainRulesCore.rrule(
        MadDiff.jacobian_vector_product!, sens, Δp,
    )
    @test result isa MadDiff.JVPResult

    # The JVP is linear in Δp ⇒ adjoint is the VJP with the same seed.
    dx_seed = ones(NLPModels.get_nvar(solver.nlp))
    Δresult = ChainRulesCore.Tangent{typeof(result)}(dx = dx_seed)
    _, _, Δp_adj = pullback(Δresult)
    expected = MadDiff.vector_jacobian_product!(sens; dL_dx = dx_seed).grad_p
    @test isapprox(Δp_adj, expected; atol)
end

# ---------- GPU smoke test ----------
# Every scalar test above also works on GPU when the user wires `CuArray`
# through `madnlp_opts`; we only need to confirm the dispatch lands on the
# GPU code path and produces the same gradient as `vector_jacobian_product!`.

if HAS_CUDA
    @testset "differentiable_solve GPU" begin
        atol = 1e-6
        build, nparam, _ = PROBLEMS["qp_mixed"]
        model = _maddiff_model(MadNLP.Optimizer,
            (; linear_solver = CUDSSSolver), (;))
        set_silent(model)
        set_optimizer_attribute(model, "array_type", CuArray)
        build(model); optimize!(model)
        solver = _solver_from_model(model)
        spec = MadDiff.SolverSpec(p -> (solver, JuMP.objective_value(model)))

        sens = MadDiff.MadDiffSolver(solver)
        fn = MadDiff.differentiable_solve(spec; components = (:x,))
        sol = fn(CUDA.zeros(nparam))
        @test sol.x isa CuArray

        _, pullback = ChainRulesCore.rrule(MadDiff._solve_forward,
                                            spec, MadDiff.Components((:x,)),
                                            CUDA.zeros(nparam))
        seed = CUDA.zeros(length(sol.x)); CUDA.@allowscalar seed[1] = 1.0
        _, _, _, grad_p = pullback((x = seed,))
        expected = MadDiff.vector_jacobian_product!(sens; dL_dx = seed).grad_p
        @test isapprox(Array(grad_p), Array(expected); atol)
    end
end
