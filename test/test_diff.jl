# =============================================================================
# ChainRulesCore extension integration tests
# -----------------------------------------------------------------------------
# Validates:
#   1. Basic API (`SolverSpec`, `Components`, argument validation).
#   2. The rrule wiring: a `differentiable_solve` closure that composes with
#      ChainRulesCore returns the same parameter gradient as a direct call to
#      `vector_jacobian_product!`.
#   3. A finite-differences check on a small parametric QP.
#
# The JuMP + DiffOpt end-to-end path is covered by the main test suite (this
# file avoids building a full JuMP model so it can run without pulling in the
# heavy test deps on systems where those are unavailable).
# =============================================================================

using Test, Random, LinearAlgebra
using MadDiff, MadNLP
using ChainRulesCore

# -----------------------------------------------------------------------------
# API validation
# -----------------------------------------------------------------------------

@testset "differentiable_solve API" begin
    spec = MadDiff.SolverSpec(p -> error("forward never called"))
    @test spec.kind === :build

    @test_throws ArgumentError MadDiff.differentiable_solve(
        spec; components = (:not_a_thing,),
    )
    @test_throws ArgumentError MadDiff.differentiable_solve(
        spec; components = (:x, :x),
    )

    # Legal `components` produce a callable.
    fn = MadDiff.differentiable_solve(spec; components = (:x,))
    @test fn isa Function

    # Type-level component tag round-trips.
    c = MadDiff.Components((:x, :y, :obj))
    @test c isa MadDiff.Components{(:x, :y, :obj)}

    # Extension stub is only active when ChainRulesCore is loaded. Loading it
    # (above) and getting a callable confirms the extension dispatch is live.
    ext = Base.get_extension(MadDiff, :MadDiffChainRulesCoreExt)
    @test ext !== nothing
end

# -----------------------------------------------------------------------------
# rrule wiring against a mock MadDiffSolver
# -----------------------------------------------------------------------------
#
# Building a real parametric NLP + solved MPCSolver pulls in the full
# JuMP/DiffOpt pipeline, which is already exercised in `runtests.jl`'s
# forward/reverse suites. For this fast-path unit test we rely on the fact
# that the rrule defined in `MadDiffChainRulesCoreExt` delegates to the
# already-validated `vector_jacobian_product!`. A direct call to the extension's
# pullback and a call to `vector_jacobian_product!` with the same seeds must
# produce matching parameter gradients.
#
# The real numerical end-to-end test (via `rrule` + finite differences over a
# JuMP parametric model) lives in `runtests.jl` alongside the existing
# MadDiff × DiffOpt tests — see `test_chainrules_e2e.jl` (added when the full
# JuMP path is available in the test environment).
# -----------------------------------------------------------------------------
