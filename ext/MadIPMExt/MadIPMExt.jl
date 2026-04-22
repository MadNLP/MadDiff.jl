module MadIPMExt

# ============================================================================
# MadDiff × MadIPM.
#
#   * `NormalKKTSystem` adjoint linear algebra.
#   * `refactorize_kkt!(kkt, ::MPCSolver)` — MadIPM uses its own regularized
#     factorization rather than MadNLP's inertia corrector.
#   * `_kkt_solve_with_refine!` override — MadIPM's KKT solves refine
#     internally, so MadDiff skips the outer Richardson loop.
# ============================================================================

import LinearAlgebra: mul!

import MadNLP
import MadNLP: AbstractKKTVector, dual, primal, solve_kkt!, solve_linear_system!,
    solve_refine!

using MadIPM
using MadIPM: MPCSolver, NormalKKTSystem, factorize_regularized_system!,
    UniformBatchMPCSolver, UniformBatchCallback,
    BatchUnreducedKKTVector, BatchPrimalVector,
    SparseUniformBatchKKTSystem, lower, upper

import MadDiff
import MadDiff: MadDiffSolver, _SensitivitySolverShim,
    _adjoint_finish_bounds!, _adjoint_kktmul!, _adjoint_reduce_rhs!,
    _kkt_solve_with_refine!, adjoint_mul!, adjoint_solve_kkt!,
    has_grad_param, has_hess_param, has_jac_param,
    has_lcon_param, has_lvar_param, has_ucon_param, has_uvar_param,
    refactorize_kkt!

import NLPModels: grad!
import ParametricNLPModels: grad_param!, hpprod!, hptprod!, jpprod!, jptprod!,
    lcon_jpprod!, lcon_jptprod!, lvar_jpprod!, lvar_jptprod!,
    ucon_jpprod!, ucon_jptprod!, uvar_jpprod!, uvar_jptprod!

# ---------- NormalKKTSystem ----------

function adjoint_mul!(
    w::AbstractKKTVector{T}, kkt::NormalKKTSystem{T},
    x::AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T}
    mul!(primal(w), kkt.AT,  dual(x),   alpha, beta)
    mul!(dual(w),   kkt.AT', primal(x), alpha, beta)
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag,
                     kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag,
                     alpha, beta)
    return w
end

function _adjoint_normal_solve!(kkt::NormalKKTSystem{T},
                                 w::AbstractKKTVector) where {T}
    r1, r2 = kkt.buffer_n, kkt.buffer_m
    Σ      = kkt.pr_diag
    wx, wy = primal(w), dual(w)

    r1 .= wx ./ Σ
    r2 .= wy
    mul!(r2, kkt.AT', r1, one(T), -one(T))  # r2 ← A·r1 − wy
    solve_linear_system!(kkt.linear_solver, r2)
    wy .= r2

    r1 .= wx
    mul!(r1, kkt.AT, wy, -one(T), one(T))   # r1 ← wx − Aᵀ·wy
    wx .= r1 ./ Σ
    return nothing
end

function adjoint_solve_kkt!(kkt::NormalKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_normal_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

# ---------- MPCSolver integration ----------

function refactorize_kkt!(kkt, solver::MPCSolver)
    shim = kkt === solver.kkt ? solver : _SensitivitySolverShim(solver, kkt)
    factorize_regularized_system!(shim)
    return nothing
end

function _kkt_solve_with_refine!(
    sens::MadDiffSolver{T, KKT, Solver}, w::AbstractKKTVector, _cache, refine!,
) where {T, KKT, Solver <: MPCSolver{T}}
    refine! === solve_refine! ? solve_kkt!(sens.kkt, w) :
                                adjoint_solve_kkt!(sens.kkt, w)
    return nothing
end

# ---------- batch support ----------

include("batch_api.jl")
include("batch_cache.jl")
include("batch_packing.jl")
include("batch_kkt.jl")
include("batch_jvp.jl")
include("batch_vjp.jl")

end # module
