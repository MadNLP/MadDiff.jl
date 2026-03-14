module MadIPMExt

using LinearAlgebra: mul!
import MadDiff
import MadNLP
import MadNLP: AbstractKKTVector, primal, dual, dual_lb, dual_ub, solve_linear_system!
import MadIPM: NormalKKTSystem, MPCSolver, factorize_regularized_system!,
               AbstractBatchMPCSolver, UniformBatchMPCSolver, UniformBatchCallback,
               BatchUnreducedKKTVector, BatchPrimalVector, BatchVector,
               SparseUniformBatchKKTSystem, xp_lr, xp_ur,
               lower, upper
import MadIPM
import MadDiff: MadDiffSolver, refactorize_kkt!, _SensitivitySolverShim,
                _solve_with_refine!, _adjoint_solve_with_refine!,
                adjoint_solve_kkt!, adjoint_mul!,
                _adjoint_kktmul!, _adjoint_finish_bounds!, _adjoint_reduce_rhs!,
                has_hess_param, has_jac_param, has_lvar_param, has_uvar_param,
                has_lcon_param, has_ucon_param, has_grad_param
import NLPModels: hpprod!, jpprod!, lvar_jpprod!, uvar_jpprod!, lcon_jpprod!, ucon_jpprod!,
                  grad!, grad_param!, hptprod!, jptprod!,
                  lvar_jptprod!, uvar_jptprod!, lcon_jptprod!, ucon_jptprod!

function _adjoint_normal_solve!(kkt::NormalKKTSystem{T}, w::AbstractKKTVector) where {T}
    r1 = kkt.buffer_n
    r2 = kkt.buffer_m
    Σ = kkt.pr_diag
    wx = primal(w)
    wy = dual(w)

    r1 .= wx ./ Σ
    r2 .= wy
    mul!(r2, kkt.AT', r1, one(T), -one(T))  # A * r1 - wy

    solve_linear_system!(kkt.linear_solver, r2)
    wy .= r2

    r1 .= wx
    mul!(r1, kkt.AT, wy, -one(T), one(T))   # wx - A' * wy
    wx .= r1 ./ Σ
    return
end

function adjoint_solve_kkt!(kkt::NormalKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_normal_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::NormalKKTSystem{T},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T}
    wx = primal(w)
    wy = dual(w)
    xx = primal(x)
    xy = dual(x)

    mul!(wx, kkt.AT, xy, alpha, beta)
    mul!(wy, kkt.AT', xx, alpha, beta)
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function refactorize_kkt!(kkt, solver::MPCSolver)
    _solver = (kkt === solver.kkt) ? solver : _SensitivitySolverShim(solver, kkt)
    factorize_regularized_system!(_solver)
    return nothing
end

function _solve_with_refine!(
    sens::MadDiffSolver{T, KKT, Solver, VB, FC, RC, JC, TC},
    w::MadNLP.AbstractKKTVector,
    cache,
) where {T, KKT<:MadNLP.AbstractKKTSystem{T}, Solver<:MPCSolver{T}, VB, FC, RC, JC, TC}
    MadNLP.solve_kkt!(sens.kkt, w)
    return nothing
end

function _adjoint_solve_with_refine!(
    sens::MadDiffSolver{T, KKT, Solver, VB, FC, RC, JC, TC},
    w::MadNLP.AbstractKKTVector,
    cache,
) where {T, KKT<:MadNLP.AbstractKKTSystem{T}, Solver<:MPCSolver{T}, VB, FC, RC, JC, TC}
    adjoint_solve_kkt!(sens.kkt, w)
    return nothing
end

include("batch_api.jl")
include("batch_cache.jl")
include("batch_api_constructor.jl")
include("batch_packing.jl")
include("batch_kkt.jl")
include("batch_jvp.jl")
include("batch_vjp.jl")

end # module
