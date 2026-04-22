# ============================================================================
# SparseUnreducedKKTSystem — bounds carried explicitly in the augmented system.
#
# Forward solve:
#   rₗ = Dₗ⁻¹ rₗ, rᵤ = Dᵤ⁻¹ rᵤ   (Dₗ = diag(l_lower_aug), Dᵤ = diag(u_lower_aug))
#   K x = r
#   xₗ = -Dₗ xₗ,  xᵤ = Dᵤ xᵤ
#
# Adjoint updates:
#   gₗ = -Dₗ gₗ,  gᵤ = Dᵤ gᵤ
#   K y = g
#   gₗ = Dₗ⁻¹ gₗ, gᵤ = Dᵤ⁻¹ gᵤ    (with division guarded on zero diagonals)
# ============================================================================

function AdjointKKT(kkt::SparseUnreducedKKTSystem{T, VT}) where {T, VT}
    n = size(kkt.hess_com, 1)
    return AdjointKKT{T, VT, typeof(kkt)}(
        kkt, _csc_diag(kkt.hess_com), similar(kkt.pr_diag, n),
    )
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    akkt::AdjointKKT{T, VT, <:SparseUnreducedKKTSystem{T, VT}},
    x::AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T, VT}
    kkt = akkt.kkt
    _sym_hess_mul!(primal(w), kkt.hess_com, akkt.hess_diag, akkt.scratch,
                   primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x),   alpha, one(T))
    mul!(dual(w),   kkt.jac_com,  primal(x), alpha, beta)
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag,
                     kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag,
                     alpha, beta)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T}, kkt::SparseUnreducedKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T, VT, MT, QN}
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x),   alpha, one(T))
    mul!(dual(w),   kkt.jac_com,  primal(x), alpha, beta)
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag,
                     kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag,
                     alpha, beta)
    return w
end

function _adjoint_finish_bounds!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    dual_lb(w) .*= .-kkt.l_lower_aug
    dual_ub(w) .*=  kkt.u_lower_aug
    return nothing
end

function _adjoint_reduce_rhs!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    _safe_div(x, y) = iszero(y) ? x : x / y
    dual_lb(w) .= _safe_div.(dual_lb(w), kkt.l_lower_aug)
    dual_ub(w) .= _safe_div.(dual_ub(w), kkt.u_lower_aug)
    return nothing
end

function adjoint_solve_kkt!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    solve_linear_system!(kkt.linear_solver, full(w))
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

adjoint_solve_kkt!(
    ::SparseUnreducedKKTSystem{T, VT, MT, QN}, ::AbstractKKTVector,
) where {T, VT, MT, QN<:CompactLBFGS} =
    error("MadDiff: SparseUnreducedKKTSystem does not support CompactLBFGS. Use SparseKKTSystem instead.")
