# ============================================================================
# SparseKKTSystem — reduced (bounds eliminated) sparse augmented system.
#
# Forward solve (with bound elimination):
#   r̂ₚ = rₚ - Dₗ⁻¹ rₗ - Dᵤ⁻¹ rᵤ
#   K_red x = r̂
#   yₗ = Dₗ⁻¹(-rₗ + L xₗ),  yᵤ = Dᵤ⁻¹(rᵤ - U xᵤ)
# with Dₗ = diag(l_diag), Dᵤ = diag(u_diag), L = diag(l_lower), U = diag(u_lower).
#
# Adjoint updates:
#   xₗ += L Dₗ⁻¹ gₗ,  xᵤ += -U Dᵤ⁻¹ gᵤ
#   gₗ = -Dₗ⁻¹ gₗ,   gᵤ =  Dᵤ⁻¹ gᵤ
#   K_red y = [gₚ; g_d]
#   gₗ += -Dₗ⁻¹ yₗ,  gᵤ += -Dᵤ⁻¹ yᵤ
# ============================================================================

function AdjointKKT(kkt::SparseKKTSystem{T, VT}) where {T, VT}
    n = size(kkt.hess_com, 1)
    return AdjointKKT{T, VT, typeof(kkt)}(
        kkt, _csc_diag(kkt.hess_com), similar(kkt.pr_diag, n),
    )
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    akkt::AdjointKKT{T, VT, <:SparseKKTSystem{T, VT}},
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

# Bare-kkt fallback (used by tests/helpers on CPU, where `Symmetric(:L)` works
# correctly; on GPU this path is not reachable because `AdjointKKT` intercepts).
function adjoint_mul!(
    w::AbstractKKTVector{T}, kkt::SparseKKTSystem{T,VT,MT,QN},
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

function adjoint_solve_kkt!(kkt::SparseKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    solve_linear_system!(kkt.linear_solver, primal_dual(w))
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

adjoint_solve_kkt!(
    ::SparseKKTSystem{T, VT, MT, QN}, ::AbstractKKTVector,
) where {T, VT, MT, QN<:CompactLBFGS} =
    error("MadDiff: reverse mode does not yet support CompactLBFGS Hessian approximation.")
