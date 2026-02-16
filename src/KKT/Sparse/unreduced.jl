function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::SparseUnreducedKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT, MT, QN}
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function _adjoint_unreduced_post!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    dual_lb(w) .*= .-kkt.l_lower_aug
    dual_ub(w) .*= kkt.u_lower_aug
    return
end

function _adjoint_unreduced_pre!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    f(x,y) = iszero(y) ? x : x/y
    wzl = dual_lb(w)
    wzu = dual_ub(w)
    wzl .= f.(wzl, kkt.l_lower_aug)
    wzu .= f.(wzu, kkt.u_lower_aug)
    return
end

# Adjoint solve for the unreduced KKT system (SparseUnreducedKKTSystem).
#
# Forward solve:
#   rₗ = Dₗ⁻¹ rₗ, rᵤ = Dᵤ⁻¹ rᵤ   (with Dₗ = diag(l_lower_aug), Dᵤ = diag(u_lower_aug))
#   K x = r
#   xₗ = -Dₗ xₗ,  xᵤ = Dᵤ xᵤ
#
# Adjoint (reverse) updates:
#   gₗ = -Dₗ gₗ,  gᵤ = Dᵤ gᵤ
#   K y = g
#   gₗ = Dₗ⁻¹ gₗ, gᵤ = Dᵤ⁻¹ gᵤ
function adjoint_solve_kkt_system!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    _adjoint_unreduced_post!(kkt, w)
    solve_linear_system!(kkt.linear_solver, full(w))
    _adjoint_unreduced_pre!(kkt, w)
    return w
end

function adjoint_solve_kkt_system!(
    kkt::SparseUnreducedKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
    error("Quasi-Newton approximation of the Hessian is not supported by the KKT formulation SparseUnreducedKKTSystem. Please use SparseKKTSystem instead.")
end
