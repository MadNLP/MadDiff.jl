function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::ScaledSparseKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT, MT, QN}
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    _adjoint_scaled_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function _adjoint_scaled_post!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    w.xp_lr .-= (kkt.l_lower ./ kkt.l_diag) .* dlb
    w.xp_ur .+= (kkt.u_lower ./ kkt.u_diag) .* dub
    dlb .= dlb ./ kkt.l_diag
    dub .= .-dub ./ kkt.u_diag
    return
end

function _adjoint_scaled_pre!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    r3 = kkt.buffer1
    r4 = kkt.buffer2
    r3 .= w.xp
    r4 .= w.xp
    w.xp .*= kkt.scaling_factor
    dual_lb(w) .+= r3[kkt.ind_lb] ./ sqrt.(kkt.l_diag)
    dual_ub(w) .+= r4[kkt.ind_ub] ./ sqrt.(kkt.u_diag)
    return
end

# Adjoint solve for the scaled reduced KKT system (ScaledSparseKKTSystem).
#
# Forward solve:
#   r̂ₚ = S rₚ + Dₗ⁻½ rₗ + Dᵤ⁻½ rᵤ
#   K_scaled z = [r̂ₚ; r_d]
#   xₚ = S zₚ
#   rₗ = Dₗ⁻¹(rₗ - L xₗ), rᵤ = Dᵤ⁻¹(-rᵤ + U xᵤ)
# with S = diag(scaling_factor), Dₗ = diag(l_diag), Dᵤ = diag(u_diag),
# L = diag(l_lower), U = diag(u_lower).
#
# Adjoint (reverse) updates:
#   xₗ += -L Dₗ⁻¹ gₗ, xᵤ += U Dᵤ⁻¹ gᵤ
#   gₗ = Dₗ⁻¹ gₗ,     gᵤ = -Dᵤ⁻¹ gᵤ
#   zₚ += S gₚ
#   gₗ += Dₗ⁻½ zₚ,    gᵤ += Dᵤ⁻½ zₚ
function adjoint_solve_kkt_system!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    _adjoint_scaled_post!(kkt, w)
    w.xp .*= kkt.scaling_factor
    solve_linear_system!(kkt.linear_solver, primal_dual(w))
    _adjoint_scaled_pre!(kkt, w)
    return w
end

function adjoint_solve_kkt_system!(
    kkt::ScaledSparseKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
    error("Quasi-Newton approximation of the Hessian is not supported by the KKT formulation ScaledSparseKKTSystem. Please use SparseKKTSystem instead.")
end
