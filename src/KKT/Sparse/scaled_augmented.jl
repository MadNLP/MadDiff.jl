# ============================================================================
# ScaledSparseKKTSystem — reduced sparse augmented system with primal scaling.
#
# Forward solve:
#   r̂ₚ = S rₚ + Dₗ⁻½ rₗ + Dᵤ⁻½ rᵤ
#   K_scaled z = [r̂ₚ; r_d]
#   xₚ = S zₚ
#   rₗ = Dₗ⁻¹(rₗ - L xₗ), rᵤ = Dᵤ⁻¹(-rᵤ + U xᵤ)
#
# Adjoint updates:
#   xₗ += -L Dₗ⁻¹ gₗ, xᵤ += U Dᵤ⁻¹ gᵤ
#   gₗ = Dₗ⁻¹ gₗ,     gᵤ = -Dᵤ⁻¹ gᵤ
#   zₚ += S gₚ
#   gₗ += Dₗ⁻½ zₚ,    gᵤ += Dᵤ⁻½ zₚ
# ============================================================================

function AdjointKKT(kkt::ScaledSparseKKTSystem{T, VT}) where {T, VT}
    n = size(kkt.hess_com, 1)
    return AdjointKKT{T, VT, typeof(kkt)}(
        kkt, _csc_diag(kkt.hess_com), similar(kkt.pr_diag, n),
    )
end

# Shared by AdjointKKT and bare-kkt variants — the ScaledSparse sign
# convention on the bound-dual coupling is different enough from
# `_adjoint_kktmul!` that it has to be inlined here.
@inline function _scaled_adjoint_bound_update!(w, x, kkt, alpha::T, beta::T) where {T}
    primal(w) .+= alpha .* kkt.reg     .* primal(x)
    dual(w)   .+= alpha .* kkt.du_diag .* dual(x)
    w.xp_lr   .+= alpha .* kkt.l_lower .* dual_lb(x)
    w.xp_ur   .+= alpha .* kkt.u_lower .* dual_ub(x)
    dual_lb(w) .= beta .* dual_lb(w) .+ alpha .* (.-x.xp_lr .+ kkt.l_diag .* dual_lb(x))
    dual_ub(w) .= beta .* dual_ub(w) .+ alpha .* ( x.xp_ur .- kkt.u_diag .* dual_ub(x))
    return nothing
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    akkt::AdjointKKT{T, VT, <:ScaledSparseKKTSystem{T, VT}},
    x::AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T, VT}
    kkt = akkt.kkt
    _sym_hess_mul!(primal(w), kkt.hess_com, akkt.hess_diag, akkt.scratch,
                   primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x),   alpha, one(T))
    mul!(dual(w),   kkt.jac_com,  primal(x), alpha, beta)
    _scaled_adjoint_bound_update!(w, x, kkt, alpha, beta)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T}, kkt::ScaledSparseKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T, VT, MT, QN}
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x),   alpha, one(T))
    mul!(dual(w),   kkt.jac_com,  primal(x), alpha, beta)
    _scaled_adjoint_bound_update!(w, x, kkt, alpha, beta)
    return w
end

function _adjoint_finish_bounds!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    dlb, dub = dual_lb(w), dual_ub(w)
    w.xp_lr .-= (kkt.l_lower ./ kkt.l_diag) .* dlb
    w.xp_ur .+= (kkt.u_lower ./ kkt.u_diag) .* dub
    dlb .=   dlb ./ kkt.l_diag
    dub .= .-dub ./ kkt.u_diag
    return nothing
end

function _adjoint_reduce_rhs!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    r3, r4 = kkt.buffer1, kkt.buffer2
    r3 .= w.xp
    r4 .= w.xp
    w.xp .*= kkt.scaling_factor
    dual_lb(w) .+= r3[kkt.ind_lb] ./ sqrt.(kkt.l_diag)
    dual_ub(w) .+= r4[kkt.ind_ub] ./ sqrt.(kkt.u_diag)
    return nothing
end

function adjoint_solve_kkt!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    w.xp .*= kkt.scaling_factor
    solve_linear_system!(kkt.linear_solver, primal_dual(w))
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

adjoint_solve_kkt!(
    ::ScaledSparseKKTSystem{T, VT, MT, QN}, ::AbstractKKTVector,
) where {T, VT, MT, QN<:CompactLBFGS} =
    error("MadDiff: ScaledSparseKKTSystem does not support CompactLBFGS. Use SparseKKTSystem instead.")
