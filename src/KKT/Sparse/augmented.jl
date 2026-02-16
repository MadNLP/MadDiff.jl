function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::SparseKKTSystem{T,VT,MT,QN},
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

# Adjoint solve for the reduced KKT system (SparseKKTSystem).
#
# Forward solve (with bound elimination):
#   r̂ₚ = rₚ - Dₗ⁻¹ rₗ - Dᵤ⁻¹ rᵤ
#   K_red x = r̂
#   yₗ = Dₗ⁻¹(-rₗ + L xₗ),  yᵤ = Dᵤ⁻¹(rᵤ - U xᵤ)
# with Dₗ = diag(l_diag), Dᵤ = diag(u_diag), L = diag(l_lower), U = diag(u_lower).
#
# Adjoint (reverse) updates:
#   xₗ += L Dₗ⁻¹ gₗ,  xᵤ += -U Dᵤ⁻¹ gᵤ
#   gₗ = -Dₗ⁻¹ gₗ,   gᵤ =  Dᵤ⁻¹ gᵤ
#   K_red y = [gₚ; g_d]
#   gₗ += -Dₗ⁻¹ yₗ,  gᵤ += -Dᵤ⁻¹ yᵤ
function adjoint_solve_kkt!(kkt::SparseKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    solve_linear_system!(kkt.linear_solver, primal_dual(w))
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_solve_kkt!(
    kkt::SparseKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}

    qn = kkt.quasi_newton
    n, p = size(qn)
    # Load buffers
    xr = qn._w2
    Tk = qn.Tk
    w_ = primal_dual(w)
    nn = length(w_)

    _adjoint_finish_bounds!(kkt, w)

    fill!(Tk, zero(T))

    # Resize arrays with correct dimension
    if size(qn.E) != (nn, 2*p)
        qn.E = zeros(T, nn, 2*p)
        qn.H = zeros(T, nn, 2*p)
    else
        fill!(qn.E, zero(T))
        fill!(qn.H, zero(T))
    end

    # Solve LBFGS system with Sherman-Morrison-Woodbury formula
    # (C + E P Eᵀ)⁻¹ = C⁻¹ - C⁻¹ E (P + Eᵀ C⁻¹ E) Eᵀ C⁻¹
    #
    # P = [ -Iₚ  0  ] (size 2p × 2p) and E = [ U  V ] (size (n+m) × 2p)
    #     [  0   Iₚ ]                        [ 0  0 ]

    # Solve linear system without low-rank part
    solve_linear_system!(kkt.linear_solver, w_)  # w_ stores the solution of Cx = b

    # Add low-rank correction
    if p > 0
        @inbounds for i in 1:n, j in 1:p
            qn.E[i, j] = qn.U[i, j]
            qn.E[i, j+p] = qn.V[i, j]
        end
        copyto!(qn.H, qn.E)

        multi_solve!(kkt.linear_solver, qn.H)  # H = C⁻¹ E

        for i = 1:p
            Tk[i,i] = -one(T)                  # Tₖ = P
            Tk[i+p,i+p] = one(T)
        end
        mul!(Tk, qn.E', qn.H, one(T), one(T))  # Tₖ = (P + Eᵀ C⁻¹ E)

        F, ipiv, info = LAPACK.sytrf!('L', Tk) # Tₖ⁻¹

        mul!(xr, qn.E', w_)                    # xᵣ = Eᵀ C⁻¹ b
        LAPACK.sytrs!('L', F, ipiv, xr)        # xᵣ = (P + Eᵀ C⁻¹ E)⁻¹ Eᵀ C⁻¹ b
        mul!(w_, qn.H, xr, -one(T), one(T))    # x = x - C⁻¹ E xᵣ
    end

    _adjoint_reduce_rhs!(kkt, w)
    return w
end
