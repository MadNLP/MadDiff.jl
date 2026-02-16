function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::SparseCondensedKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT, MT, QN}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+m)
    xz = view(full(x), n+m+1:n+2*m)

    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2*m)

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)
    mul!(wx, kkt.jt_csc,  xz, alpha, one(T))
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    wz .-= alpha .* xs
    ws .= beta .* ws .- alpha .* xz

    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function _adjoint_condensed_solve!(kkt::SparseCondensedKKTSystem{T}, w::AbstractKKTVector) where T
    (n,m) = size(kkt.jt_csc)

    # Decompose buffers
    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2*m)
    Σs = view(kkt.pr_diag, n+1:n+m)

    buf = kkt.buffer
    buf2 = kkt.buffer2

    # Reverse ws = (ws + wz) ./ Σs
    wz .+= ws ./ Σs
    ws ./= Σs

    # Reverse wz = -buf + diag_buffer .* buf2
    buf .= .-wz
    buf2 .= kkt.diag_buffer .* wz

    # Reverse buf2 = jt_csc' * wx
    mul!(wx, kkt.jt_csc, buf2, one(T), one(T))

    # Reverse wx = A⁻¹(wx)
    solve_linear_system!(kkt.linear_solver, wx)

    # Reverse wx = wx + jt_csc * buf
    mul!(buf, kkt.jt_csc', wx, one(T), one(T))

    # Reverse buf = diag_buffer .* (wz + ws ./ Σs)
    buf .= kkt.diag_buffer .* buf
    wz .= buf
    ws .+= buf ./ Σs
    return
end

# Adjoint solve for the condensed KKT system (SparseCondensedKKTSystem).
#
# Forward solve:
#   r̂ₚ = rₚ + Jᵀ (D (r_z + Σₛ⁻¹ r_s))
#   K_cond x = r̂ₚ
#   r_z = -D (r_z + Σₛ⁻¹ r_s) + D J x
#   r_s = Σₛ⁻¹ (r_s + r_z)
#
# Adjoint (reverse) updates:
#   g_z += Σₛ⁻¹ g_s, g_s = Σₛ⁻¹ g_s
#   g_buf = -g_z, g_buf2 = D g_z
#   g_x += J g_buf2, g_x = K_cond⁻¹ g_x
#   g_buf += Jᵀ g_x
#   g_z += D g_buf, g_s += Σₛ⁻¹ D g_buf
function adjoint_solve_kkt!(kkt::SparseCondensedKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_condensed_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_solve_kkt!(
    kkt::SparseCondensedKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
    error("Quasi-Newton approximation of the Hessian is not supported by the KKT formulation SparseCondensedKKTSystem. Please use SparseKKTSystem instead.")
end
