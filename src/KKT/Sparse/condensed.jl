# ============================================================================
# SparseCondensedKKTSystem — condensed sparse system (slacks eliminated).
#
# Forward solve:
#   r̂ₚ = rₚ + Jᵀ (D (r_z + Σₛ⁻¹ r_s))
#   K_cond x = r̂ₚ
#   r_z = -D (r_z + Σₛ⁻¹ r_s) + D J x
#   r_s = Σₛ⁻¹ (r_s + r_z)
#
# Adjoint updates:
#   g_z += Σₛ⁻¹ g_s, g_s = Σₛ⁻¹ g_s
#   g_buf = -g_z, g_buf2 = D g_z
#   g_x += J g_buf2, g_x = K_cond⁻¹ g_x
#   g_buf += Jᵀ g_x
#   g_z += D g_buf, g_s += Σₛ⁻¹ D g_buf
# ============================================================================

# `adjoint_mul!` on `AdjointKKT{…, <:SparseCondensedKKTSystem}` routes the
# primal–primal (Hessian) block through `_sym_hess_mul!` so Richardson IR
# is correct on GPU; see the header comment on `_sym_hess_mul!` in
# `KKT/adjoint.jl`. The bare-kkt `adjoint_mul!` below keeps the original
# `Symmetric(:L)` path and is used directly by tests/helpers on CPU.

function AdjointKKT(kkt::SparseCondensedKKTSystem{T, VT}) where {T, VT}
    n = size(kkt.hess_com, 1)
    return AdjointKKT{T, VT, typeof(kkt)}(
        kkt, _csc_diag(kkt.hess_com), similar(kkt.pr_diag, n),
    )
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    akkt::AdjointKKT{T, VT, <:SparseCondensedKKTSystem{T, VT}},
    x::AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T, VT}
    kkt       = akkt.kkt
    hess_diag = akkt.hess_diag
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc,   2)

    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+m)
    xz = view(full(x), n+m+1:n+2m)

    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2m)

    _sym_hess_mul!(wx, kkt.hess_com, hess_diag, akkt.scratch, xx, alpha, beta)
    mul!(wx, kkt.jt_csc,  xz, alpha, one(T))
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    wz .-= alpha .* xs
    ws .= beta .* ws .- alpha .* xz

    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag,
                     kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag,
                     alpha, beta)
    return w
end

# `adjoint_solve_kkt!` doesn't need the Hessian diagonal — the condensed
# factor is symmetric and CUSPARSE transpose-SpMV on jt_csc works correctly
# — so the default `AdjointKKT` delegation (defined in `adjoint.jl`)
# forwards to the bare-kkt method below unchanged.

# ---------- bare-kkt methods (still used directly by tests/helpers) ---------

function adjoint_mul!(
    w::AbstractKKTVector{T}, kkt::SparseCondensedKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T, VT, MT, QN}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc,   2)

    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+m)
    xz = view(full(x), n+m+1:n+2m)

    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2m)

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)
    mul!(wx, kkt.jt_csc,  xz, alpha, one(T))
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    wz .-= alpha .* xs
    ws .= beta .* ws .- alpha .* xz

    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag,
                     kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag,
                     alpha, beta)
    return w
end

function _adjoint_condensed_solve!(kkt::SparseCondensedKKTSystem{T},
                                    w::AbstractKKTVector) where {T}
    n, m = size(kkt.jt_csc)
    wx   = _madnlp_unsafe_wrap(full(w), n)
    ws   = view(full(w), n+1:n+m)
    wz   = view(full(w), n+m+1:n+2m)
    Σs   = view(kkt.pr_diag, n+1:n+m)
    buf  = kkt.buffer
    buf2 = kkt.buffer2

    # Reverse of `r_s ← Σs⁻¹(r_s + r_z)`:
    #   wz ← g_r_z + Σs⁻¹ g_r_s        (propagate through the sum)
    #   ws ← Σs⁻¹ g_r_s                 (partial g_r_s from this step)
    wz .+= ws ./ Σs
    ws ./= Σs

    # Reverse of `r_z ← -D·(r_z + Σs⁻¹ r_s) + D J x`:
    #   save -wz in `buf` (the -D·(r_z + Σs⁻¹ r_s) branch, scaled by D later),
    #   stage D·wz in `buf2` for the Jᵀ-contribution to g_x.
    buf  .= .-wz
    buf2 .= kkt.diag_buffer .* wz

    #   g_x ← g_x + Jᵀ · (D·wz)
    mul!(wx, kkt.jt_csc, buf2, one(T), one(T))

    # Reverse of `x = K_cond⁻¹ r̂_p` (symmetric K_cond ⇒ K_cond⁻¹ = K_cond⁻ᵀ):
    solve_linear_system!(kkt.linear_solver, wx)

    # Reverse of step A and the remaining C contribution fold together:
    #   buf ← -g_r_z_total + J·g_r̂_p
    # We write J·g_r̂_p into `buf2` first (β = 0) because CUSPARSE's
    # `mul!(dest, Transpose(CuSparseMatrixCSC), src, α, 1)` accumulation is
    # unreliable on GPU — the forward `solve_kkt!` avoids it for the same
    # reason. Using scratch-overwrite + broadcast keeps us on the supported
    # path and costs nothing (buf2 is already allocated).
    mul!(buf2, kkt.jt_csc', wx)
    buf .+= buf2

    # buf ← D·(J·g_r̂_p - g_r_z_total)  (fuses D from steps C and A).
    buf .= kkt.diag_buffer .* buf
    wz  .= buf
    ws  .+= buf ./ Σs
    return nothing
end

function adjoint_solve_kkt!(kkt::SparseCondensedKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_condensed_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

adjoint_solve_kkt!(
    ::SparseCondensedKKTSystem{T, VT, MT, QN}, ::AbstractKKTVector,
) where {T, VT, MT, QN<:CompactLBFGS} =
    error("MadDiff: SparseCondensedKKTSystem does not support CompactLBFGS. Use SparseKKTSystem instead.")
