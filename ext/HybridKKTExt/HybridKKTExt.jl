module HybridKKTExt

# ============================================================================
# MadDiff × HybridKKT.
#
# Provides `adjoint_mul!` / `adjoint_solve_kkt!` for HybridKKT's
# `HybridCondensedKKTSystem` (condensed KKT with Golub–Greif extraction and a
# Krylov inner solve on the Schur complement).
# ============================================================================

import LinearAlgebra: Symmetric, axpy!, mul!

import MadNLP: AbstractKKTVector, _madnlp_unsafe_wrap,
    dual_lb, dual_ub, full, solve_linear_system!

import MadDiff: adjoint_mul!, adjoint_solve_kkt!, AdjointKKT, _csc_diag,
    _adjoint_finish_bounds!, _adjoint_kktmul!, _adjoint_reduce_rhs!,
    _sym_hess_mul!

import HybridKKT
import HybridKKT: HybridCondensedKKTSystem, index_copy!

const Krylov = HybridKKT.Krylov

# ---------- adjoint mul: w ← α Kᵀ x + β w ----------

function AdjointKKT(kkt::HybridCondensedKKTSystem{T, VT}) where {T, VT}
    n = size(kkt.hess_com, 1)
    return AdjointKKT{T, VT, typeof(kkt)}(
        kkt, _csc_diag(kkt.hess_com), similar(kkt.pr_diag, n),
    )
end

# Core body — shared between the AdjointKKT method (GPU-safe hess_diag path)
# and the bare-kkt method (CPU-only `Symmetric(:L)` path, used by tests).
@inline function _hybrid_adjoint_mul_core!(w, kkt, x, alpha::T, beta::T,
                                            hess_mul!::F) where {T, F}
    n, m, mi = size(kkt.hess_com, 1), size(kkt.jt_csc, 2), length(kkt.ind_ineq)
    xx, xs, xz = view(full(x), 1:n),
                 view(full(x), n+1:n+mi),
                 view(full(x), n+mi+1:n+mi+m)
    wx, ws, wz = view(full(w), 1:n),
                 view(full(w), n+1:n+mi),
                 view(full(w), n+mi+1:n+mi+m)
    wz_ineq = view(wz, kkt.ind_ineq)
    xz_ineq = view(xz, kkt.ind_ineq)

    hess_mul!(wx, xx)
    mul!(wx, kkt.jt_csc,  xz, alpha, one(T))
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    axpy!(-alpha, xs, wz_ineq)
    ws .= beta .* ws .- alpha .* xz_ineq

    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag,
                     kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag,
                     alpha, beta)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    akkt::AdjointKKT{T, VT, <:HybridCondensedKKTSystem{T}},
    x::AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T, VT}
    kkt = akkt.kkt
    return _hybrid_adjoint_mul_core!(w, kkt, x, alpha, beta,
        (wx, xx) -> _sym_hess_mul!(wx, kkt.hess_com, akkt.hess_diag, akkt.scratch,
                                   xx, alpha, beta))
end

function adjoint_mul!(
    w::AbstractKKTVector{T}, kkt::HybridCondensedKKTSystem{T},
    x::AbstractKKTVector, alpha = one(T), beta = zero(T),
) where {T}
    return _hybrid_adjoint_mul_core!(w, kkt, x, alpha, beta,
        (wx, xx) -> mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta))
end

# ---------- adjoint solve: reverse condensation + Golub–Greif ----------

function _adjoint_hybrid_solve!(kkt::HybridCondensedKKTSystem{T},
                                 w::AbstractKKTVector) where {T}
    n, m, mi = size(kkt.hess_com, 1), size(kkt.jt_csc, 2), length(kkt.ind_ineq)
    G  = kkt.G_csc
    Σs = view(kkt.pr_diag, n+1:n+mi)

    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+mi)
    wc = view(full(w), n+mi+1:n+mi+m)
    wy = kkt.buffer6
    wz = kkt.buffer5

    buf2, buf3, buf4 = kkt.buffer2, kkt.buffer3, kkt.buffer4

    index_copy!(wy, wc, kkt.ind_eq)
    index_copy!(wz, wc, kkt.ind_ineq)

    # reverse: extract condensation
    buf4 .= .-wz
    ws   .+= Σs .* wz
    wz   .= .-ws
    fill!(buf2, zero(T))
    index_copy!(buf2, kkt.ind_ineq, ws)
    mul!(wx, kkt.jt_csc, buf2, one(T), one(T))

    # reverse: Golub & Greif
    solve_linear_system!(kkt.linear_solver, wx)
    copyto!(buf3, wx)
    mul!(wy, G, wx, -one(T), one(T))

    _hybrid_krylov_adjoint!(kkt, wy)

    mul!(wx, G', wy, one(T), zero(T))
    solve_linear_system!(kkt.linear_solver, wx)
    wx .+= buf3
    wy .*= .-one(T)
    mul!(wy, G, wx, kkt.gamma[], one(T))

    # reverse: condensation
    fill!(buf2, zero(T))
    mul!(buf2, kkt.jt_csc', wx, one(T), zero(T))
    buf4 .+= view(buf2, kkt.ind_ineq)
    wz   .+= Σs .* view(buf2, kkt.ind_ineq)

    ws .= buf4
    fill!(wc, zero(T))
    index_copy!(wc, kkt.ind_eq,   wy)
    index_copy!(wc, kkt.ind_ineq, wz)
    return nothing
end

function _hybrid_krylov_adjoint!(kkt, wy)
    alg = kkt.etc[:cg_algorithm]
    if alg in (:cg, :gmres, :cr, :minres, :car)
        Krylov.krylov_solve!(kkt.iterative_linear_solver, kkt.S, wy;
                             atol = 0.0, rtol = 1e-10, verbose = 0)
        copyto!(wy, kkt.iterative_linear_solver.x)
    elseif alg === :craigmr
        Krylov.krylov_solve!(kkt.iterative_linear_solver, kkt.G_csc, wy;
                             N = kkt.S, atol = 0.0, rtol = 1e-10, verbose = 0)
        copyto!(wy, kkt.iterative_linear_solver.y)
    else
        error("HybridKKTExt: unsupported CG algorithm $(alg).")
    end
    return nothing
end

function adjoint_solve_kkt!(kkt::HybridCondensedKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_hybrid_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

end # module
