module HybridKKTExt

import MadDiff
import MadDiff: adjoint_solve_kkt_system!, adjoint_mul!, _adjoint_kktmul!
import MadNLP: AbstractKKTVector, _madnlp_unsafe_wrap, dual_lb, dual_ub, full, solve_linear_system!

import HybridKKT
import HybridKKT: HybridCondensedKKTSystem, index_copy!
const Krylov = HybridKKT.Krylov
import LinearAlgebra: mul!, axpy!, Symmetric

function _adjoint_finish_bounds!(kkt::HybridCondensedKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    w.xp_lr .+= (kkt.l_lower ./ kkt.l_diag) .* dlb
    dlb .= .-dlb ./ kkt.l_diag
    w.xp_ur .-= (kkt.u_lower ./ kkt.u_diag) .* dub
    dub .= dub ./ kkt.u_diag
    return
end

function _adjoint_reduce_rhs!(kkt::HybridCondensedKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    dlb .-= w.xp_lr ./ kkt.l_diag
    dub .-= w.xp_ur ./ kkt.u_diag
    return
end

function _adjoint_hybrid_solve!(kkt::HybridCondensedKKTSystem{T}, w::AbstractKKTVector) where {T}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)
    mi = length(kkt.ind_ineq)
    G = kkt.G_csc

    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+mi)
    wc = view(full(w), n+mi+1:n+mi+m)

    wy = kkt.buffer6
    wz = kkt.buffer5
    index_copy!(wy, wc, kkt.ind_eq)
    index_copy!(wz, wc, kkt.ind_ineq)

    Σs = view(kkt.pr_diag, n+1:n+mi)

    buffer2 = kkt.buffer2
    buffer3 = kkt.buffer3
    buffer4 = kkt.buffer4

    # Reverse: extract condensation
    buffer4 .= .-wz
    ws .+= Σs .* wz
    wz .= .-ws
    fill!(buffer2, zero(T))
    index_copy!(buffer2, kkt.ind_ineq, ws)
    mul!(wx, kkt.jt_csc, buffer2, one(T), one(T))

    # Reverse: Golub & Greif extraction
    solve_linear_system!(kkt.linear_solver, wx)
    copyto!(buffer3, wx)
    mul!(wy, G, wx, -one(T), one(T))

    if kkt.etc[:cg_algorithm] ∈ (:cg, :gmres, :cr, :minres, :car)
        Krylov.krylov_solve!(
            kkt.iterative_linear_solver,
            kkt.S,
            wy;
            atol=0.0,
            rtol=1e-10,
            verbose=0,
        )
        copyto!(wy, kkt.iterative_linear_solver.x)
    elseif kkt.etc[:cg_algorithm] ∈ (:craigmr, )
        Krylov.krylov_solve!(
            kkt.iterative_linear_solver,
            kkt.G_csc,
            wy;
            N=kkt.S,
            atol=0.0,
            rtol=1e-10,
            verbose=0,
        )
        copyto!(wy, kkt.iterative_linear_solver.y)
    end

    mul!(wx, G', wy, one(T), zero(T))
    solve_linear_system!(kkt.linear_solver, wx)
    wx .+= buffer3
    wy .*= .-one(T)
    mul!(wy, G, wx, kkt.gamma[], one(T))

    # Reverse: condensation
    fill!(buffer2, zero(T))
    mul!(buffer2, kkt.jt_csc', wx, one(T), zero(T))
    buffer4 .+= view(buffer2, kkt.ind_ineq)
    wz .+= Σs .* view(buffer2, kkt.ind_ineq)

    ws .= buffer4
    fill!(wc, zero(T))
    index_copy!(wc, kkt.ind_eq, wy)
    index_copy!(wc, kkt.ind_ineq, wz)
    return
end

function adjoint_solve_kkt_system!(kkt::HybridCondensedKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_hybrid_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::HybridCondensedKKTSystem{T},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)
    mi = length(kkt.ind_ineq)

    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+mi)
    xz = view(full(x), n+mi+1:n+mi+m)

    wx = view(full(w), 1:n)
    ws = view(full(w), n+1:n+mi)
    wz = view(full(w), n+mi+1:n+mi+m)

    wz_ineq = view(wz, kkt.ind_ineq)
    xz_ineq = view(xz, kkt.ind_ineq)

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)
    mul!(wx, kkt.jt_csc, xz, alpha, one(T))
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    axpy!(-alpha, xs, wz_ineq)

    ws .= beta .* ws .- alpha .* xz_ineq

    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

end # module
