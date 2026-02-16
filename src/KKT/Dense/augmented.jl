function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::AbstractDenseKKTSystem,
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T}
    (m, n) = size(kkt.jac)
    wx = view(primal(w), 1:n)
    ws = view(primal(w), n+1:length(primal(w)))
    wy = dual(w)
    wz = view(dual(w), kkt.ind_ineq)

    xx = view(primal(x), 1:n)
    xs = view(primal(x), n+1:length(primal(x)))
    xy = dual(x)
    xz = view(dual(x), kkt.ind_ineq)

    _symv!('L', alpha, kkt.hess, xx, beta, wx)
    if m > 0
        mul!(wx, kkt.jac', xy, alpha, one(T))
        mul!(wy, kkt.jac,  xx, alpha, beta)
    end
    ws .= beta .* ws .- alpha .* xz
    wz .-= alpha .* xs
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function _adjoint_finish_bounds!(kkt::DenseKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    w.xp_lr .+= (kkt.l_lower ./ kkt.l_diag) .* dlb
    dlb .= .-dlb ./ kkt.l_diag
    w.xp_ur .-= (kkt.u_lower ./ kkt.u_diag) .* dub
    dub .= dub ./ kkt.u_diag
    return
end

function _adjoint_reduce_rhs!(kkt::DenseKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    dlb .-= w.xp_lr ./ kkt.l_diag
    dub .-= w.xp_ur ./ kkt.u_diag
    return
end

function adjoint_solve_kkt_system!(kkt::DenseKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    solve_linear_system!(kkt.linear_solver, primal_dual(w))
    _adjoint_reduce_rhs!(kkt, w)
    return w
end
