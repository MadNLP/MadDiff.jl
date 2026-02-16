module MadNCLExt

import MadDiff
import MadDiff: adjoint_solve_kkt!, adjoint_mul!, _adjoint_kktmul!, MadDiffSolver, _adjoint_finish_bounds!, _adjoint_reduce_rhs!
import MadNLP: AbstractKKTVector, primal, dual, dual_lb, dual_ub, full, solve_linear_system!, _symv!
import MadNCL: NCLSolver, K1sAuglagKKTSystem, K2rAuglagKKTSystem, symul!
import LinearAlgebra: mul!

function MadDiffSolver(solver::NCLSolver; kwargs...)
    return MadDiffSolver(solver.ipm; kwargs...)
end

function _adjoint_k1s_solve!(kkt::K1sAuglagKKTSystem{T}, w::AbstractKKTVector) where {T}
    nx, nr, ns = kkt.nlp.nx, kkt.nlp.nr, length(kkt.ind_ineq)
    m = kkt.m
    Σs = view(kkt.pr_diag, nx+nr+1:nx+nr+ns)
    θk = kkt.θk
    θk_ineq = view(θk, kkt.ind_ineq)
    denom = Σs .+ θk_ineq

    ds = view(kkt.buffer1, nx+1:nx+ns)
    dy = kkt.buffer2
    dx = kkt.buffer3

    w_ = full(w)
    wx = view(w_, 1:nx)
    wr = view(w_, nx+1:nx+nr)
    ws = view(w_, nx+nr+1:nx+nr+ns)
    wy = view(w_, nx+nr+ns+1:nx+nr+ns+m)

    # Step N (adjoint): wr = (wr - wy) ./ θk
    dy .= wr ./ θk
    wy .-= dy
    wr .= dy

    # Step M (adjoint): wy = θk .* (dy - wy) .+ wr
    dy .= θk .* wy
    wr .+= wy
    wy .*= .-θk

    # Step L (adjoint): dy[ind_ineq] .-= ws
    dy_ineq = view(dy, kkt.ind_ineq)
    ws .-= dy_ineq

    # Step K (adjoint): ws = (ds .+ θk .* dy) ./ (Σs + θk)
    ds .= ws ./ denom
    dy_ineq .+= (θk_ineq ./ denom) .* ws

    # Step J (adjoint): dy = J * wx
    mul!(wx, kkt.jt_csc, dy, one(T), one(T))

    # Step I (adjoint): wx = dx
    dx .= wx

    # Step H (adjoint): dx = A \ dx
    solve_linear_system!(kkt.linear_solver, dx)

    # Step G (adjoint): dx = dx + J' * dy
    fill!(dy, zero(T))
    mul!(dy, kkt.jt_csc', dx, one(T), zero(T))

    # Step F (adjoint): dy[ind_ineq] = (θk .* ds) ./ (Σs + θk)
    ds .+= (θk_ineq ./ denom) .* dy_ineq

    # Step E (adjoint): dy = 0
    fill!(dy, zero(T))

    # Step D (adjoint): ds = ws - dy[ind_ineq]
    ws .= ds
    dy_ineq = view(dy, kkt.ind_ineq)
    dy_ineq .-= ds

    # Step C (adjoint): dx = dx + wx
    wx .= dx

    # Step B (adjoint): dx = J' * dy
    mul!(dy, kkt.jt_csc', dx, one(T), one(T))

    # Step A (adjoint): dy = θk .* wy - wr
    wy .+= θk .* dy
    wr .-= dy
    return
end

function _adjoint_k2r_solve!(kkt::K2rAuglagKKTSystem{T}, w::AbstractKKTVector) where {T}
    nx, nr, ns = kkt.nlp.nx, kkt.nlp.nr, length(kkt.ind_ineq)
    m = kkt.m
    θk = kkt.θk

    d = kkt.buffer3

    w_ = full(w)
    wx = view(w_, 1:nx)
    wr = view(w_, nx+1:nx+nr)
    ws = view(w_, nx+nr+1:nx+nr+ns)
    wy = view(w_, nx+nr+ns+1:nx+nr+ns+m)

    # Step 4 (adjoint): wr = θk .* (wr - wy)
    wy .-= θk .* wr
    wr .*= θk

    # Step 3/2/1 (adjoint): d = [wx; ws; wy]
    d[1:nx] .= wx
    d[nx+1:nx+ns] .= ws
    d[nx+ns+1:nx+ns+m] .= wy
    solve_linear_system!(kkt.linear_solver, d)

    wx .= view(d, 1:nx)
    ws .= view(d, nx+1:nx+ns)
    wy .= view(d, nx+ns+1:nx+ns+m)

    # Step 1 (adjoint): d3 = wy - θk .* wr
    wr .-= θk .* wy
    return
end

function adjoint_solve_kkt!(kkt::K1sAuglagKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_k1s_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_solve_kkt!(kkt::K2rAuglagKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_k2r_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::K1sAuglagKKTSystem{T,VT,MT},
    v::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT, MT}
    nx, nr, ns = kkt.nlp.nx, kkt.nlp.nr, length(kkt.ind_ineq)
    ρk = kkt.ρk
    vx = view(full(v), 1:nx)
    vr = view(full(v), nx+1:nx+nr)
    vs = view(full(v), nx+nr+1:nx+nr+ns)
    vy = view(full(v), nx+nr+ns+1:nx+nr+ns+kkt.m)
    wx = view(full(w), 1:nx)
    wr = view(full(w), nx+1:nx+nr)
    ws = view(full(w), nx+nr+1:nx+nr+ns)
    wy = view(full(w), nx+nr+ns+1:nx+nr+ns+kkt.m)

    wy_ineq = view(wy, kkt.ind_ineq)
    vy_ineq = view(vy, kkt.ind_ineq)

    symul!(wx, kkt.hess_com, vx, alpha, beta)
    mul!(wx, kkt.jt_csc, vy, alpha, one(T))

    ws .= beta .* ws .- alpha .* vy_ineq
    wr .= beta .* wr .+ alpha .* (ρk .* vr .+ vy)

    mul!(wy, kkt.jt_csc', vx, alpha, beta)
    wy_ineq .-= alpha .* vs
    wy .+= alpha .* vr

    _adjoint_kktmul!(
        w, v, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta
    )
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::K2rAuglagKKTSystem{T,VT,MT},
    v::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT, MT}
    nx, nr, ns = kkt.nlp.nx, kkt.nlp.nr, length(kkt.ind_ineq)
    ρk = kkt.ρk
    vx = view(full(v), 1:nx)
    vr = view(full(v), nx+1:nx+nr)
    vs = view(full(v), nx+nr+1:nx+nr+ns)
    vy = view(full(v), nx+nr+ns+1:nx+nr+ns+kkt.m)
    wx = view(full(w), 1:nx)
    wr = view(full(w), nx+1:nx+nr)
    ws = view(full(w), nx+nr+1:nx+nr+ns)
    wy = view(full(w), nx+nr+ns+1:nx+nr+ns+kkt.m)

    yp = kkt.buffer1
    yp[1:nx] .= beta .* wx
    yp[nx+1:nx+ns] .= beta .* ws
    xp = view(kkt.buffer3, 1:nx+ns)
    xp[1:nx] .= vx
    xp[nx+1:nx+ns] .= vs

    symul!(yp, kkt.hess_com, xp, alpha, one(T))
    mul!(yp, kkt.jac_com', vy, alpha, one(T))

    wx .= view(yp, 1:nx)
    ws .= view(yp, nx+1:nx+ns)
    wr .= beta .* wr .+ alpha .* (ρk .* vr .+ vy)

    mul!(wy, kkt.jac_com, xp, alpha, beta)
    wy .+= alpha .* vr

    _adjoint_kktmul!(
        w, v, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta
    )
    return w
end

end # module
