function _adjoint_dense_condensed_solve!(kkt::DenseCondensedKKTSystem{T}, w::AbstractKKTVector) where T
    n = num_variables(kkt)
    n_eq, ns = kkt.n_eq, kkt.n_ineq

    # Decompose rhs
    wx = view(full(w), 1:n)
    ws = view(full(w), n+1:n+ns)
    wy = view(full(w), kkt.ind_eq_shifted)
    wz = view(full(w), kkt.ind_ineq_shifted)

    x = kkt.pd_buffer
    xx = view(x, 1:n)
    xy = view(x, n+1:n+n_eq)

    Σs = get_slack_regularization(kkt)
    buf = kkt.buffer
    buf_ineq = view(buf, kkt.ind_ineq)

    fill!(x, zero(T))

    # g_z += Σs⁻¹ g_s, g_s = Σs⁻¹ g_s
    wz .+= ws ./ Σs
    ws ./= Σs

    # Save g_z and build g_x + Jᵀ (D g_z)
    fill!(buf, zero(T))
    buf_ineq .= wz
    wz .*= kkt.diag_buffer

    xy .= wy
    wy .= 0
    mul!(wx, kkt.jac', dual(w), one(T), one(T))

    # Solve K_condᵀ g = [g_x; g_y]
    xx .= wx
    solve_linear_system!(kkt.linear_solver, x)

    # g_r_x, g_r_y
    wx .= xx
    wy .+= xy

    # g_B = -g_z + J g_r_x
    buf_ineq .*= .-one(T)
    mul!(buf, kkt.jac, xx, one(T), one(T))

    # g_r_z, g_r_s
    buf_ineq .*= kkt.diag_buffer
    wz .= buf_ineq
    ws .+= buf_ineq ./ Σs
    return
end

function adjoint_solve_kkt!(kkt::DenseCondensedKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_dense_condensed_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end
