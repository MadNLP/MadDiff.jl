function _jacobian_transpose_ranges(n_x::Int, n_con::Int)
    r_dx = 1:n_x
    r_dy = (last(r_dx) + 1):(last(r_dx) + n_con)
    r_dzl = (last(r_dy) + 1):(last(r_dy) + n_x)
    r_dzu = (last(r_dzl) + 1):(last(r_dzl) + n_x)
    c_obj = last(r_dzu) + 1
    return r_dx, r_dy, r_dzl, r_dzu, c_obj
end

function _jacobian_transpose_fill_pv!(::AbstractUnreducedKKTSystem, pv_lr::AbstractMatrix, pv_ur::AbstractMatrix, cb, rhs_lb::AbstractMatrix, rhs_ub::AbstractMatrix)
    fill!(pv_lr, zero(eltype(pv_lr)))
    fill!(pv_ur, zero(eltype(pv_ur)))
    pv_lr[cb.ind_lb, :] .= rhs_lb
    pv_ur[cb.ind_ub, :] .= .-rhs_ub
    return nothing
end

function _jacobian_transpose_fill_pv!(kkt, pv_lr::AbstractMatrix, pv_ur::AbstractMatrix, cb, rhs_lb::AbstractMatrix, rhs_ub::AbstractMatrix)
    fill!(pv_lr, zero(eltype(pv_lr)))
    fill!(pv_ur, zero(eltype(pv_ur)))
    pv_lr[cb.ind_lb, :] .= kkt.l_lower .* rhs_lb
    pv_ur[cb.ind_ub, :] .= .-kkt.u_lower .* rhs_ub
    return nothing
end

function pack_jacobian_transpose!(sens::MadDiffSolver{T}, tcache) where {T}
    solver = sens.solver
    cb = solver.cb
    n_x = get_nvar(solver.nlp)
    n_con = get_ncon(solver.nlp)
    n_var_cb = cb.nvar

    r_dx, r_dy, r_dzl, r_dzu, c_obj = _jacobian_transpose_ranges(n_x, n_con)

    W = tcache.W
    fill!(W, zero(T))

    n_primal = length(sens.kkt.pr_diag)
    n_dual = length(sens.kkt.du_diag)
    n_lb = length(sens.kkt.l_diag)
    n_ub = length(sens.kkt.u_diag)

    primal_seed_rows = 1:n_var_cb
    primal_rows = 1:n_primal
    dual_rows = n_primal + 1:n_primal + n_dual
    lb_rows = n_primal + n_dual + 1:n_primal + n_dual + n_lb
    ub_rows = n_primal + n_dual + n_lb + 1:n_primal + n_dual + n_lb + n_ub

    dx_seed = view(tcache.dx_solved, :, r_dx)
    dy_seed = view(tcache.dy_solved, :, r_dy)
    dzl_seed = view(tcache.dzl_solved, :, r_dzl)
    dzu_seed = view(tcache.dzu_solved, :, r_dzu)

    fill!(dx_seed, zero(T))
    fill!(dy_seed, zero(T))
    fill!(dzl_seed, zero(T))
    fill!(dzu_seed, zero(T))

    for i in 1:n_x
        dx_seed[i, i] = one(T)
        dzl_seed[i, i] = one(T)
        dzu_seed[i, i] = one(T)
    end
    for i in 1:n_con
        dy_seed[i, i] = one(T)
    end

    pack_dx!(view(W, primal_seed_rows, r_dx), cb, dx_seed)
    pack_dy!(view(W, dual_rows, r_dy), cb, dy_seed)

    pack_dzl!(view(W, lb_rows, r_dzl), cb, dzl_seed, view(tcache.dz_work, :, r_dzl))
    pack_dzu!(view(W, ub_rows, r_dzu), cb, dzu_seed, view(tcache.dz_work, :, r_dzu))

    grad!(solver.nlp, tcache.x_nlp, tcache.grad_p)
    pack_dx!(view(W, primal_seed_rows, c_obj:c_obj), cb, reshape(tcache.grad_p, :, 1))
    return nothing
end

function solve_jacobian_transpose!(sens::MadDiffSolver, tcache)
    adjoint_multi_solve_kkt_system!(sens.kkt, tcache.W)
    return nothing
end

function unpack_jacobian_transpose!(sens::MadDiffSolver, tcache)
    cb = sens.solver.cb
    n_primal = length(sens.kkt.pr_diag)
    n_dual = length(sens.kkt.du_diag)
    n_lb = length(sens.kkt.l_diag)
    n_ub = length(sens.kkt.u_diag)

    primal_rows = 1:n_primal
    dual_rows = n_primal + 1:n_primal + n_dual
    lb_rows = n_primal + n_dual + 1:n_primal + n_dual + n_lb
    ub_rows = n_primal + n_dual + n_lb + 1:n_primal + n_dual + n_lb + n_ub

    unpack_dx!(tcache.dx_solved, cb, view(tcache.W, primal_rows, :))
    unpack_y!(tcache.dy_solved, cb, view(tcache.W, dual_rows, :))
    unpack_dzl!(tcache.dzl_solved, cb, view(tcache.W, lb_rows, :), tcache.dz_work)
    unpack_dzu!(tcache.dzu_solved, cb, view(tcache.W, ub_rows, :), tcache.dz_work)
    return nothing
end

function pullback_jacobian_transpose!(result::JacobianTransposeResult, sens::MadDiffSolver{T}, tcache) where {T}
    solver = sens.solver
    nlp = solver.nlp
    cb = solver.cb
    n_x = get_nvar(nlp)
    n_con = get_ncon(nlp)
    σ = cb.obj_sign
    obj_scale = cb.obj_scale[]
    σ_scaled = σ * obj_scale

    r_dx, r_dy, r_dzl, r_dzu, c_obj = _jacobian_transpose_ranges(n_x, n_con)

    unpack_x!(tcache.x_nlp, cb, variable(solver.x))
    unpack_y!(tcache.y_nlp, cb, solver.y)
    tcache.y_nlp .*= σ_scaled

    hess_param!(nlp, tcache.x_nlp, tcache.y_nlp, tcache.hess_nlp; obj_weight = σ_scaled)
    jac_param!(nlp, tcache.x_nlp, tcache.jac_nlp)
    lvar_jac_param!(nlp, tcache.dlvar_nlp)
    uvar_jac_param!(nlp, tcache.duvar_nlp)
    lcon_jac_param!(nlp, tcache.dlcon_nlp)
    ucon_jac_param!(nlp, tcache.ducon_nlp)
    grad_param!(nlp, tcache.x_nlp, tcache.grad_p)

    tcache.grad_all .= transpose(tcache.hess_nlp) * tcache.dx_solved

    tcache.dy_nlp .= tcache.dy_solved .* σ_scaled
    tcache.grad_all .+= transpose(tcache.jac_nlp) * tcache.dy_nlp
    view(tcache.grad_all, :, c_obj) .-= tcache.grad_p

    n_primal = length(sens.kkt.pr_diag)
    n_dual = length(sens.kkt.du_diag)
    n_lb = length(sens.kkt.l_diag)
    lb_rows = n_primal + n_dual + 1:n_primal + n_dual + n_lb
    ub_rows = n_primal + n_dual + n_lb + 1:n_primal + n_dual + n_lb + length(sens.kkt.u_diag)

    _jacobian_transpose_fill_pv!(
        sens.kkt,
        tcache.pv_lr,
        tcache.pv_ur,
        cb,
        view(tcache.W, lb_rows, :),
        view(tcache.W, ub_rows, :),
    )

    unpack_dx!(tcache.x_lr, cb, view(tcache.pv_lr, 1:cb.nvar, :))
    tcache.grad_all .-= transpose(tcache.dlvar_nlp) * tcache.x_lr
    
    unpack_dx!(tcache.x_ur, cb, view(tcache.pv_ur, 1:cb.nvar, :))
    tcache.grad_all .+= transpose(tcache.duvar_nlp) * tcache.x_ur

    unpack_slack!(tcache.y_lr, cb, tcache.pv_lr, sens.is_eq, view(tcache.W, n_primal + 1:n_primal + n_dual, :))
    tcache.grad_all .-= transpose(tcache.dlcon_nlp) * tcache.y_lr
    
    unpack_slack!(tcache.y_ur, cb, tcache.pv_ur, sens.is_eq, view(tcache.W, n_primal + 1:n_primal + n_dual, :))
    tcache.grad_all .+= transpose(tcache.ducon_nlp) * tcache.y_ur

    tcache.grad_all .*= -one(T)

    result.dx .= view(tcache.grad_all, :, r_dx)
    result.dy .= view(tcache.grad_all, :, r_dy)
    result.dzl .= view(tcache.grad_all, :, r_dzl)
    result.dzu .= view(tcache.grad_all, :, r_dzu)
    result.dobj .= view(tcache.grad_all, :, c_obj)
    return result
end

function jacobian_transpose!(result::JacobianTransposeResult, sens::MadDiffSolver{T}) where {T}
    tcache = get_jact_cache!(sens)
    cb = sens.solver.cb
    unpack_x!(tcache.x_nlp, cb, variable(sens.solver.x))

    pack_jacobian_transpose!(sens, tcache)
    solve_jacobian_transpose!(sens, tcache)
    unpack_jacobian_transpose!(sens, tcache)
    pullback_jacobian_transpose!(result, sens, tcache)
    return result
end
