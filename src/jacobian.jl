function pack_jacobian!(sens::MadDiffSolver{T}, jcache) where {T}
    solver = sens.solver
    cb = solver.cb

    n_var_cb = size(jcache.d2L_dxdp, 1)
    n_ineq = length(cb.ind_ineq)

    pack_hess!(jcache.d2L_dxdp, cb, jcache.hess_nlp)
    pack_cons!(jcache.dg_dp, cb, jcache.jac_nlp)
    pack_cons!(jcache.dlcon_dp, cb, jcache.dlcon_nlp)
    pack_cons!(jcache.ducon_dp, cb, jcache.ducon_nlp)
    @views pack_dx!(jcache.dlvar_dp[1:n_var_cb, :], cb, jcache.dlvar_nlp)
    @views pack_dx!(jcache.duvar_dp[1:n_var_cb, :], cb, jcache.duvar_nlp)
    @views pack_slack!(jcache.dlvar_dp[n_var_cb + 1:n_var_cb + n_ineq, :], cb, jcache.dlcon_dp)
    @views pack_slack!(jcache.duvar_dp[n_var_cb + 1:n_var_cb + n_ineq, :], cb, jcache.ducon_dp)
    return nothing
end

function jacobian_set_bound_rhs!(kkt, W::AbstractMatrix, ind_lb, ind_ub, dlvar_dp_lr::AbstractMatrix, duvar_dp_ur::AbstractMatrix)
    @views W[ind_lb, :] .= kkt.l_lower .* dlvar_dp_lr
    @views W[ind_ub, :] .= .-kkt.u_lower .* duvar_dp_ur
    return nothing
end
function jacobian_set_bound_rhs!(::AbstractUnreducedKKTSystem, W::AbstractMatrix, ind_lb, ind_ub, dlvar_dp_lr::AbstractMatrix, duvar_dp_ur::AbstractMatrix)
    @views W[ind_lb, :] .= dlvar_dp_lr
    @views W[ind_ub, :] .= .-duvar_dp_ur
    return nothing
end

function solve_jacobian!(sens::MadDiffSolver{T}, jcache) where {T}
    W = jcache.W
    assemble_jacobian_rhs!(sens, W, jcache)
    multi_solve_kkt!(sens.kkt, W)
    return nothing
end

function assemble_jacobian_rhs!(sens::MadDiffSolver{T}, W, jcache) where {T}
    cb = sens.solver.cb
    n_var_cb = size(jcache.d2L_dxdp, 1)
    n_primal = length(sens.kkt.pr_diag)
    n_dual = length(sens.kkt.du_diag)
    n_lb = length(sens.kkt.l_diag)
    n_ub = length(sens.kkt.u_diag)

    dual_rows = n_primal + 1:n_primal + n_dual
    lb_rows = n_primal + n_dual + 1:n_primal + n_dual + n_lb
    ub_rows = n_primal + n_dual + n_lb + 1:n_primal + n_dual + n_lb + n_ub

    fill!(W, zero(T))
    @views W[1:n_var_cb, :] .= .-jcache.d2L_dxdp
    @views W[dual_rows, :] .= .-jcache.dg_dp .+ sens.is_eq .* (jcache.dlcon_dp .+ jcache.ducon_dp) ./ 2
    @views jacobian_set_bound_rhs!(
        sens.kkt, W, lb_rows, ub_rows,
        jcache.dlvar_dp[cb.ind_lb, :], jcache.duvar_dp[cb.ind_ub, :],
    )
    return W
end

function unpack_jacobian!(result::JacobianResult, sens::MadDiffSolver, jcache)
    cb = sens.solver.cb
    W = jcache.W

    n_primal = length(sens.kkt.pr_diag)
    n_dual = length(sens.kkt.du_diag)
    n_lb = length(sens.kkt.l_diag)
    n_ub = length(sens.kkt.u_diag)

    primal_rows = 1:n_primal
    dual_rows = n_primal + 1:n_primal + n_dual
    lb_rows = n_primal + n_dual + 1:n_primal + n_dual + n_lb
    ub_rows = n_primal + n_dual + n_lb + 1:n_primal + n_dual + n_lb + n_ub

    @views unpack_dx!(result.dx, cb, W[primal_rows, :])
    set_fixed_sensitivity!(result.dx, cb, jcache.dlvar_nlp, jcache.duvar_nlp)
    @views unpack_y!(result.dy, cb, W[dual_rows, :])
    @views unpack_dzl!(result.dzl, cb, W[lb_rows, :], jcache.dz_work)
    @views unpack_dzu!(result.dzu, cb, W[ub_rows, :], jcache.dz_work)
    return result
end

function compute_jacobian_objective_sensitivity!(result::JacobianResult, sens::MadDiffSolver{T}, jcache) where {T}
    nlp = sens.solver.nlp
    x = jcache.x_nlp
    n_p = sens.n_p

    grad!(nlp, x, jcache.grad_x)
    grad_param!(nlp, x, jcache.grad_p)
    for j in 1:n_p
        result.dobj[j] = dot(jcache.grad_x, view(result.dx, :, j)) + jcache.grad_p[j]
    end
    return nothing
end

function jacobian!(result::JacobianResult, sens::MadDiffSolver{T}) where {T}
    solver = sens.solver
    cb = solver.cb
    nlp = solver.nlp
    jcache = get_jac_cache!(sens)
    x = jcache.x_nlp
    y = jcache.y_nlp

    unpack_x!(x, cb, variable(solver.x))
    unpack_y!(y, cb, solver.y)

    hess_param!(nlp, x, y, jcache.hess_nlp; obj_weight = cb.obj_sign)
    jac_param!(nlp, x, jcache.jac_nlp)
    lvar_jac_param!(nlp, jcache.dlvar_nlp)
    uvar_jac_param!(nlp, jcache.duvar_nlp)
    lcon_jac_param!(nlp, jcache.dlcon_nlp)
    ucon_jac_param!(nlp, jcache.ducon_nlp)

    pack_jacobian!(sens, jcache)
    solve_jacobian!(sens, jcache)
    unpack_jacobian!(result, sens, jcache)
    compute_jacobian_objective_sensitivity!(result, sens, jcache)
    return result
end
