function pack_jacobian!(sens::MadDiffSolver{T}, jcache) where {T}
    cb = sens.solver.cb
    kkt = sens.kkt
    V = jcache.rhs_raw.V
    hess_vals = jcache.hess_raw.V
    jac_vals  = jcache.jac_raw.V
    lvar_vals = jcache.lvar_raw.V
    uvar_vals = jcache.uvar_raw.V
    lcon_vals = jcache.lcon_raw.V
    ucon_vals = jcache.ucon_raw.V
    fill!(V, zero(T))

    if has_hess_param(jcache)
        b = jcache.blk_hess
        pack_hess_param!(view(V, b.seg), hess_vals[b.src], cb.obj_scale[])
    end

    if has_jac_param(jcache)
        b = jcache.blk_jac
        pack_jac_param!(view(V, b.seg), jac_vals[b.src], cb.con_scale[b.row])
    end

    if has_lcon_eq_param(jcache)
        b = jcache.blk_lcon_eq
        pack_lcon_eq_param!(view(V, b.seg), lcon_vals[b.src], cb.con_scale[b.row], sens.is_eq[b.row])
    end

    if has_ucon_eq_param(jcache)
        b = jcache.blk_ucon_eq
        pack_ucon_eq_param!(view(V, b.seg), ucon_vals[b.src], cb.con_scale[b.row], sens.is_eq[b.row])
    end

    if has_lvar_param(jcache)
        b = jcache.blk_lvar
        pack_lvar_param!(view(V, b.seg), lvar_vals[b.src], kkt, b.pos)
    end

    if has_uvar_param(jcache)
        b = jcache.blk_uvar
        pack_uvar_param!(view(V, b.seg), uvar_vals[b.src], kkt, b.pos)
    end

    if has_lcon_slack_param(jcache)
        b = jcache.blk_lcon_slack
        pack_lcon_slack_param!(view(V, b.seg), lcon_vals[b.src], cb.con_scale[b.row], kkt, b.pos)
    end

    if has_ucon_slack_param(jcache)
        b = jcache.blk_ucon_slack
        pack_ucon_slack_param!(view(V, b.seg), ucon_vals[b.src], cb.con_scale[b.row], kkt, b.pos)
    end

    transfer!(jcache.W, jcache.rhs_raw, jcache.rhs_map)
    return nothing
end

function solve_jacobian!(sens::MadDiffSolver{T}, jcache) where {T}
    multi_solve_kkt!(sens.kkt, jcache.W)
    return nothing
end

function unpack_jacobian!(result::JacobianResult, sens::MadDiffSolver, jcache)
    cb = sens.solver.cb
    W = jcache.W

    primal_rows, dual_rows, lb_rows, ub_rows = _jacobian_row_ranges(sens.kkt)

    @views unpack_dx!(result.dx, cb, W[primal_rows, :])
    _set_fixed_sensitivity_from_sparse!(result.dx, cb, jcache)
    @views unpack_y!(result.dy, cb, W[dual_rows, :])
    @views unpack_dzl!(result.dzl, cb, W[lb_rows, :], jcache.dz_work)
    @views unpack_dzu!(result.dzu, cb, W[ub_rows, :], jcache.dz_work)
    return result
end

function compute_jacobian_objective_sensitivity!(result::JacobianResult, sens::MadDiffSolver{T}, jcache) where {T}
    nlp = sens.solver.nlp
    pmeta = nlp.pmeta
    x = jcache.x_nlp

    grad!(nlp, x, jcache.grad_x)
    if pmeta.grad_param_available
        grad_param!(nlp, x, jcache.grad_p)
    else
        fill!(jcache.grad_p, zero(T))
    end
    mul!(result.dobj, transpose(result.dx), jcache.grad_x)
    result.dobj .+= jcache.grad_p
    return nothing
end

function jacobian!(result::JacobianResult, sens::MadDiffSolver{T}) where {T}
    solver = sens.solver
    cb = solver.cb
    nlp = solver.nlp
    pmeta = nlp.pmeta
    jcache = get_jac_cache!(sens)

    unpack_x!(jcache.x_nlp, cb, variable(solver.x))
    unpack_y!(jcache.y_nlp, cb, solver.y)

    x = jcache.x_nlp
    y = jcache.y_nlp

    fill!(jcache.hess_raw.V, zero(T))
    fill!(jcache.jac_raw.V, zero(T))
    fill!(jcache.lvar_raw.V, zero(T))
    fill!(jcache.uvar_raw.V, zero(T))
    fill!(jcache.lcon_raw.V, zero(T))
    fill!(jcache.ucon_raw.V, zero(T))

    pmeta.hess_param_available && hess_param_coord!(nlp, x, y, jcache.hess_raw.V; obj_weight = cb.obj_sign)
    pmeta.jac_param_available  && jac_param_coord!(nlp, x, jcache.jac_raw.V)
    pmeta.lvar_jac_available   && lvar_jac_param_coord!(nlp, jcache.lvar_raw.V)
    pmeta.uvar_jac_available   && uvar_jac_param_coord!(nlp, jcache.uvar_raw.V)
    pmeta.lcon_jac_available   && lcon_jac_param_coord!(nlp, jcache.lcon_raw.V)
    pmeta.ucon_jac_available   && ucon_jac_param_coord!(nlp, jcache.ucon_raw.V)

    pack_jacobian!(sens, jcache)
    solve_jacobian!(sens, jcache)
    unpack_jacobian!(result, sens, jcache)
    compute_jacobian_objective_sensitivity!(result, sens, jcache)

    return result
end
