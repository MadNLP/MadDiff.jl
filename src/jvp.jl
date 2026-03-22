function jacobian_vector_product!(
    result::JVPResult, sens::MadDiffSolver{T}, Δp::AbstractVector,
) where {T}
    solver = sens.solver
    cb = solver.cb
    nlp = solver.nlp
    cache = get_jvp_cache!(sens)

    unpack_x!(cache.x_nlp, cb, variable(solver.x))
    unpack_y!(cache.y_nlp, cb, solver.y)

    x = cache.x_nlp
    y = cache.y_nlp

    fill!(cache.hpv_nlp, zero(T))
    fill!(cache.jpv_nlp, zero(T))
    fill!(cache.dlvar_nlp, zero(T))
    fill!(cache.duvar_nlp, zero(T))
    fill!(cache.dlcon_nlp, zero(T))
    fill!(cache.ducon_nlp, zero(T))

    has_hess_param(nlp) && hpprod!(nlp, x, y, Δp, cache.hpv_nlp; obj_weight = cb.obj_sign)
    has_jac_param(nlp)  && jpprod!(nlp, x, Δp, cache.jpv_nlp)
    has_lvar_param(nlp) && lvar_jpprod!(nlp, Δp, cache.dlvar_nlp)
    has_uvar_param(nlp) && uvar_jpprod!(nlp, Δp, cache.duvar_nlp)
    has_lcon_param(nlp) && lcon_jpprod!(nlp, Δp, cache.dlcon_nlp)
    has_ucon_param(nlp) && ucon_jpprod!(nlp, Δp, cache.ducon_nlp)

    pack_jvp!(sens, cache)
    solve_jvp!(sens)
    unpack_jvp!(result, sens, cache)
    result.dobj[] = zero(T)

    return result
end

function compute_objective_sensitivity!(
    result::JVPResult, sens::MadDiffSolver{T}, Δp::AbstractVector,
) where {T}
    cache = get_jvp_cache!(sens)
    return _compute_objective_sensitivity!(result, sens, cache, Δp)
end

function _compute_objective_sensitivity!(
    result::JVPResult, sens::MadDiffSolver{T}, cache, Δp::AbstractVector,
) where {T}
    solver = sens.solver
    nlp = solver.nlp
    x = cache.x_nlp

    fill!(cache.grad_x, zero(T))
    fill!(cache.grad_p, zero(T))

    grad!(nlp, x, cache.grad_x)
    has_grad_param(nlp) && grad_param!(nlp, x, cache.grad_p)

    result.dobj[] = dot(cache.grad_x, result.dx) + dot(cache.grad_p, Δp)

    return nothing
end

function pack_jvp!(sens::MadDiffSolver{T}, cache) where {T}
    cb = sens.solver.cb

    fill!(cache.d2L_dxdp, zero(T))
    fill!(cache.dg_dp, zero(T))
    fill!(cache.dlcon_dp, zero(T))
    fill!(cache.ducon_dp, zero(T))
    fill!(full(cache.dlvar_dp), zero(T))
    fill!(full(cache.duvar_dp), zero(T))

    pack_hess!(cache.d2L_dxdp, cb, cache.hpv_nlp)
    pack_cons!(cache.dg_dp, cb, cache.jpv_nlp)
    pack_cons!(cache.dlcon_dp, cb, cache.dlcon_nlp)
    pack_cons!(cache.ducon_dp, cb, cache.ducon_nlp)
    pack_dx!(variable(cache.dlvar_dp), cb, cache.dlvar_nlp)
    pack_dx!(variable(cache.duvar_dp), cb, cache.duvar_nlp)
    pack_slack!(slack(cache.dlvar_dp), cb, cache.dlcon_nlp)
    pack_slack!(slack(cache.duvar_dp), cb, cache.ducon_nlp)
    return nothing
end

function jvp_set_bound_rhs!(kkt, w, dlvar_dp, duvar_dp)
    dual_lb(w) .= kkt.l_lower .* dlvar_dp.values_lr
    dual_ub(w) .= .-kkt.u_lower .* duvar_dp.values_ur
    return nothing
end
function jvp_set_bound_rhs!(::AbstractUnreducedKKTSystem, w, dlvar_dp, duvar_dp)
    dual_lb(w) .= dlvar_dp.values_lr
    dual_ub(w) .= .-duvar_dp.values_ur
    return nothing
end

function solve_jvp!(sens::MadDiffSolver{T}) where {T}
    cache = get_jvp_cache!(sens)
    w = cache.kkt_rhs
    assemble_jvp_rhs!(sens, w, cache)

    _solve_with_refine!(sens, w, cache)
    return nothing
end

function assemble_jvp_rhs!(sens::MadDiffSolver{T}, w, cache) where {T}
    n_x = length(cache.d2L_dxdp)

    fill!(full(w), zero(T))
    primal(w)[1:n_x] .= .-cache.d2L_dxdp
    dual(w) .= .-cache.dg_dp .+ sens.is_eq .* (cache.dlcon_dp .+ cache.ducon_dp) ./ 2
    jvp_set_bound_rhs!(sens.kkt, w, cache.dlvar_dp, cache.duvar_dp)
    return w
end

function unpack_jvp!(result::JVPResult, sens::MadDiffSolver, cache)
    cb = sens.solver.cb
    w = cache.kkt_rhs

    unpack_dx!(result.dx, cb, primal(w))
    set_fixed_sensitivity!(result.dx, cb, cache.dlvar_nlp, cache.duvar_nlp)
    unpack_y!(result.dy, cb, dual(w))
    unpack_dzl!(result.dzl, cb, dual_lb(w), cache.dlvar_dp)
    unpack_dzu!(result.dzu, cb, dual_ub(w), cache.duvar_dp)

    return result
end
