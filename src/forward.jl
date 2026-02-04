function forward_differentiate!(
    result::ForwardResult, sens::MadDiffSolver{T}, Δp::AbstractVector,
) where {T}
    solver = sens.solver
    cb = solver.cb
    nlp = solver.nlp
    cache = get_forward_cache!(sens)

    unpack_x!(cache.x_nlp, cb, variable(solver.x))
    unpack_y!(cache.y_nlp, cb, solver.y)
    
    x = cache.x_nlp
    y = cache.y_nlp
    d2L_dxdp = cache.hpv_nlp
    dg_dp = cache.jpv_nlp
    dlvar_dp = cache.dlvar_nlp
    duvar_dp = cache.duvar_nlp
    dlcon_dp = cache.dlcon_nlp
    ducon_dp = cache.ducon_nlp

    ParametricNLPModels.hpprod!(nlp, x, y, Δp, d2L_dxdp; obj_weight = cb.obj_sign)
    ParametricNLPModels.jpprod!(nlp, x, Δp, dg_dp)
    ParametricNLPModels.lvar_jpprod!(nlp, Δp, dlvar_dp)
    ParametricNLPModels.uvar_jpprod!(nlp, Δp, duvar_dp)
    ParametricNLPModels.lcon_jpprod!(nlp, Δp, dlcon_dp)
    ParametricNLPModels.ucon_jpprod!(nlp, Δp, ducon_dp)

    pack_jvp!(sens; d2L_dxdp, dg_dp, dlvar_dp, duvar_dp, dlcon_dp, ducon_dp)
    solve_jvp!(sens)
    unpack_jvp!(result, sens; dlvar_dp, duvar_dp)
    return result
end

function pack_jvp!(
    sens::MadDiffSolver{T};
    d2L_dxdp = nothing,
    dg_dp = nothing,
    dlvar_dp = nothing,
    duvar_dp = nothing,
    dlcon_dp = nothing,
    ducon_dp = nothing,
) where {T}
    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    !isnothing(d2L_dxdp) && @lencheck n_x d2L_dxdp
    !isnothing(dg_dp) && @lencheck n_con dg_dp
    !isnothing(dlvar_dp) && @lencheck n_x dlvar_dp
    !isnothing(duvar_dp) && @lencheck n_x duvar_dp
    !isnothing(dlcon_dp) && @lencheck n_con dlcon_dp
    !isnothing(ducon_dp) && @lencheck n_con ducon_dp

    cache = get_forward_cache!(sens)
    cb = sens.solver.cb

    fill!(full(cache.dlvar_dp), zero(T))
    fill!(full(cache.duvar_dp), zero(T))
    fill!(cache.d2L_dxdp, zero(T))
    fill!(cache.dg_dp, zero(T))
    fill!(cache.dlcon_dp, zero(T))
    fill!(cache.ducon_dp, zero(T))

    !isnothing(d2L_dxdp) && pack_hess(cache.d2L_dxdp, cb, d2L_dxdp)
    !isnothing(dg_dp) && pack_cons!(cache.dg_dp, cb, dg_dp)
    !isnothing(dlcon_dp) && pack_cons!(cache.dlcon_dp, cb, dlcon_dp)
    !isnothing(ducon_dp) && pack_cons!(cache.ducon_dp, cb, ducon_dp)
    !isnothing(dlvar_dp) && pack_dx!(variable(cache.dlvar_dp), cb, dlvar_dp)
    !isnothing(duvar_dp) && pack_dx!(variable(cache.duvar_dp), cb, duvar_dp)
    !isnothing(dlcon_dp) && pack_slack!(slack(cache.dlvar_dp), cb, cache.dlcon_dp)
    !isnothing(ducon_dp) && pack_slack!(slack(cache.duvar_dp), cb, cache.ducon_dp)
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
    cache = get_forward_cache!(sens)
    kkt = sens.kkt
    w = cache.kkt_rhs
    n_x = length(cache.d2L_dxdp)

    fill!(full(w), zero(T))
    primal(w)[1:n_x] .= .-cache.d2L_dxdp
    dual(w) .= .-cache.dg_dp .+ sens.is_eq .* (cache.dlcon_dp .+ cache.ducon_dp) ./ 2
    jvp_set_bound_rhs!(kkt, w, cache.dlvar_dp, cache.duvar_dp)

    _solve_with_refine!(sens, w, cache)
    return nothing
end

function unpack_jvp!(result::ForwardResult, sens::MadDiffSolver; dlvar_dp=nothing, duvar_dp=nothing)
    cache = get_forward_cache!(sens)
    cb = sens.solver.cb

    unpack_dx!(result.dx, cb, primal(cache.kkt_rhs))
    set_fixed_sensitivity!(result.dx, cb, dlvar_dp, duvar_dp)
    unpack_y!(result.dy, cb, dual(cache.kkt_rhs))
    unpack_dzl!(result.dzl, cb, dual_lb(cache.kkt_rhs), cache.dlvar_dp)
    unpack_dzu!(result.dzu, cb, dual_ub(cache.kkt_rhs), cache.duvar_dp)

    return result
end
