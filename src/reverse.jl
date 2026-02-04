function reverse_differentiate!(
    result::ReverseResult, sens::MadDiffSolver{T};
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing,
) where {T}
    pack_vjp!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu)
    solve_vjp!(sens)
    unpack_vjp!(result, sens)

    solver = sens.solver
    nlp = solver.nlp
    cb = solver.cb
    cache = get_reverse_cache!(sens)
    obj_scale = cb.obj_scale[]
    σ = cb.obj_sign

    unpack_x!(cache.x_nlp, cb, variable(solver.x))
    unpack_y!(cache.y_nlp, cb, solver.y)
    x = cache.x_nlp
    y = cache.y_nlp
    dx = result.dx
    dy = cache.dy_scaled

    grad_p = result.grad_p
    fill!(grad_p, zero(T))

    y .*= obj_scale
    ParametricNLPModels.hptprod!(nlp, x, y, dx, grad_p; obj_weight = σ * obj_scale)
    
    dy .= result.dy .* (cb.obj_sign * obj_scale)
    ParametricNLPModels.jptprod!(nlp, x, dy, cache.tmp_p)
    grad_p .+= cache.tmp_p

    _add_bound_grad_p!(grad_p, sens, cache)
    grad_p .*= -one(T)

    return result
end

function pack_vjp!(
    sens::MadDiffSolver{T};
    dL_dx = nothing,
    dL_dy = nothing,
    dL_dzl = nothing,
    dL_dzu = nothing,
) where {T}
    all(isnothing, (dL_dx, dL_dy, dL_dzl, dL_dzu)) &&
        throw(ArgumentError("At least one of dL_dx, dL_dy, dL_dzl, dL_dzu must be provided"))

    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    isnothing(dL_dx) || @lencheck n_x dL_dx
    isnothing(dL_dy) || @lencheck n_con dL_dy
    isnothing(dL_dzl) || @lencheck n_x dL_dzl
    isnothing(dL_dzu) || @lencheck n_x dL_dzu

    cache = get_reverse_cache!(sens)
    cb = sens.solver.cb

    fill!(cache.dL_dx, zero(T))
    fill!(cache.dL_dy, zero(T))
    fill!(cache.dL_dzl, zero(T))
    fill!(cache.dL_dzu, zero(T))
    fill!(full(cache.dzl_full), zero(T))
    fill!(full(cache.dzu_full), zero(T))

    isnothing(dL_dx) || pack_dx!(cache.dL_dx, cb, dL_dx)
    isnothing(dL_dy) || pack_dy!(cache.dL_dy, cb, dL_dy)
    isnothing(dL_dzl) || pack_dzl!(cache.dL_dzl, cb, dL_dzl, cache.dzl_full)
    isnothing(dL_dzu) || pack_dzu!(cache.dL_dzu, cb, dL_dzu, cache.dzu_full)
    return nothing
end

function solve_vjp!(sens::MadDiffSolver{T}) where {T}
    cache = get_reverse_cache!(sens)
    w = cache.kkt_rhs
    n_x = length(cache.dL_dx)

    fill!(full(w), zero(T))
    primal(w)[1:n_x] .= cache.dL_dx
    dual(w) .= cache.dL_dy
    dual_lb(w) .= cache.dL_dzl
    dual_ub(w) .= cache.dL_dzu

    _adjoint_solve_with_refine!(sens, w, cache)
    return nothing
end

function unpack_vjp!(result::ReverseResult, sens::MadDiffSolver)
    cache = get_reverse_cache!(sens)
    cb = sens.solver.cb

    unpack_dx!(result.dx, cb, primal(cache.kkt_rhs))
    unpack_y!(result.dy, cb, dual(cache.kkt_rhs))
    unpack_dzl!(result.dzl, cb, dual_lb(cache.kkt_rhs), cache.dzl_full)
    unpack_dzu!(result.dzu, cb, dual_ub(cache.kkt_rhs), cache.dzu_full)

    return result
end

function vjp_bound_rhs!(kkt, dlvar_pv, duvar_pv, w)
    dlvar_pv.values_lr .= kkt.l_lower .* dual_lb(w)
    duvar_pv.values_ur .= .-kkt.u_lower .* dual_ub(w)
    return nothing
end
function vjp_bound_rhs!(::AbstractUnreducedKKTSystem, dlvar_pv, duvar_pv, w)
    dlvar_pv.values_lr .= dual_lb(w)
    duvar_pv.values_ur .= .-dual_ub(w)
    return nothing
end

function _add_bound_grad_p!(grad_p::AbstractVector{T}, sens::MadDiffSolver{T}, cache) where {T}
    solver = sens.solver
    nlp = solver.nlp
    cb = solver.cb
    kkt = sens.kkt

    pv_lb = cache.dzl_full
    pv_ub = cache.dzu_full

    fill!(full(pv_lb), zero(T))
    fill!(full(pv_ub), zero(T))
    vjp_bound_rhs!(kkt, pv_lb, pv_ub, cache.kkt_rhs)

    tmp = cache.tmp_p
    work_x = cache.x_nlp
    work_con = cache.y_nlp

    fill!(work_x, zero(T))
    unpack_dx!(work_x, cb, variable(pv_lb))
    ParametricNLPModels.lvar_jptprod!(nlp, work_x, tmp)
    grad_p .+= tmp

    fill!(work_x, zero(T))
    unpack_dx!(work_x, cb, variable(pv_ub))
    ParametricNLPModels.uvar_jptprod!(nlp, work_x, tmp)
    grad_p .+= tmp

    fill!(work_con, zero(T))
    work_con .+= sens.is_eq .* dual(cache.kkt_rhs) ./ 2
    work_con[cb.ind_ineq] .+= slack(pv_lb) .* cb.con_scale[cb.ind_ineq]
    work_con .*= cb.con_scale
    ParametricNLPModels.lcon_jptprod!(nlp, work_con, tmp)
    grad_p .+= tmp

    fill!(work_con, zero(T))
    work_con .+= sens.is_eq .* dual(cache.kkt_rhs) ./ 2
    work_con[cb.ind_ineq] .+= slack(pv_ub) .* cb.con_scale[cb.ind_ineq]
    work_con .*= cb.con_scale
    ParametricNLPModels.ucon_jptprod!(nlp, work_con, tmp)
    grad_p .+= tmp

    return nothing
end