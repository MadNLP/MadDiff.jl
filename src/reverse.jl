function reverse_differentiate!(
    result::ReverseResult, sens::MadDiffSolver{T};
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing,
) where {T}
    pack_vjp!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu)
    solve_vjp!(sens)
    unpack_vjp!(result, sens)
    return vjp_pullback!(result, sens)
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
    w = cache.kkt_rhs

    unpack_dx!(result.dx, cb, primal(w))
    unpack_y!(result.dy, cb, dual(w))
    unpack_dzl!(result.dzl, cb, dual_lb(w), cache.dzl_full)
    unpack_dzu!(result.dzu, cb, dual_ub(w), cache.dzu_full)

    return result
end

function vjp_fill_pv!(kkt, dlvar_pv, duvar_pv, w)
    fill!(full(dlvar_pv), zero(eltype(full(dlvar_pv))))
    fill!(full(duvar_pv), zero(eltype(full(duvar_pv))))
    dlvar_pv.values_lr .= kkt.l_lower .* dual_lb(w)
    duvar_pv.values_ur .= .-kkt.u_lower .* dual_ub(w)
    return nothing
end
function vjp_fill_pv!(::AbstractUnreducedKKTSystem, dlvar_pv, duvar_pv, w)
    fill!(full(dlvar_pv), zero(eltype(full(dlvar_pv))))
    fill!(full(duvar_pv), zero(eltype(full(duvar_pv))))
    dlvar_pv.values_lr .= dual_lb(w)
    duvar_pv.values_ur .= .-dual_ub(w)
    return nothing
end

function vjp_pullback!(result::ReverseResult, sens::MadDiffSolver{T}) where {T}
    solver = sens.solver
    nlp = solver.nlp
    cb = solver.cb
    cache = get_reverse_cache!(sens)
    w = cache.kkt_rhs
    x = cache.x_nlp
    y = cache.y_nlp
    dx = result.dx
    dy = cache.dy_scaled
    dzl = cache.dzl_full
    dzu = cache.dzu_full
    tmp = cache.tmp_p
    obj_scale = cb.obj_scale[]
    σ = cb.obj_sign
    σ_scaled = σ * obj_scale

    unpack_x!(x, cb, variable(solver.x))
    unpack_y!(y, cb, solver.y)

    grad_p = result.grad_p
    fill!(grad_p, zero(T))

    y .*= σ_scaled
    ParametricNLPModels.hptprod!(nlp, x, y, dx, grad_p; obj_weight = σ_scaled)
    
    dy .= result.dy .* σ_scaled
    ParametricNLPModels.jptprod!(nlp, x, dy, tmp)
    grad_p .+= tmp

    vjp_fill_pv!(sens.kkt, dzl, dzu, w)

    unpack_dx!(x, cb, variable(dzl))
    ParametricNLPModels.lvar_jptprod!(nlp, x, tmp)
    grad_p .-= tmp

    unpack_dx!(x, cb, variable(dzu))
    ParametricNLPModels.uvar_jptprod!(nlp, x, tmp)
    grad_p .+= tmp

    unpack_slack!(y, cb, dzl, sens, dual(w))
    ParametricNLPModels.lcon_jptprod!(nlp, y, tmp)
    grad_p .-= tmp

    unpack_slack!(y, cb, dzu, sens, dual(w))
    ParametricNLPModels.ucon_jptprod!(nlp, y, tmp)
    grad_p .+= tmp

    grad_p .*= -one(T)

    return result
end