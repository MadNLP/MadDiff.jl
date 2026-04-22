# ============================================================================
# Vector–Jacobian product — reverse implicit differentiation.
#
# Given cotangents (dL_dx, dL_dy, dL_dzl, dL_dzu) at the optimum, solve the
# adjoint KKT system and contract the parametric operators to obtain
# `grad_p = ∂L/∂p`. At least one seed must be non-`nothing`.
# ============================================================================

function vector_jacobian_product!(
    result::VJPResult, sens::MadDiffSolver{T};
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing,
    dobj  = nothing,
) where {T}
    # Promote host-typed seeds to the solver's device (no-op when matching).
    proto  = _solver_proto(sens)
    dL_dx  = dL_dx  === nothing ? nothing : _adapt_device(proto, dL_dx)
    dL_dy  = dL_dy  === nothing ? nothing : _adapt_device(proto, dL_dy)
    dL_dzl = dL_dzl === nothing ? nothing : _adapt_device(proto, dL_dzl)
    dL_dzu = dL_dzu === nothing ? nothing : _adapt_device(proto, dL_dzu)

    _pack_vjp!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
    _solve_vjp!(sens)
    _unpack_vjp!(result, sens)
    _vjp_pullback!(result, sens; dobj)
    return result
end

# ---------- stages ----------

function _pack_vjp!(
    sens::MadDiffSolver{T};
    dL_dx, dL_dy, dL_dzl, dL_dzu, dobj,
) where {T}
    all(isnothing, (dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)) &&
        throw(ArgumentError(
            "vector_jacobian_product!: at least one of " *
            "`dL_dx`, `dL_dy`, `dL_dzl`, `dL_dzu`, `dobj` must be provided."))

    nlp   = sens.solver.nlp
    n_x   = get_nvar(nlp)
    n_con = get_ncon(nlp)
    isnothing(dL_dx)  || @lencheck n_x   dL_dx
    isnothing(dL_dy)  || @lencheck n_con dL_dy
    isnothing(dL_dzl) || @lencheck n_x   dL_dzl
    isnothing(dL_dzu) || @lencheck n_x   dL_dzu

    cache = get_vjp_cache!(sens)
    cb    = sens.solver.cb

    for buf in (cache.dL_dx, cache.dL_dy, cache.dL_dzl, cache.dL_dzu)
        fill!(buf, zero(T))
    end
    fill!(full(cache.dzl_full), zero(T))
    fill!(full(cache.dzu_full), zero(T))

    isnothing(dL_dx)  || pack_dx!( cache.dL_dx,  cb, dL_dx)
    isnothing(dL_dy)  || pack_dy!( cache.dL_dy,  cb, dL_dy)
    isnothing(dL_dzl) || pack_dzl!(cache.dL_dzl, cb, dL_dzl, cache.dzl_full)
    isnothing(dL_dzu) || pack_dzu!(cache.dL_dzu, cb, dL_dzu, cache.dzu_full)

    if !isnothing(dobj)
        _eval_grad_f_wrapper!(cb, variable(sens.solver.x), cache.grad_x)
        axpy!(dobj, cache.grad_x, cache.dL_dx)
    end
    return nothing
end

function _solve_vjp!(sens::MadDiffSolver{T}) where {T}
    cache = get_vjp_cache!(sens)
    w     = cache.kkt_rhs
    n_x   = length(cache.dL_dx)

    fill!(full(w), zero(T))
    primal(w)[1:n_x] .= cache.dL_dx
    dual(w)          .= cache.dL_dy
    dual_lb(w)       .= cache.dL_dzl
    dual_ub(w)       .= cache.dL_dzu

    _kkt_solve_with_refine!(sens, w, cache, adjoint_solve_refine!)
    return nothing
end

function _unpack_vjp!(result::VJPResult, sens::MadDiffSolver)
    cache = get_vjp_cache!(sens)
    cb    = sens.solver.cb
    w     = cache.kkt_rhs

    unpack_dx!(result.dx, cb, primal(w))
    unpack_y!(result.dy, cb, dual(w))
    unpack_dzl!(result.dzl, cb, dual_lb(w), cache.dzl_full)
    unpack_dzu!(result.dzu, cb, dual_ub(w), cache.dzu_full)
    return result
end

# ---------- parameter-gradient pullback ----------

function _vjp_pullback!(result::VJPResult, sens::MadDiffSolver{T}; dobj) where {T}
    solver    = sens.solver
    nlp       = solver.nlp
    cb        = solver.cb
    cache     = get_vjp_cache!(sens)
    w         = cache.kkt_rhs
    x, y      = cache.x_nlp, cache.y_nlp
    dx, dy    = result.dx, cache.dy_scaled
    pvl, pvu  = cache.dzl_full, cache.dzu_full
    tmp       = cache.tmp_p
    σ_scaled  = cb.obj_sign * cb.obj_scale[]
    grad_p    = fill!(result.grad_p, zero(T))

    unpack_x!(x, cb, variable(solver.x))
    if has_hess_param(nlp)
        unpack_y!(y, cb, solver.y)
        y .*= σ_scaled
        hptprod!(nlp, x, y, dx, grad_p; obj_weight = σ_scaled)
    end

    if !isnothing(dobj) && has_grad_param(nlp)
        grad_param!(nlp, x, tmp)
        axpy!(-dobj, tmp, grad_p)
    end

    if has_jac_param(nlp)
        dy .= result.dy .* σ_scaled
        jptprod!(nlp, x, dy, tmp)
        grad_p .+= tmp
    end

    vjp_fill_bound_pv!(sens.kkt, pvl, pvu, w)

    if has_lvar_param(nlp)
        unpack_dx!(x, cb, variable(pvl))
        lvar_jptprod!(nlp, x, tmp); grad_p .-= tmp
    end
    if has_uvar_param(nlp)
        unpack_dx!(x, cb, variable(pvu))
        uvar_jptprod!(nlp, x, tmp); grad_p .+= tmp
    end
    if has_lcon_param(nlp)
        unpack_slack!(y, cb, pvl, sens.is_eq, dual(w))
        lcon_jptprod!(nlp, y, tmp); grad_p .-= tmp
    end
    if has_ucon_param(nlp)
        unpack_slack!(y, cb, pvu, sens.is_eq, dual(w))
        ucon_jptprod!(nlp, y, tmp); grad_p .+= tmp
    end

    grad_p .*= -one(T)
    return result
end

function vjp_fill_bound_pv!(kkt, pvl, pvu, w)
    fill!(full(pvl), zero(eltype(full(pvl))))
    fill!(full(pvu), zero(eltype(full(pvu))))
    pvl.values_lr .=  kkt.l_lower .* dual_lb(w)
    pvu.values_ur .= .-kkt.u_lower .* dual_ub(w)
    return nothing
end
function vjp_fill_bound_pv!(::AbstractUnreducedKKTSystem, pvl, pvu, w)
    fill!(full(pvl), zero(eltype(full(pvl))))
    fill!(full(pvu), zero(eltype(full(pvu))))
    pvl.values_lr .=  dual_lb(w)
    pvu.values_ur .= .-dual_ub(w)
    return nothing
end
