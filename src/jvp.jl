# ============================================================================
# Jacobian–vector product — forward implicit differentiation.
#
#   (∂F/∂z)·(dz/dp)·Δp = -(∂F/∂p)·Δp
#
# where F is the KKT residual and z = (x, y, zl, zu). We evaluate the RHS by
# contracting ∂/∂p operators (`hpprod!`, `jpprod!`, bound/con param Jacobians)
# against `Δp`, then solve against the KKT factorization.
# ============================================================================

function jacobian_vector_product!(
    result::JVPResult, sens::MadDiffSolver{T}, Δp::AbstractVector,
) where {T}
    cache  = get_jvp_cache!(sens)
    Δp_dev = _adapt_device(cache.grad_p, Δp)
    _fill_jvp_rhs!(cache, sens, Δp_dev)
    _pack_jvp!(cache, sens)
    _solve_jvp!(sens, cache)
    # `dobj` is populated only by `compute_objective_sensitivity!`; reset here
    # so that a user who calls `jacobian_vector_product!` twice without an
    # interleaving `compute_objective_sensitivity!` doesn't read stale data.
    _unpack_jvp!(result, cache, sens)
    result.dobj[] = zero(T)
    return result
end

function compute_objective_sensitivity!(
    result::JVPResult, sens::MadDiffSolver{T}, Δp::AbstractVector,
) where {T}
    cache = get_jvp_cache!(sens)
    nlp   = sens.solver.nlp

    fill!(cache.grad_x, zero(T))
    fill!(cache.grad_p, zero(T))

    grad!(nlp, cache.x_nlp, cache.grad_x)
    has_grad_param(nlp) && grad_param!(nlp, cache.x_nlp, cache.grad_p)

    # Match `Δp`'s device to the internal buffers so a host `Δp` still works
    # against a GPU solver. `_adapt_device` is a no-op when already matching.
    Δp_dev = _adapt_device(cache.grad_p, Δp)
    result.dobj[] = dot(cache.grad_x, result.dx) + dot(cache.grad_p, Δp_dev)
    return result
end

# ---------- stages ----------

function _fill_jvp_rhs!(cache, sens::MadDiffSolver{T}, Δp) where {T}
    solver = sens.solver
    cb     = solver.cb
    nlp    = solver.nlp

    unpack_x!(cache.x_nlp, cb, variable(solver.x))
    unpack_y!(cache.y_nlp, cb, solver.y)

    for buf in (cache.hpv_nlp, cache.jpv_nlp,
                cache.dlvar_nlp, cache.duvar_nlp,
                cache.dlcon_nlp, cache.ducon_nlp)
        fill!(buf, zero(T))
    end

    x, y = cache.x_nlp, cache.y_nlp
    has_hess_param(nlp) && hpprod!(nlp, x, y, Δp, cache.hpv_nlp; obj_weight = cb.obj_sign)
    has_jac_param(nlp)  && jpprod!(nlp, x, Δp, cache.jpv_nlp)
    has_lvar_param(nlp) && lvar_jpprod!(nlp, Δp, cache.dlvar_nlp)
    has_uvar_param(nlp) && uvar_jpprod!(nlp, Δp, cache.duvar_nlp)
    has_lcon_param(nlp) && lcon_jpprod!(nlp, Δp, cache.dlcon_nlp)
    has_ucon_param(nlp) && ucon_jpprod!(nlp, Δp, cache.ducon_nlp)
    return nothing
end

function _pack_jvp!(cache, sens::MadDiffSolver{T}) where {T}
    cb = sens.solver.cb

    fill!(cache.d2L_dxdp, zero(T))
    fill!(cache.dg_dp,    zero(T))
    fill!(cache.dlcon_dp, zero(T))
    fill!(cache.ducon_dp, zero(T))
    fill!(full(cache.dlvar_dp), zero(T))
    fill!(full(cache.duvar_dp), zero(T))

    pack_hess!(cache.d2L_dxdp,            cb, cache.hpv_nlp)
    pack_cons!(cache.dg_dp,               cb, cache.jpv_nlp)
    pack_cons!(cache.dlcon_dp,            cb, cache.dlcon_nlp)
    pack_cons!(cache.ducon_dp,            cb, cache.ducon_nlp)
    pack_dx!(variable(cache.dlvar_dp),    cb, cache.dlvar_nlp)
    pack_dx!(variable(cache.duvar_dp),    cb, cache.duvar_nlp)
    pack_slack!(slack(cache.dlvar_dp),    cb, cache.dlcon_nlp)
    pack_slack!(slack(cache.duvar_dp),    cb, cache.ducon_nlp)
    return nothing
end

function _solve_jvp!(sens::MadDiffSolver{T}, cache) where {T}
    w   = cache.kkt_rhs
    n_x = length(cache.d2L_dxdp)

    fill!(full(w), zero(T))
    primal(w)[1:n_x] .= .-cache.d2L_dxdp
    # Extract the LHS alias so the `@.` macro doesn't try to reinterpret
    # `dual(w) = …` as a method definition. The broadcast stays fused.
    dw = dual(w)
    @. dw = -cache.dg_dp + sens.is_eq * (cache.dlcon_dp + cache.ducon_dp) / 2
    jvp_bound_rhs!(sens.kkt, w, cache.dlvar_dp, cache.duvar_dp)

    _kkt_solve_with_refine!(sens, w, cache, solve_refine!)
    return nothing
end

# Condensed/augmented systems scale the bound RHS by `l_lower`/`u_lower`;
# unreduced systems carry the bound multipliers directly.
function jvp_bound_rhs!(kkt, w, dlvar_dp, duvar_dp)
    dual_lb(w) .=  kkt.l_lower .* dlvar_dp.values_lr
    dual_ub(w) .= .-kkt.u_lower .* duvar_dp.values_ur
    return nothing
end
function jvp_bound_rhs!(::AbstractUnreducedKKTSystem, w, dlvar_dp, duvar_dp)
    dual_lb(w) .=  dlvar_dp.values_lr
    dual_ub(w) .= .-duvar_dp.values_ur
    return nothing
end

function _unpack_jvp!(result::JVPResult, cache, sens::MadDiffSolver)
    cb = sens.solver.cb
    w  = cache.kkt_rhs

    unpack_dx!(result.dx, cb, primal(w))
    set_fixed_sensitivity!(result.dx, cb, cache.dlvar_nlp, cache.duvar_nlp)
    unpack_y!(result.dy, cb, dual(w))
    unpack_dzl!(result.dzl, cb, dual_lb(w), cache.dlvar_dp)
    unpack_dzu!(result.dzu, cb, dual_ub(w), cache.duvar_dp)
    return result
end
