# ============================================================================
# Batch Jacobian–vector product — forward implicit differentiation.
#
# `Δp` is `(nparam, batch_size)`; every output is per-instance.
# ============================================================================

MadDiff.jacobian_vector_product!(sens::BatchMadDiffSolver, Δp::AbstractMatrix) =
    MadDiff.jacobian_vector_product!(BatchJVPResult(sens), sens, Δp)

function MadDiff.jacobian_vector_product!(
    result::BatchJVPResult, sens::BatchMadDiffSolver{T}, Δp::AbstractMatrix,
) where {T}
    cache  = get_batch_jvp_cache!(sens)
    solver = sens.solver
    bcb    = solver.problem.bcb
    nlp    = solver.problem.nlp
    meta   = nlp.meta
    Δp     = MadDiff._adapt_device(MadDiff._solver_proto(sens), Δp)

    MadNLP.unpack_x!(cache.bx, bcb, solver.state.x)
    MadNLP.unpack_y!(cache.by, bcb, MadNLP.full(solver.state.y))

    for buf in (cache.hpv_nlp, cache.jpv_nlp,
                cache.dlvar_nlp, cache.duvar_nlp,
                cache.dlcon_nlp, cache.ducon_nlp)
        fill!(buf, zero(T))
    end

    bx, by     = cache.bx, cache.by
    bobj_sign  = vec(bcb.obj_sign)
    has_hess_param(cache, meta) && hpprod!(nlp, bx, by, Δp, bobj_sign, cache.hpv_nlp)
    has_jac_param(cache, meta)  && jpprod!(nlp, bx, Δp, cache.jpv_nlp)
    has_lvar_param(cache, meta) && lvar_jpprod!(nlp, Δp, cache.dlvar_nlp)
    has_uvar_param(cache, meta) && uvar_jpprod!(nlp, Δp, cache.duvar_nlp)
    has_lcon_param(cache, meta) && lcon_jpprod!(nlp, Δp, cache.dlcon_nlp)
    has_ucon_param(cache, meta) && ucon_jpprod!(nlp, Δp, cache.ducon_nlp)

    _batch_pack_jvp!(cache, bcb)
    _batch_solve_jvp!(sens, cache)
    _batch_unpack_jvp!(result, cache, bcb)
    _batch_compute_objective_sensitivity!(result, cache, nlp, Δp)
    return result
end

# ---------- stages ----------

function _batch_pack_jvp!(cache::BatchJVPCache{MT}, bcb) where {MT}
    T = eltype(MT)
    fill!(cache.d2L_dxdp, zero(T))
    fill!(cache.dg_dp,    zero(T))
    fill!(cache.dlcon_dp, zero(T))
    fill!(cache.ducon_dp, zero(T))
    fill!(MadNLP.full(cache.dlvar_dp), zero(T))
    fill!(MadNLP.full(cache.duvar_dp), zero(T))

    MadDiff.pack_hess!(cache.d2L_dxdp, bcb, cache.hpv_nlp)
    MadDiff.pack_cons!(cache.dg_dp,    bcb, cache.jpv_nlp)
    MadDiff.pack_cons!(cache.dlcon_dp, bcb, cache.dlcon_nlp)
    MadDiff.pack_cons!(cache.ducon_dp, bcb, cache.ducon_nlp)
    MadDiff.pack_dx!(MadNLP.variable(cache.dlvar_dp), bcb, cache.dlvar_nlp)
    MadDiff.pack_dx!(MadNLP.variable(cache.duvar_dp), bcb, cache.duvar_nlp)
    isempty(bcb.ind_ineq) || begin
        MadDiff.pack_slack!(MadNLP.slack(cache.dlvar_dp), bcb, cache.dlcon_nlp)
        MadDiff.pack_slack!(MadNLP.slack(cache.duvar_dp), bcb, cache.ducon_nlp)
    end
    return nothing
end

function _batch_solve_jvp!(sens::BatchMadDiffSolver{T},
                            cache::BatchJVPCache) where {T}
    w   = cache.kkt_rhs
    bcb = sens.solver.problem.bcb

    fill!(MadNLP.full(w), zero(T))
    view(MadNLP.primal(w), 1:bcb.nvar, :) .= .-cache.d2L_dxdp
    # Extract the LHS alias so the `@.` macro doesn't try to reinterpret
    # `MadNLP.dual(w) = …` as a method definition. The broadcast stays fused.
    dw = MadNLP.dual(w)
    @. dw = -cache.dg_dp + sens.is_eq * (cache.dlcon_dp + cache.ducon_dp) / 2
    MadDiff.jvp_bound_rhs!(sens.solver.problem.kkt, w,
                                cache.dlvar_dp, cache.duvar_dp)

    MadDiff._kkt_solve_with_refine!(sens, w, cache, MadNLP.solve_refine!)
    return nothing
end

function _batch_unpack_jvp!(result::BatchJVPResult, cache::BatchJVPCache, bcb)
    w = cache.kkt_rhs
    MadDiff.unpack_dx!(result.dx, bcb, MadNLP.primal(w))
    MadDiff.set_fixed_sensitivity!(result.dx, bcb, cache.dlvar_nlp, cache.duvar_nlp)
    MadNLP.unpack_y!(result.dy, bcb, MadNLP.dual(w))
    MadDiff.unpack_dzl!(result.dzl, bcb, MadNLP.dual_lb(w), cache.dlvar_dp)
    MadDiff.unpack_dzu!(result.dzu, bcb, MadNLP.dual_ub(w), cache.duvar_dp)
    return result
end

function _batch_compute_objective_sensitivity!(
    result::BatchJVPResult, cache::BatchJVPCache{MT}, nlp, Δp,
) where {MT}
    T    = eltype(MT)
    meta = nlp.meta

    grad!(nlp, cache.bx, cache.grad_x)
    has_grad_param(cache, meta) ?
        grad_param!(nlp, cache.bx, cache.grad_p) :
        fill!(cache.grad_p, zero(T))

    # dobj[j] = Σᵢ grad_x[i,j]·dx[i,j] + Σᵢ grad_p[i,j]·Δp[i,j].
    # Do it allocation-free: mutate grad_x / grad_p in place (they're
    # write-only here), column-sum each into dobj / dobj_scratch with `sum!`,
    # then accumulate. `sum!` doesn't support β=1, hence the scratch.
    cache.grad_x .*= result.dx
    cache.grad_p .*= Δp
    sum!(reshape(result.dobj,        1, :), cache.grad_x)
    sum!(reshape(cache.dobj_scratch, 1, :), cache.grad_p)
    result.dobj .+= cache.dobj_scratch
    return nothing
end
