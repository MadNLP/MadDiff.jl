"""
    jacobian_vector_product!(sens::BatchMadDiffSolver, Δp::AbstractMatrix)

Compute batch JVP of the optimal solution with respect to parameters.
`Δp` is `(nparam × batch_size)`.

Returns a [`BatchJVPResult`] with per-instance solution sensitivities.
"""
function MadDiff.jacobian_vector_product!(sens::BatchMadDiffSolver, Δp::AbstractMatrix)
    return MadDiff.jacobian_vector_product!(BatchJVPResult(sens), sens, Δp)
end

function MadDiff.jacobian_vector_product!(
    result::BatchJVPResult, sens::BatchMadDiffSolver{T}, Δp::AbstractMatrix,
) where {T}
    solver = sens.solver
    bcb = solver.bcb
    nlp = solver.nlp
    meta = nlp.meta
    cache = get_batch_jvp_cache!(sens)

    MadNLP.unpack_x!(cache.bx, bcb, solver.x)
    MadNLP.unpack_y!(cache.by, bcb, MadNLP.full(solver.y))

    bx = cache.bx
    by = cache.by

    fill!(cache.hpv_nlp, zero(T))
    fill!(cache.jpv_nlp, zero(T))
    fill!(cache.dlvar_nlp, zero(T))
    fill!(cache.duvar_nlp, zero(T))
    fill!(cache.dlcon_nlp, zero(T))
    fill!(cache.ducon_nlp, zero(T))

    bobj_sign = vec(bcb.obj_sign)
    has_hess_param(cache, meta) && hpprod!(nlp, bx, by, Δp, bobj_sign, cache.hpv_nlp)
    has_jac_param(cache, meta)  && jpprod!(nlp, bx, Δp, cache.jpv_nlp)
    has_lvar_param(cache, meta) && lvar_jpprod!(nlp, Δp, cache.dlvar_nlp)
    has_uvar_param(cache, meta) && uvar_jpprod!(nlp, Δp, cache.duvar_nlp)
    has_lcon_param(cache, meta) && lcon_jpprod!(nlp, Δp, cache.dlcon_nlp)
    has_ucon_param(cache, meta) && ucon_jpprod!(nlp, Δp, cache.ducon_nlp)

    _batch_pack_jvp!(sens, cache)
    _batch_solve_jvp!(sens)
    _batch_unpack_jvp!(result, sens, cache)
    _batch_compute_objective_sensitivity!(result, sens, cache, Δp)

    return result
end

function _batch_pack_jvp!(sens::BatchMadDiffSolver{T}, cache::BatchJVPCache) where {T}
    bcb = sens.solver.bcb
    nx = bcb.nvar
    ind_ineq = bcb.ind_ineq

    fill!(cache.d2L_dxdp, zero(T))
    fill!(cache.dg_dp, zero(T))
    fill!(cache.dlcon_dp, zero(T))
    fill!(cache.ducon_dp, zero(T))
    fill!(MadNLP.full(cache.dlvar_dp), zero(T))
    fill!(MadNLP.full(cache.duvar_dp), zero(T))

    MadDiff.pack_hess!(cache.d2L_dxdp, bcb, cache.hpv_nlp)
    MadDiff.pack_cons!(cache.dg_dp, bcb, cache.jpv_nlp)
    MadDiff.pack_cons!(cache.dlcon_dp, bcb, cache.dlcon_nlp)
    MadDiff.pack_cons!(cache.ducon_dp, bcb, cache.ducon_nlp)

    MadDiff.pack_dx!(MadNLP.variable(cache.dlvar_dp), bcb, cache.dlvar_nlp)
    MadDiff.pack_dx!(MadNLP.variable(cache.duvar_dp), bcb, cache.duvar_nlp)

    ns = length(ind_ineq)
    if ns > 0
        MadDiff.pack_slack!(MadNLP.slack(cache.dlvar_dp), bcb, cache.dlcon_nlp)
        MadDiff.pack_slack!(MadNLP.slack(cache.duvar_dp), bcb, cache.ducon_nlp)
    end

    return nothing
end

function _batch_solve_jvp!(sens::BatchMadDiffSolver{T}) where {T}
    cache = get_batch_jvp_cache!(sens)
    w = cache.kkt_rhs
    _batch_assemble_jvp_rhs!(sens, w, cache)
    _solve_with_refine!(sens, w, cache)
    return nothing
end

function _batch_assemble_jvp_rhs!(
    sens::BatchMadDiffSolver{T}, w::BatchUnreducedKKTVector, cache::BatchJVPCache,
) where {T}
    bcb = sens.solver.bcb
    nx = bcb.nvar

    fill!(MadNLP.full(w), zero(T))
    view(MadNLP.primal(w), 1:nx, :) .= .-cache.d2L_dxdp
    MadNLP.dual(w) .= .-cache.dg_dp .+ sens.is_eq .* (cache.dlcon_dp .+ cache.ducon_dp) ./ 2
    MadDiff.jvp_set_bound_rhs!(sens.solver.kkt, w, cache.dlvar_dp, cache.duvar_dp)

    return w
end

function _batch_unpack_jvp!(
    result::BatchJVPResult, sens::BatchMadDiffSolver, cache::BatchJVPCache,
)
    bcb = sens.solver.bcb
    w = cache.kkt_rhs
    ind_lb = bcb.ind_lb
    ind_ub = bcb.ind_ub

    MadDiff.unpack_dx!(result.dx, bcb, MadNLP.primal(w))
    MadDiff.set_fixed_sensitivity!(result.dx, bcb, cache.dlvar_nlp, cache.duvar_nlp)
    MadNLP.unpack_y!(result.dy, bcb, MadNLP.dual(w))

    MadDiff.unpack_dzl!(result.dzl, bcb, MadNLP.dual_lb(w), cache.dlvar_dp)
    MadDiff.unpack_dzu!(result.dzu, bcb, MadNLP.dual_ub(w), cache.duvar_dp)

    return result
end

function _batch_compute_objective_sensitivity!(
    result::BatchJVPResult, sens::BatchMadDiffSolver{T}, cache::BatchJVPCache, Δp::AbstractMatrix,
) where {T}
    solver = sens.solver
    nlp = solver.nlp
    meta = nlp.meta
    bx = cache.bx

    grad!(nlp, bx, cache.grad_x)
    if has_grad_param(cache, meta)
        grad_param!(nlp, bx, cache.grad_p)
    else
        fill!(cache.grad_p, zero(T))
    end

    # dobj[j] = dot(grad_x[:,j], dx[:,j]) + dot(grad_p[:,j], Δp[:,j])
    result.dobj .= vec(sum(cache.grad_x .* result.dx, dims=1)) .+
                   vec(sum(cache.grad_p .* Δp, dims=1))

    return nothing
end
