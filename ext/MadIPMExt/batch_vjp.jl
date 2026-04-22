# ============================================================================
# Batch vector–Jacobian product — reverse implicit differentiation.
#
# Seeds are `(dim, bs)` matrices (or `Vector{<:Real}` of length `bs` for
# `dobj`). Returns a `BatchVJPResult` whose `grad_p` is `(nparam, bs)`.
# ============================================================================

function MadDiff.vector_jacobian_product!(
    sens::BatchMadDiffSolver;
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing,
    dobj  = nothing,
)
    return MadDiff.vector_jacobian_product!(BatchVJPResult(sens), sens;
        dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
end

function MadDiff.vector_jacobian_product!(
    result::BatchVJPResult, sens::BatchMadDiffSolver{T};
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing,
    dobj  = nothing,
) where {T}
    all(isnothing, (dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)) &&
        throw(ArgumentError(
            "vector_jacobian_product!: at least one of " *
            "`dL_dx`, `dL_dy`, `dL_dzl`, `dL_dzu`, `dobj` must be provided."))

    # Promote host-typed seeds to the batch solver's device. All five seeds
    # participate in on-device broadcasts (`dobj` via `reshape(·, 1, :)` in
    # `_batch_pack_vjp!` / `_batch_vjp_pullback!`), so any CPU-only seed
    # would force an implicit transfer every call without this.
    proto  = MadDiff._solver_proto(sens)
    dL_dx  = dL_dx  === nothing ? nothing : MadDiff._adapt_device(proto, dL_dx)
    dL_dy  = dL_dy  === nothing ? nothing : MadDiff._adapt_device(proto, dL_dy)
    dL_dzl = dL_dzl === nothing ? nothing : MadDiff._adapt_device(proto, dL_dzl)
    dL_dzu = dL_dzu === nothing ? nothing : MadDiff._adapt_device(proto, dL_dzu)
    dobj   = dobj   === nothing ? nothing : MadDiff._adapt_device(proto, dobj)

    _batch_pack_vjp!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
    _batch_solve_vjp!(sens)
    _batch_unpack_vjp!(result, sens)
    _batch_vjp_pullback!(result, sens; dobj)
    return result
end

# ---------- stages ----------

function _batch_pack_vjp!(
    sens::BatchMadDiffSolver{T};
    dL_dx, dL_dy, dL_dzl, dL_dzu, dobj,
) where {T}
    cache = get_batch_vjp_cache!(sens)
    bcb   = sens.solver.problem.bcb

    for buf in (cache.dL_dx, cache.dL_dy, cache.dL_dzl, cache.dL_dzu)
        fill!(buf, zero(T))
    end
    fill!(MadNLP.full(cache.dzl_full), zero(T))
    fill!(MadNLP.full(cache.dzu_full), zero(T))

    isnothing(dL_dx)  || MadDiff.pack_dx!(cache.dL_dx,  bcb, dL_dx)
    isnothing(dL_dy)  || MadDiff.pack_dy!(cache.dL_dy,  bcb, dL_dy)
    isnothing(dL_dzl) || MadDiff.pack_dzl!(cache.dL_dzl, bcb, dL_dzl, cache.dzl_full)
    isnothing(dL_dzu) || MadDiff.pack_dzu!(cache.dL_dzu, bcb, dL_dzu, cache.dzu_full)

    if !isnothing(dobj)
        MadNLP.unpack_x!(cache.bx, bcb, sens.solver.state.x)
        MadNLP._eval_grad_f_wrapper!(bcb, cache.bx, cache.grad_x)
        cache.dL_dx .+= cache.grad_x .* reshape(dobj, 1, :)
    end
    return nothing
end

function _batch_solve_vjp!(sens::BatchMadDiffSolver{T}) where {T}
    cache = get_batch_vjp_cache!(sens)
    w     = cache.kkt_rhs
    nx    = size(cache.dL_dx, 1)

    fill!(MadNLP.full(w), zero(T))
    view(MadNLP.primal(w), 1:nx, :) .= cache.dL_dx
    MadNLP.dual(w)    .= cache.dL_dy
    MadNLP.dual_lb(w) .= cache.dL_dzl
    MadNLP.dual_ub(w) .= cache.dL_dzu

    MadDiff._kkt_solve_with_refine!(sens, w, cache, MadDiff.adjoint_solve_refine!)
    return nothing
end

function _batch_unpack_vjp!(result::BatchVJPResult, sens::BatchMadDiffSolver)
    cache = get_batch_vjp_cache!(sens)
    bcb   = sens.solver.problem.bcb
    w     = cache.kkt_rhs

    MadDiff.unpack_dx!(result.dx, bcb, MadNLP.primal(w))
    MadNLP.unpack_y!(result.dy,   bcb, MadNLP.dual(w))
    MadDiff.unpack_dzl!(result.dzl, bcb, MadNLP.dual_lb(w), cache.dzl_full)
    MadDiff.unpack_dzu!(result.dzu, bcb, MadNLP.dual_ub(w), cache.dzu_full)
    return result
end

function _batch_vjp_pullback!(
    result::BatchVJPResult, sens::BatchMadDiffSolver{T}; dobj,
) where {T}
    solver   = sens.solver
    nlp      = solver.problem.nlp
    meta     = nlp.meta
    bcb      = solver.problem.bcb
    cache    = get_batch_vjp_cache!(sens)
    w        = cache.kkt_rhs
    bx, by   = cache.bx, cache.by
    dx       = result.dx
    dy       = cache.dy_scaled
    pvl, pvu = cache.dzl_full, cache.dzu_full
    tmp      = cache.tmp_p
    grad_p   = fill!(result.grad_p, zero(T))

    bσ_scaled = cache.bσ_scaled

    MadNLP.unpack_x!(bx, bcb, solver.state.x)

    if has_hess_param(cache, meta)
        MadNLP.unpack_y!(by, bcb, MadNLP.full(solver.state.y))
        by .*= reshape(bσ_scaled, 1, :)
        hptprod!(nlp, bx, by, dx, bσ_scaled, grad_p)
    end

    if !isnothing(dobj) && has_grad_param(cache, meta)
        grad_param!(nlp, bx, tmp)
        tmp .*= reshape(dobj, 1, :)
        grad_p .-= tmp
    end

    if has_jac_param(cache, meta)
        dy .= result.dy .* reshape(bσ_scaled, 1, :)
        jptprod!(nlp, bx, dy, tmp)
        grad_p .+= tmp
    end

    MadDiff.vjp_fill_bound_pv!(solver.problem.kkt, pvl, pvu, w)

    if has_lvar_param(cache, meta)
        MadDiff.unpack_dx!(bx, bcb, MadNLP.variable(pvl))
        lvar_jptprod!(nlp, bx, tmp); grad_p .-= tmp
    end
    if has_uvar_param(cache, meta)
        MadDiff.unpack_dx!(bx, bcb, MadNLP.variable(pvu))
        uvar_jptprod!(nlp, bx, tmp); grad_p .+= tmp
    end
    if has_lcon_param(cache, meta)
        MadDiff.unpack_slack!(by, bcb, pvl, sens.is_eq, MadNLP.dual(w))
        lcon_jptprod!(nlp, by, tmp); grad_p .-= tmp
    end
    if has_ucon_param(cache, meta)
        MadDiff.unpack_slack!(by, bcb, pvu, sens.is_eq, MadNLP.dual(w))
        ucon_jptprod!(nlp, by, tmp); grad_p .+= tmp
    end

    grad_p .*= -one(T)
    return result
end
