# ============================================================================
# Batch JVP/VJP scratch caches and result containers.
# ============================================================================

_bzeros(proto::AbstractMatrix{T}, dims::Int...) where {T} =
    fill!(similar(proto, T, dims...), zero(T))

# ---------- JVP ----------

struct BatchJVPCache{MT, BPV, VT}
    kkt_rhs::BatchUnreducedKKTVector
    d2L_dxdp::MT            # (nvar, bs)      packed Hessian-param product
    dg_dp::MT               # (m, bs)         packed Jacobian-param product
    dlvar_dp::BPV           # lower variable-bound param product
    duvar_dp::BPV           # upper variable-bound param product
    dlcon_dp::MT            # (m, bs)
    ducon_dp::MT            # (m, bs)
    bx::MT                  # (nvar_nlp, bs)  current x (NLP space)
    by::MT                  # (m, bs)         current y (NLP space)
    hpv_nlp::MT             # (nvar_nlp, bs)  hpprod output
    jpv_nlp::MT             # (m, bs)         jpprod output
    dlvar_nlp::MT           # (nvar_nlp, bs)
    duvar_nlp::MT           # (nvar_nlp, bs)
    dlcon_nlp::MT           # (m, bs)
    ducon_nlp::MT           # (m, bs)
    grad_x::MT              # (nvar_nlp, bs)
    grad_p::MT              # (nparam, bs)
    dobj_scratch::VT        # (bs,) — scratch for the second column-sum in
                            #         `_batch_compute_objective_sensitivity!`
end

function get_batch_jvp_cache!(sens::BatchMadDiffSolver{T}) where {T}
    sens.jvp_cache === nothing || return sens.jvp_cache

    solver   = sens.solver
    bcb      = solver.problem.bcb
    nlp      = solver.problem.nlp
    bs       = solver.problem.batch_size
    nvar_nlp = nlp.meta.nvar
    m        = bcb.ncon
    nx, ns   = bcb.nvar, length(bcb.ind_ineq)
    n_tot    = nx + ns
    nlb, nub = length(bcb.ind_lb), length(bcb.ind_ub)

    proto = solver.state.workspace.bx
    MT    = typeof(proto)
    VT    = typeof(similar(proto, T, 0))
    z(n)  = _bzeros(proto, n, bs)
    zv(n) = fill!(similar(proto, T, n), zero(T))
    pv()  = BatchPrimalVector(MT, VT, nx, ns, bs, bcb.ind_lb, bcb.ind_ub)

    sens.jvp_cache = BatchJVPCache{MT, BatchPrimalVector{T, MT, typeof(bcb.ind_lb)}, VT}(
        BatchUnreducedKKTVector(MT, VT, n_tot, m, nlb, nub, bs, bcb.ind_lb, bcb.ind_ub),
        z(nx), z(m),
        pv(), pv(),
        z(m), z(m),
        z(nvar_nlp), z(m),
        z(nvar_nlp), z(m),
        z(nvar_nlp), z(nvar_nlp),
        z(m), z(m),
        z(nvar_nlp), z(sens.n_p),
        zv(bs),
    )
    return sens.jvp_cache
end

"""
    BatchJVPResult{MT, VT}

Per-instance directional sensitivities from [`jacobian_vector_product!`](@ref).
Matrix fields are `(dim, bs)`; `dobj::Vector` is length `bs`.
"""
struct BatchJVPResult{MT, VT}
    dx::MT          # (nvar_nlp, bs)
    dy::MT          # (ncon, bs)
    dzl::MT         # (nvar_nlp, bs)
    dzu::MT         # (nvar_nlp, bs)
    dobj::VT        # (bs,)
end

function BatchJVPResult(sens::BatchMadDiffSolver{T}) where {T}
    solver   = sens.solver
    bcb      = solver.problem.bcb
    nvar_nlp = solver.problem.nlp.meta.nvar
    bs       = solver.problem.batch_size
    proto    = solver.state.workspace.bx

    return BatchJVPResult(
        _bzeros(proto, nvar_nlp, bs),
        _bzeros(proto, bcb.ncon, bs),
        _bzeros(proto, nvar_nlp, bs),
        _bzeros(proto, nvar_nlp, bs),
        fill!(similar(proto, T, bs), zero(T)),
    )
end

# ---------- VJP ----------

struct BatchVJPCache{MT, BPV, VT}
    kkt_rhs::BatchUnreducedKKTVector
    dL_dx::MT               # (nvar, bs)
    dL_dy::MT               # (m, bs)
    dL_dzl::MT              # (nlb, bs)
    dL_dzu::MT              # (nub, bs)
    dzl_full::BPV
    dzu_full::BPV
    bx::MT                  # (nvar_nlp, bs)
    by::MT                  # (m, bs)
    dy_scaled::MT           # (m, bs)
    tmp_p::MT               # (nparam, bs)
    grad_x::MT              # (nvar, bs)
    bσ_scaled::VT           # (bs,) — `obj_sign .* obj_scale`, cached once
                            #         (constant across sensitivity calls).
end

function get_batch_vjp_cache!(sens::BatchMadDiffSolver{T}) where {T}
    sens.vjp_cache === nothing || return sens.vjp_cache

    solver   = sens.solver
    bcb      = solver.problem.bcb
    nlp      = solver.problem.nlp
    bs       = solver.problem.batch_size
    nvar_nlp = nlp.meta.nvar
    m        = bcb.ncon
    nx, ns   = bcb.nvar, length(bcb.ind_ineq)
    n_tot    = nx + ns
    nlb, nub = length(bcb.ind_lb), length(bcb.ind_ub)

    proto = solver.state.workspace.bx
    MT    = typeof(proto)
    VT    = typeof(similar(proto, T, 0))
    z(n)  = _bzeros(proto, n, bs)
    pv()  = BatchPrimalVector(MT, VT, nx, ns, bs, bcb.ind_lb, bcb.ind_ub)

    bσ_scaled = vec(bcb.obj_sign) .* vec(bcb.obj_scale)

    sens.vjp_cache = BatchVJPCache{MT, BatchPrimalVector{T, MT, typeof(bcb.ind_lb)}, VT}(
        BatchUnreducedKKTVector(MT, VT, n_tot, m, nlb, nub, bs, bcb.ind_lb, bcb.ind_ub),
        z(nx),  z(m),
        z(nlb), z(nub),
        pv(), pv(),
        z(nvar_nlp), z(m), z(m),
        z(sens.n_p), z(nx),
        bσ_scaled,
    )
    return sens.vjp_cache
end

"""
    BatchVJPResult{MT}

Per-instance result from [`vector_jacobian_product!`](@ref). `grad_p` is the
parameter gradient `(nparam, bs)`.
"""
struct BatchVJPResult{MT}
    dx::MT
    dy::MT
    dzl::MT
    dzu::MT
    grad_p::MT
end

function BatchVJPResult(sens::BatchMadDiffSolver{T}) where {T}
    solver   = sens.solver
    bcb      = solver.problem.bcb
    nvar_nlp = solver.problem.nlp.meta.nvar
    bs       = solver.problem.batch_size
    proto    = solver.state.workspace.bx

    return BatchVJPResult(
        _bzeros(proto, nvar_nlp, bs),
        _bzeros(proto, bcb.ncon, bs),
        _bzeros(proto, nvar_nlp, bs),
        _bzeros(proto, nvar_nlp, bs),
        _bzeros(proto, sens.n_p, bs),
    )
end

# ---------- parametric-capability probes (batch NLP meta) ----------

const _BatchCache = Union{BatchJVPCache, BatchVJPCache}

has_hess_param(::_BatchCache, meta) = meta.nnzhp     != 0
has_jac_param( ::_BatchCache, meta) = meta.nnzjp     != 0
has_lvar_param(::_BatchCache, meta) = meta.nnzjplvar != 0
has_uvar_param(::_BatchCache, meta) = meta.nnzjpuvar != 0
has_lcon_param(::_BatchCache, meta) = meta.nnzjplcon != 0
has_ucon_param(::_BatchCache, meta) = meta.nnzjpucon != 0
has_grad_param(::_BatchCache, meta) = meta.nnzgp     != 0

# ---------- constructor ----------

function MadDiff.BatchMadDiffSolver(batch_solver::UniformBatchMPCSolver{T}) where {T}
    bcb   = batch_solver.problem.bcb
    # `is_eq` must live on the solver's device — it participates in GPU
    # broadcasts inside `_batch_solve_jvp!` and `_batch_vjp_pullback!`.
    proto = batch_solver.state.workspace.bx
    is_eq = fill!(similar(proto, Bool, bcb.ncon), false)
    is_eq[bcb.ind_eq] .= true

    n_p = batch_solver.problem.nlp.meta.nparam
    MT  = typeof(proto)
    VT  = typeof(similar(proto, T, 0))
    FC  = BatchJVPCache{MT, BatchPrimalVector{T, MT, typeof(bcb.ind_lb)}, VT}
    RC  = BatchVJPCache{MT, BatchPrimalVector{T, MT, typeof(bcb.ind_lb)}, VT}

    # Refactorise the batch KKT system at the solution before use.
    _batch_refactorize!(batch_solver.problem.kkt, batch_solver)

    return BatchMadDiffSolver{T, typeof(batch_solver), typeof(is_eq), FC, RC}(
        batch_solver, n_p, is_eq, nothing, nothing,
    )
end

function _batch_refactorize!(kkt::SparseUniformBatchKKTSystem, batch_solver::UniformBatchMPCSolver)
    MadIPM.reset_active_view!(kkt.batch_views)
    MadIPM.set_aug_diagonal_reg!(kkt, batch_solver)
    MadNLP.eval_jac_wrapper!(batch_solver, kkt)
    MadNLP.eval_lag_hess_wrapper!(batch_solver, kkt)
    MadNLP.build_kkt!(kkt)
    MadNLP.factorize_kkt!(kkt)
    return nothing
end
