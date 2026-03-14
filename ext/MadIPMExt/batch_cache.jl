struct BatchJVPCache{MT, BPV}
    kkt_rhs::BatchUnreducedKKTVector
    d2L_dxdp::MT        # (nvar × batch_size) Hessian-param product (packed)
    dg_dp::MT            # (m × batch_size) Jacobian-param product (packed)
    dlvar_dp::BPV        # BatchPrimalVector: lower var bound param product
    duvar_dp::BPV        # BatchPrimalVector: upper var bound param product
    dlcon_dp::MT         # (m × batch_size) lower con bound param product
    ducon_dp::MT         # (m × batch_size) upper con bound param product
    bx::MT               # (nvar_nlp × batch_size) current x in NLP space
    by::MT               # (m × batch_size) current y in NLP space
    hpv_nlp::MT          # (nvar_nlp × batch_size) hpprod output
    jpv_nlp::MT          # (m × batch_size) jpprod output
    dlvar_nlp::MT        # (nvar_nlp × batch_size) lvar_jpprod output
    duvar_nlp::MT        # (nvar_nlp × batch_size) uvar_jpprod output
    dlcon_nlp::MT        # (m × batch_size) lcon_jpprod output
    ducon_nlp::MT        # (m × batch_size) ucon_jpprod output
    grad_x::MT           # (nvar_nlp × batch_size) grad_f for dobj
    grad_p::MT           # (nparam × batch_size) grad_param for dobj
end

function _zeros_like(proto::AbstractMatrix{T}, dims::Int...) where {T}
    fill!(similar(proto, T, dims...), zero(T))
end

function get_batch_jvp_cache!(sens::BatchMadDiffSolver{T}) where {T}
    if isnothing(sens.jvp_cache)
        solver = sens.solver
        bcb = solver.bcb
        nlp = solver.nlp
        bs = solver.batch_size

        nvar_nlp = nlp.meta.nvar
        n_con = bcb.ncon
        n_p = sens.n_p
        nx = bcb.nvar
        ns = length(bcb.ind_ineq)
        n_tot = nx + ns
        m = bcb.ncon
        nlb = length(bcb.ind_lb)
        nub = length(bcb.ind_ub)

        proto = solver.workspace.bx
        MT = typeof(proto)
        VT = typeof(similar(proto, T, 0))

        sens.jvp_cache = BatchJVPCache{MT, BatchPrimalVector{T, MT, typeof(bcb.ind_lb)}}(
            BatchUnreducedKKTVector(MT, VT, n_tot, m, nlb, nub, bs, bcb.ind_lb, bcb.ind_ub),
            _zeros_like(proto, nx, bs),          # d2L_dxdp
            _zeros_like(proto, m, bs),            # dg_dp
            BatchPrimalVector(MT, VT, nx, ns, bs, bcb.ind_lb, bcb.ind_ub),  # dlvar_dp
            BatchPrimalVector(MT, VT, nx, ns, bs, bcb.ind_lb, bcb.ind_ub),  # duvar_dp
            _zeros_like(proto, m, bs),            # dlcon_dp
            _zeros_like(proto, m, bs),            # ducon_dp
            _zeros_like(proto, nvar_nlp, bs),     # bx
            _zeros_like(proto, n_con, bs),        # by
            _zeros_like(proto, nvar_nlp, bs),     # hpv_nlp
            _zeros_like(proto, n_con, bs),        # jpv_nlp
            _zeros_like(proto, nvar_nlp, bs),     # dlvar_nlp
            _zeros_like(proto, nvar_nlp, bs),     # duvar_nlp
            _zeros_like(proto, n_con, bs),        # dlcon_nlp
            _zeros_like(proto, n_con, bs),        # ducon_nlp
            _zeros_like(proto, nvar_nlp, bs),     # grad_x
            _zeros_like(proto, n_p, bs),          # grad_p
        )
    end
    return sens.jvp_cache
end

struct BatchJVPResult{MT, VT}
    dx::MT        # (nvar_nlp × batch_size)
    dy::MT        # (ncon × batch_size)
    dzl::MT       # (nvar_nlp × batch_size)
    dzu::MT       # (nvar_nlp × batch_size)
    dobj::VT      # (batch_size,)
end

function BatchJVPResult(sens::BatchMadDiffSolver{T}) where {T}
    solver = sens.solver
    bcb = solver.bcb
    nlp = solver.nlp
    bs = solver.batch_size
    nvar_nlp = nlp.meta.nvar
    n_con = bcb.ncon
    proto = solver.workspace.bx
    BatchJVPResult(
        _zeros_like(proto, nvar_nlp, bs),
        _zeros_like(proto, n_con, bs),
        _zeros_like(proto, nvar_nlp, bs),
        _zeros_like(proto, nvar_nlp, bs),
        fill!(similar(proto, T, bs), zero(T)),
    )
end

struct BatchVJPCache{MT, BPV}
    kkt_rhs::BatchUnreducedKKTVector
    dL_dx::MT         # (nvar × batch_size)
    dL_dy::MT         # (m × batch_size)
    dL_dzl::MT        # (nlb × batch_size)
    dL_dzu::MT        # (nub × batch_size)
    dzl_full::BPV     # BatchPrimalVector: work buffer for bound unpacking
    dzu_full::BPV     # BatchPrimalVector: work buffer for bound unpacking
    bx::MT            # (nvar_nlp × batch_size)
    by::MT            # (m × batch_size) y scaled for pullback
    dy_scaled::MT     # (m × batch_size) adjoint dy scaled for jptprod
    tmp_p::MT         # (nparam × batch_size)
    grad_x::MT        # (nvar × batch_size)
end

function get_batch_vjp_cache!(sens::BatchMadDiffSolver{T}) where {T}
    if isnothing(sens.vjp_cache)
        solver = sens.solver
        bcb = solver.bcb
        nlp = solver.nlp
        bs = solver.batch_size

        nvar_nlp = nlp.meta.nvar
        n_con = bcb.ncon
        n_p = sens.n_p
        nx = bcb.nvar
        ns = length(bcb.ind_ineq)
        n_tot = nx + ns
        m = bcb.ncon
        nlb = length(bcb.ind_lb)
        nub = length(bcb.ind_ub)

        proto = solver.workspace.bx
        MT = typeof(proto)
        VT = typeof(similar(proto, T, 0))

        sens.vjp_cache = BatchVJPCache{MT, BatchPrimalVector{T, MT, typeof(bcb.ind_lb)}}(
            BatchUnreducedKKTVector(MT, VT, n_tot, m, nlb, nub, bs, bcb.ind_lb, bcb.ind_ub),
            _zeros_like(proto, nx, bs),           # dL_dx
            _zeros_like(proto, m, bs),             # dL_dy
            _zeros_like(proto, nlb, bs),           # dL_dzl
            _zeros_like(proto, nub, bs),           # dL_dzu
            BatchPrimalVector(MT, VT, nx, ns, bs, bcb.ind_lb, bcb.ind_ub),  # dzl_full
            BatchPrimalVector(MT, VT, nx, ns, bs, bcb.ind_lb, bcb.ind_ub),  # dzu_full
            _zeros_like(proto, nvar_nlp, bs),      # bx
            _zeros_like(proto, n_con, bs),         # by
            _zeros_like(proto, n_con, bs),         # dy_scaled
            _zeros_like(proto, n_p, bs),           # tmp_p
            _zeros_like(proto, nx, bs),            # grad_x
        )
    end
    return sens.vjp_cache
end

struct BatchVJPResult{MT}
    dx::MT        # (nvar_nlp × batch_size)
    dy::MT        # (ncon × batch_size)
    dzl::MT       # (nvar_nlp × batch_size)
    dzu::MT       # (nvar_nlp × batch_size)
    grad_p::MT    # (nparam × batch_size)
end

function BatchVJPResult(sens::BatchMadDiffSolver{T}) where {T}
    solver = sens.solver
    bcb = solver.bcb
    nlp = solver.nlp
    bs = solver.batch_size
    nvar_nlp = nlp.meta.nvar
    n_con = bcb.ncon
    proto = solver.workspace.bx
    BatchVJPResult(
        _zeros_like(proto, nvar_nlp, bs),
        _zeros_like(proto, n_con, bs),
        _zeros_like(proto, nvar_nlp, bs),
        _zeros_like(proto, nvar_nlp, bs),
        _zeros_like(proto, sens.n_p, bs),
    )
end

has_hess_param(::Union{BatchJVPCache, BatchVJPCache}, meta) = meta.nnzhp != 0
has_jac_param(::Union{BatchJVPCache, BatchVJPCache}, meta)  = meta.nnzjp != 0
has_lvar_param(::Union{BatchJVPCache, BatchVJPCache}, meta) = meta.nnzjplvar != 0
has_uvar_param(::Union{BatchJVPCache, BatchVJPCache}, meta) = meta.nnzjpuvar != 0
has_lcon_param(::Union{BatchJVPCache, BatchVJPCache}, meta) = meta.nnzjplcon != 0
has_ucon_param(::Union{BatchJVPCache, BatchVJPCache}, meta) = meta.nnzjpucon != 0
has_grad_param(::Union{BatchJVPCache, BatchVJPCache}, meta) = meta.nnzgp != 0
