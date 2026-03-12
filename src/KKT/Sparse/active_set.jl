struct SparseActiveSetKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT,
    QN,
    LS,
    VI <: AbstractVector{Int},
    VI32 <: AbstractVector{Int32},
} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, QN}
    # -- Full Hessian (n×n)
    hess::VT
    hess_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}

    # -- Full Jacobian (m×n)
    jac_callback::VT
    jac_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}

    # -- Active Jacobian (n_a×n)
    jac_active_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    jac_active_com::MT
    jac_active_csc_map::Union{Nothing, VI}
    jac_active_filter::VI   # indices into jac_callback for active-row entries

    # -- Reduced augmented system (dim = n + ns_a + n_a)
    aug_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}

    # -- Full-dim KKT fields
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # -- Full-dim index sets
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # -- Active-set mappings
    active_cons::VI       # active constraint indices into 1:m  (length n_a)
    active_slack_map::VI  # active slack indices into 1:ns      (length ns_a)
    ind_ineq_active::VI   # maps active-slack k → row in active block
    n_active::Int
    ns_active::Int

    # -- Gather/scatter indices
    ind_active_slack_pd::VI  # n .+ active_slack_map
    ind_active_dual_pd::VI   # (n+ns) .+ active_cons
    ind_slack_cons::VI       # active_cons[ind_ineq_active]
    work_n_a::VT

    # -- Work vector for reduced solve
    work_reduced::VT

    # -- Cached sizes for build_kkt!
    _nnzh::Int
    _nnzj_a::Int

    # -- Linear solver
    linear_solver::LS
    etc::Dict{Symbol, Any}
end

num_variables(kkt::SparseActiveSetKKTSystem) = length(kkt.pr_diag)

MadNLP.get_jacobian(kkt::SparseActiveSetKKTSystem) = kkt.jac_callback

function MadNLP.compress_hessian!(kkt::SparseActiveSetKKTSystem)
    transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function MadNLP.compress_jacobian!(kkt::SparseActiveSetKKTSystem)
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
    kkt.jac_active_raw.V .= kkt.jac_callback[kkt.jac_active_filter]
    transfer!(kkt.jac_active_com, kkt.jac_active_raw, kkt.jac_active_csc_map)
end

function MadNLP.is_inertia_correct(kkt::SparseActiveSetKKTSystem, num_pos, num_zero, num_neg)
    n = size(kkt.hess_com, 1)
    return (num_zero == 0) && (num_pos == n + kkt.ns_active) && (num_neg == kkt.n_active)
end

function MadNLP.regularize_diagonal!(kkt::SparseActiveSetKKTSystem, primal, dual)
    kkt.reg .+= primal
    kkt.pr_diag .+= primal
    kkt.du_diag .-= dual
end

MadNLP.factorize_kkt!(kkt::SparseActiveSetKKTSystem) = MadNLP.factorize!(kkt.linear_solver)

function MadNLP.initialize!(kkt::SparseActiveSetKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    fill!(nonzeros(kkt.hess_com), zero(T))
    return
end

function MadNLP.build_kkt!(kkt::SparseActiveSetKKTSystem{T}) where T
    V    = kkt.aug_raw.V
    n    = size(kkt.hess_com, 1)
    nnzh = kkt._nnzh
    nnzj_a = kkt._nnzj_a
    ns_a = kkt.ns_active
    n_a  = kkt.n_active

    off = 0
    V[off+1:off+n] .= view(kkt.pr_diag, 1:n)
    off += n

    V[off+1:off+nnzh] .= kkt.hess
    off += nnzh

    if ns_a > 0
        V[off+1:off+ns_a] .= kkt.pr_diag[kkt.ind_active_slack_pd]
    end
    off += ns_a

    if nnzj_a > 0
        V[off+1:off+nnzj_a] .= kkt.jac_callback[kkt.jac_active_filter]
    end
    off += nnzj_a

    off += ns_a

    if n_a > 0
        V[off+1:off+n_a] .= kkt.du_diag[kkt.active_cons]
    end

    transfer!(kkt.aug_com, kkt.aug_raw, kkt.aug_csc_map)
    return
end

function MadNLP.solve_kkt!(kkt::SparseActiveSetKKTSystem{T}, w::AbstractKKTVector) where T
    n    = size(kkt.hess_com, 1)
    ns   = length(kkt.ind_ineq)
    ns_a = kkt.ns_active
    m    = length(kkt.du_diag)

    MadNLP.reduce_rhs!(kkt, w)

    pd = primal_dual(w)
    wr = kkt.work_reduced

    copyto!(view(wr, 1:n), view(pd, 1:n))
    wr[n+1:n+ns_a]   .= pd[kkt.ind_active_slack_pd]
    wr[n+ns_a+1:end]  .= pd[kkt.ind_active_dual_pd]

    solve_linear_system!(kkt.linear_solver, wr)

    copyto!(view(pd, 1:n), view(wr, 1:n))
    fill!(view(pd, n+1:n+ns), zero(T))
    pd[kkt.ind_active_slack_pd] .= view(wr, n+1:n+ns_a)
    fill!(view(pd, n+ns+1:n+ns+m), zero(T))
    pd[kkt.ind_active_dual_pd]  .= view(wr, n+ns_a+1:length(wr))

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

# ---------------------------------------------------------------------------- #
#  adjoint_solve_kkt!  —  reverse solve (VJP / sensitivity)
# ---------------------------------------------------------------------------- #

function adjoint_solve_kkt!(kkt::SparseActiveSetKKTSystem{T}, w::AbstractKKTVector) where T
    n    = size(kkt.hess_com, 1)
    ns   = length(kkt.ind_ineq)
    ns_a = kkt.ns_active
    m    = length(kkt.du_diag)

    _adjoint_finish_bounds!(kkt, w)

    pd = primal_dual(w)
    wr = kkt.work_reduced

    copyto!(view(wr, 1:n), view(pd, 1:n))
    wr[n+1:n+ns_a]   .= pd[kkt.ind_active_slack_pd]
    wr[n+ns_a+1:end]  .= pd[kkt.ind_active_dual_pd]

    solve_linear_system!(kkt.linear_solver, wr)

    copyto!(view(pd, 1:n), view(wr, 1:n))
    fill!(view(pd, n+1:n+ns), zero(T))
    pd[kkt.ind_active_slack_pd] .= view(wr, n+1:n+ns_a)
    fill!(view(pd, n+ns+1:n+ns+m), zero(T))
    pd[kkt.ind_active_dual_pd]  .= view(wr, n+ns_a+1:length(wr))

    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function mul!(
    w::AbstractKKTVector{T},
    kkt::SparseActiveSetKKTSystem{T},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where T
    n   = size(kkt.hess_com, 1)
    ns  = length(kkt.ind_ineq)
    n_a = kkt.n_active

    wx = view(primal(w), 1:n)
    ws = view(primal(w), n+1:n+ns)
    wy = dual(w)

    xx = view(primal(x), 1:n)
    xs = view(primal(x), n+1:n+ns)
    xy = dual(x)

    # H * xx → wx
    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)

    # J_a' * xy_active → wx,  J_a * xx → wy at active positions
    wy .= beta .* wy
    if n_a > 0
        kkt.work_n_a .= xy[kkt.active_cons]
        mul!(wx, kkt.jac_active_com', kkt.work_n_a, alpha, one(T))
        mul!(kkt.work_n_a, kkt.jac_active_com, xx)
        wy[kkt.active_cons] .+= alpha .* kkt.work_n_a
    end

    ws .= beta .* ws
    ws[kkt.active_slack_map] .-= alpha .* xy[kkt.ind_slack_cons]
    wy[kkt.ind_slack_cons]   .-= alpha .* xs[kkt.active_slack_map]
    MadNLP._kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower,
                    kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::SparseActiveSetKKTSystem{T},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where T
    n   = size(kkt.hess_com, 1)
    ns  = length(kkt.ind_ineq)
    n_a = kkt.n_active

    wx = view(primal(w), 1:n)
    ws = view(primal(w), n+1:n+ns)
    wy = dual(w)

    xx = view(primal(x), 1:n)
    xs = view(primal(x), n+1:n+ns)
    xy = dual(x)

    # H * xx (symmetric)
    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)  # FIXME: symul?

    # J_a' * xy_active → wx,  J_a * xx → wy at active positions
    wy .= beta .* wy
    if n_a > 0
        kkt.work_n_a .= xy[kkt.active_cons]
        mul!(wx, kkt.jac_active_com', kkt.work_n_a, alpha, one(T))
        mul!(kkt.work_n_a, kkt.jac_active_com, xx)
        wy[kkt.active_cons] .+= alpha .* kkt.work_n_a
    end

    ws .= beta .* ws
    ws[kkt.active_slack_map] .-= alpha .* xy[kkt.ind_slack_cons]
    wy[kkt.ind_slack_cons]   .-= alpha .* xs[kkt.active_slack_map]
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower,
                     kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function MadNLP.create_kkt_system(
    ::Type{SparseActiveSetKKTSystem},
    cb::AbstractCallback{T, VT},
    linear_solver::Type;
    opt_linear_solver = MadNLP.default_options(linear_solver),
    solver = nothing,
    sigma = T(0.75),
    kwargs...,
) where {T, VT}
    isnothing(solver) && error("SparseActiveSetKKTSystem requires `solver` keyword argument")

    n   = cb.nvar
    m   = cb.ncon
    ns  = length(cb.ind_ineq)
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    active_cons = identify_active_set(solver; sigma)
    n_a = length(active_cons)

    active_slack_map, ind_ineq_active, ns_a = _active_slack_mapping(active_cons, cb.ind_ineq, m)

    # -- Sparsity patterns (from MadNLP callbacks) --
    jac_sp_I = create_array(cb, Int32, cb.nnzj)
    jac_sp_J = create_array(cb, Int32, cb.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb, jac_sp_I, jac_sp_J)

    hess_sp_I, hess_sp_J = MadNLP.build_hessian_structure(cb, MadNLP.ExactHessian)
    MadNLP.force_lower_triangular!(hess_sp_I, hess_sp_J)

    nnzh = length(hess_sp_I)
    nnzj = length(jac_sp_I)

    # -- Hessian COO --
    hess = fill!(VT(undef, nnzh), zero(T))
    _hI = create_array(cb, Int32, nnzh); copyto!(_hI, hess_sp_I)
    _hJ = create_array(cb, Int32, nnzh); copyto!(_hJ, hess_sp_J)
    hess_raw = SparseMatrixCOO(n, n, _hI, _hJ, hess)
    hess_com, hess_csc_map = coo_to_csc(hess_raw)

    # -- Full Jacobian COO --
    jac_callback = fill!(VT(undef, nnzj), zero(T))
    _jI = create_array(cb, Int32, nnzj); copyto!(_jI, jac_sp_I)
    _jJ = create_array(cb, Int32, nnzj); copyto!(_jJ, jac_sp_J)
    jac_raw = SparseMatrixCOO(m, n, _jI, _jJ, jac_callback)
    jac_com, jac_csc_map = coo_to_csc(jac_raw)

    # -- Active Jacobian filtering (vectorized) --
    cons_to_active = fill!(create_array(cb, Int, m), 0)
    _seq = create_array(cb, Int, n_a); _seq .= 1:n_a
    cons_to_active[active_cons] .= _seq

    _jac_rows = create_array(cb, Int, nnzj)
    copyto!(_jac_rows, Vector{Int}(jac_sp_I))
    jac_row_active = cons_to_active[_jac_rows]
    active_jac_mask = jac_row_active .> 0
    jac_active_filter = findall(active_jac_mask)
    nnzj_a = length(jac_active_filter)

    jac_active_V = fill!(VT(undef, nnzj_a), zero(T))
    _jaI = create_array(cb, Int32, nnzj_a)
    _jaI .= Int32.(jac_row_active[jac_active_filter])
    _jaJ = _jJ[jac_active_filter]
    jac_active_raw = SparseMatrixCOO(n_a, n, _jaI, _jaJ, jac_active_V)
    jac_active_com, jac_active_csc_map = coo_to_csc(jac_active_raw)

    # -- Augmented system COO --
    dim = n + ns_a + n_a
    aug_mat_length = n + nnzh + ns_a + nnzj_a + ns_a + n_a

    aug_I = create_array(cb, Int32, aug_mat_length)
    aug_J = create_array(cb, Int32, aug_mat_length)
    aug_V = fill!(VT(undef, aug_mat_length), zero(T))

    off = 0
    aug_I[off+1:off+n] .= Int32.(1:n)
    aug_J[off+1:off+n] .= Int32.(1:n)
    off += n

    aug_I[off+1:off+nnzh] .= _hI
    aug_J[off+1:off+nnzh] .= _hJ
    off += nnzh

    if ns_a > 0
        aug_I[off+1:off+ns_a] .= Int32.(n+1:n+ns_a)
        aug_J[off+1:off+ns_a] .= Int32.(n+1:n+ns_a)
    end
    off += ns_a

    if nnzj_a > 0
        aug_I[off+1:off+nnzj_a] .= Int32(n + ns_a) .+ _jaI
        aug_J[off+1:off+nnzj_a] .= _jaJ
    end
    off += nnzj_a

    if ns_a > 0
        aug_I[off+1:off+ns_a] .= Int32(n + ns_a) .+ Int32.(ind_ineq_active)
        aug_J[off+1:off+ns_a] .= Int32.(n+1:n+ns_a)
        aug_V[off+1:off+ns_a] .= -one(T)
    end
    off += ns_a

    if n_a > 0
        aug_I[off+1:off+n_a] .= Int32.(n + ns_a .+ (1:n_a))
        aug_J[off+1:off+n_a] .= Int32.(n + ns_a .+ (1:n_a))
    end

    aug_raw = SparseMatrixCOO(dim, dim, aug_I, aug_J, aug_V)
    aug_com, aug_csc_map = coo_to_csc(aug_raw)

    # -- Work vectors --
    reg     = fill!(VT(undef, n + ns), zero(T))
    pr_diag = fill!(VT(undef, n + ns), zero(T))
    du_diag = fill!(VT(undef, m),      zero(T))
    l_diag  = fill!(VT(undef, nlb), one(T))
    u_diag  = fill!(VT(undef, nub), one(T))
    l_lower = fill!(VT(undef, nlb), zero(T))
    u_lower = fill!(VT(undef, nub), zero(T))

    ind_active_slack_pd = n .+ active_slack_map
    ind_active_dual_pd  = (n + ns) .+ active_cons
    ind_slack_cons      = active_cons[ind_ineq_active]
    work_n_a            = VT(undef, n_a)
    work_reduced        = VT(undef, dim)

    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver)
    quasi_newton = MadNLP.ExactHessian{T, VT}()

    return SparseActiveSetKKTSystem(
        hess, hess_raw, hess_com, hess_csc_map,
        jac_callback, jac_raw, jac_com, jac_csc_map,
        jac_active_raw, jac_active_com, jac_active_csc_map, jac_active_filter,
        aug_raw, aug_com, aug_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        cb.ind_ineq, cb.ind_lb, cb.ind_ub,
        active_cons, active_slack_map, ind_ineq_active,
        n_a, ns_a,
        ind_active_slack_pd, ind_active_dual_pd, ind_slack_cons, work_n_a,
        work_reduced,
        nnzh, nnzj_a,
        _linear_solver,
        Dict{Symbol, Any}(),
    )
end
