struct DenseActiveSetKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
    LS,
    VI <: AbstractVector{Int},
} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, QN}
    hess::MT
    jac::MT
    quasi_newton::QN

    # -- Full-dim KKT fields
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    diag_hess::VT

    # -- Reduced augmented system (dim = n + ns_a + n_a)
    aug_com::MT

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
    jac_active::MT
    ind_active_slack_pd::VI  # n .+ active_slack_map
    ind_active_dual_pd::VI   # (n+ns) .+ active_cons
    ind_slack_cons::VI       # active_cons[ind_ineq_active]
    work_n_a::VT

    # -- Precomputed linear indices for -I coupling scatter in aug_com
    ind_coupling_32::VI
    ind_coupling_23::VI

    # -- Work vector for reduced solve
    work_reduced::VT

    # -- Linear solver
    linear_solver::LS
    etc::Dict{Symbol, Any}
end

num_variables(kkt::DenseActiveSetKKTSystem) = length(kkt.pr_diag)

function MadNLP.get_jacobian(kkt::DenseActiveSetKKTSystem)
    n = size(kkt.hess, 1)
    return view(kkt.jac, :, 1:n)
end

function MadNLP.compress_hessian!(kkt::DenseActiveSetKKTSystem)
    MadNLP.diag!(kkt.diag_hess, kkt.hess)
end

function MadNLP.compress_jacobian!(kkt::DenseActiveSetKKTSystem)
    n = size(kkt.hess, 1)
    kkt.jac_active .= kkt.jac[kkt.active_cons, 1:n]
    return
end

function MadNLP.is_inertia_correct(kkt::DenseActiveSetKKTSystem, num_pos, num_zero, num_neg)
    n = size(kkt.hess, 1)
    return (num_zero == 0) && (num_pos == n + kkt.ns_active) && (num_neg == kkt.n_active)
end

function MadNLP.regularize_diagonal!(kkt::DenseActiveSetKKTSystem, primal, dual)
    kkt.reg .+= primal
    kkt.pr_diag .+= primal
    kkt.du_diag .-= dual
end

MadNLP.factorize_kkt!(kkt::DenseActiveSetKKTSystem) = MadNLP.factorize!(kkt.linear_solver)

function MadNLP.initialize!(kkt::DenseActiveSetKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    return
end

function mul!(
    w::AbstractKKTVector{T},
    kkt::DenseActiveSetKKTSystem{T},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where T
    n  = size(kkt.hess, 1)
    ns = length(kkt.ind_ineq)
    n_a = kkt.n_active

    wx = view(primal(w), 1:n)
    ws = view(primal(w), n+1:n+ns)
    wy = dual(w)

    xx = view(primal(x), 1:n)
    xs = view(primal(x), n+1:n+ns)
    xy = dual(x)

    MadNLP._symv!('L', alpha, kkt.hess, xx, beta, wx)

    wy .= beta .* wy
    if n_a > 0
        kkt.work_n_a .= xy[kkt.active_cons]
        mul!(wx, kkt.jac_active', kkt.work_n_a, alpha, one(T))
        mul!(kkt.work_n_a, kkt.jac_active, xx)
        wy[kkt.active_cons] .+= alpha .* kkt.work_n_a
    end

    ws .= beta .* ws
    ws[kkt.active_slack_map] .-= alpha .* xy[kkt.ind_slack_cons]
    wy[kkt.ind_slack_cons]   .-= alpha .* xs[kkt.active_slack_map]
    MadNLP._kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower,
                    kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function _build_dense_active_set_kkt_system!(kkt::DenseActiveSetKKTSystem{T}) where T
    n    = size(kkt.hess, 1)
    n_a  = kkt.n_active
    ns_a = kkt.ns_active
    aug  = kkt.aug_com

    fill!(aug, zero(T))

    view(aug, 1:n, 1:n) .= kkt.hess
    _diag = diagind(aug)
    aug[_diag[1:n]] .+= view(kkt.pr_diag, 1:n)

    if ns_a > 0
        aug[_diag[n+1:n+ns_a]] .= kkt.pr_diag[kkt.ind_active_slack_pd]
    end

    if n_a > 0
        view(aug, n+ns_a+1:n+ns_a+n_a, 1:n) .= kkt.jac_active
        view(aug, 1:n, n+ns_a+1:n+ns_a+n_a) .= kkt.jac_active'
    end

    if ns_a > 0
        aug[kkt.ind_coupling_32] .= -one(T)
        aug[kkt.ind_coupling_23] .= -one(T)
    end

    if n_a > 0
        aug[_diag[n+ns_a+1:n+ns_a+n_a]] .= kkt.du_diag[kkt.active_cons]
    end

    return
end

function MadNLP.build_kkt!(kkt::DenseActiveSetKKTSystem{T}) where T
    _build_dense_active_set_kkt_system!(kkt)
    return
end

function MadNLP.solve_kkt!(kkt::DenseActiveSetKKTSystem{T}, w::AbstractKKTVector) where T
    n    = size(kkt.hess, 1)
    ns   = length(kkt.ind_ineq)
    ns_a = kkt.ns_active

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
    fill!(view(pd, n+ns+1:n+ns+size(kkt.jac, 1)), zero(T))
    pd[kkt.ind_active_dual_pd]  .= view(wr, n+ns_a+1:length(wr))

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

function adjoint_solve_kkt!(kkt::DenseActiveSetKKTSystem{T}, w::AbstractKKTVector) where T
    n    = size(kkt.hess, 1)
    ns   = length(kkt.ind_ineq)
    ns_a = kkt.ns_active

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
    fill!(view(pd, n+ns+1:n+ns+size(kkt.jac, 1)), zero(T))
    pd[kkt.ind_active_dual_pd]  .= view(wr, n+ns_a+1:length(wr))

    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::DenseActiveSetKKTSystem{T},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where T
    n  = size(kkt.hess, 1)
    ns = length(kkt.ind_ineq)
    n_a = kkt.n_active

    wx = view(primal(w), 1:n)
    ws = view(primal(w), n+1:n+ns)
    wy = dual(w)

    xx = view(primal(x), 1:n)
    xs = view(primal(x), n+1:n+ns)
    xy = dual(x)

    MadNLP._symv!('L', alpha, kkt.hess, xx, beta, wx)

    wy .= beta .* wy
    if n_a > 0
        kkt.work_n_a .= xy[kkt.active_cons]
        mul!(wx, kkt.jac_active', kkt.work_n_a, alpha, one(T))
        mul!(kkt.work_n_a, kkt.jac_active, xx)
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
    ::Type{DenseActiveSetKKTSystem},
    cb::AbstractCallback{T, VT},
    linear_solver::Type;
    opt_linear_solver = MadNLP.default_options(linear_solver),
    solver = nothing,
    sigma = T(0.75),
    kwargs...,
) where {T, VT}
    isnothing(solver) && error("DenseActiveSetKKTSystem requires `solver` keyword argument")

    n   = cb.nvar
    m   = cb.ncon
    ns  = length(cb.ind_ineq)
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    active_cons = identify_active_set(solver; sigma)
    n_a = length(active_cons)

    active_slack_map, ind_ineq_active, ns_a = _active_slack_mapping(active_cons, cb.ind_ineq, m)

    hess = fill!(create_array(cb, n, n), zero(T))
    jac  = fill!(create_array(cb, m, n), zero(T))

    reg       = fill!(VT(undef, n + ns), zero(T))
    pr_diag   = fill!(VT(undef, n + ns), zero(T))
    du_diag   = fill!(VT(undef, m),      zero(T))
    diag_hess = fill!(VT(undef, n),      zero(T))
    l_diag    = fill!(VT(undef, nlb), one(T))
    u_diag    = fill!(VT(undef, nub), one(T))
    l_lower   = fill!(VT(undef, nlb), zero(T))
    u_lower   = fill!(VT(undef, nub), zero(T))

    dim = n + ns_a + n_a
    aug_com = fill!(create_array(cb, dim, dim), zero(T))

    jac_active          = fill!(create_array(cb, n_a, n), zero(T))
    ind_active_slack_pd = n .+ active_slack_map
    ind_active_dual_pd  = (n + ns) .+ active_cons
    ind_slack_cons      = active_cons[ind_ineq_active]
    work_n_a            = VT(undef, n_a)

    # column-major linear indices: (col-1)*dim + row
    ks = create_array(cb, Int, ns_a)
    ks .= 1:ns_a
    ind_coupling_32 = (n .- 1 .+ ks) .* dim .+ (n + ns_a) .+ ind_ineq_active
    ind_coupling_23 = (n + ns_a .- 1 .+ ind_ineq_active) .* dim .+ n .+ ks

    work_reduced = VT(undef, dim)

    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver)
    quasi_newton = MadNLP.ExactHessian{T, VT}()

    return DenseActiveSetKKTSystem(
        hess, jac, quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        diag_hess, aug_com,
        cb.ind_ineq, cb.ind_lb, cb.ind_ub,
        active_cons, active_slack_map, ind_ineq_active,
        n_a, ns_a,
        jac_active, ind_active_slack_pd, ind_active_dual_pd,
        ind_slack_cons, work_n_a,
        ind_coupling_32, ind_coupling_23,
        work_reduced,
        _linear_solver,
        Dict{Symbol, Any}(),
    )
end
