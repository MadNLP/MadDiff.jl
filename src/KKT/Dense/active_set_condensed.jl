struct DenseCondensedActiveSetKKTSystem{
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

    # -- Condensed system (dim = n + n_eq_a)
    aug_com::MT
    diag_buffer::VT           # ns_a: Schur complement diagonal D
    jac_ineq_active::MT       # ns_a × n: scaled ineq Jacobian (√D · J) for build

    # -- Full-dim index sets
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # -- Active-set mappings
    active_cons::VI           # active constraint indices into 1:m (length n_a)
    active_slack_map::VI      # active slack indices into 1:ns (length ns_a)
    n_active::Int
    ns_active::Int
    n_eq_active::Int

    # -- Active equality/inequality partition
    active_eq_cons::VI        # indices into 1:m for active equalities
    active_ineq_cons::VI      # indices into 1:m for active inequalities

    # -- Gather/scatter
    ind_active_slack_pd::VI   # n .+ active_slack_map
    ind_active_dual_pd::VI    # (n+ns) .+ active_cons
    ind_active_ineq_dual_pd::VI  # (n+ns) .+ active_ineq_cons
    ind_active_eq_dual_pd::VI    # (n+ns) .+ active_eq_cons

    work_reduced::VT          # dim = n + n_eq_a
    buffer_ineq::VT           # ns_a (saved condensation buffer)
    buffer2_ineq::VT          # ns_a (back-sub temp)
    r_s_save::VT              # ns_a (saved r_s for back-sub)
    buffer_m::VT              # m (for scatter/gather with full Jacobian)

    # -- Linear solver
    linear_solver::LS
    etc::Dict{Symbol, Any}
end

num_variables(kkt::DenseCondensedActiveSetKKTSystem) = length(kkt.pr_diag)

function MadNLP.get_jacobian(kkt::DenseCondensedActiveSetKKTSystem)
    n = size(kkt.hess, 1)
    return view(kkt.jac, :, 1:n)
end

function MadNLP.compress_hessian!(kkt::DenseCondensedActiveSetKKTSystem)
    MadNLP.diag!(kkt.diag_hess, kkt.hess)
end

function MadNLP.compress_jacobian!(kkt::DenseCondensedActiveSetKKTSystem)
    return
end

function MadNLP.is_inertia_correct(kkt::DenseCondensedActiveSetKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_neg == kkt.n_eq_active)
end

function MadNLP.regularize_diagonal!(kkt::DenseCondensedActiveSetKKTSystem, primal, dual)
    kkt.reg .+= primal
    kkt.pr_diag .+= primal
    kkt.du_diag .-= dual
end

MadNLP.factorize_kkt!(kkt::DenseCondensedActiveSetKKTSystem) = MadNLP.factorize!(kkt.linear_solver)

function MadNLP.initialize!(kkt::DenseCondensedActiveSetKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    return
end

# --------------------------------------------------------------------------- #
#  build_kkt!
# --------------------------------------------------------------------------- #
function MadNLP.build_kkt!(kkt::DenseCondensedActiveSetKKTSystem{T}) where T
    n      = size(kkt.hess, 1)
    ns_a   = kkt.ns_active
    n_eq_a = kkt.n_eq_active
    aug    = kkt.aug_com

    fill!(aug, zero(T))

    # Schur complement diagonal: D = Σ_sa / (1 - Σ_d_ineq · Σ_sa)
    if ns_a > 0
        Σ_sa = kkt.pr_diag[kkt.ind_active_slack_pd]
        Σ_d_ineq = kkt.du_diag[kkt.active_ineq_cons]
        kkt.diag_buffer .= Σ_sa ./ (one(T) .- Σ_d_ineq .* Σ_sa)

        # √D · J_ineq_a  (GPU kernel: _build_jacobian_condensed_kernel!)
        MadNLP._build_ineq_jac!(kkt.jac_ineq_active, kkt.jac, kkt.diag_buffer, kkt.active_ineq_cons, n, ns_a)

        # J_ineq_a' D J_ineq_a  into top-left block
        W = n_eq_a > 0 ? view(aug, 1:n, 1:n) : aug
        mul!(W, kkt.jac_ineq_active', kkt.jac_ineq_active)
    end

    # H + Σx + equality block  (GPU kernel: _build_condensed_kkt_system_kernel!)
    MadNLP._build_condensed_kkt_system!(aug, kkt.hess, kkt.jac, kkt.pr_diag, kkt.du_diag, kkt.active_eq_cons, n, n_eq_a)

    return
end

# --------------------------------------------------------------------------- #
#  _condensed_active_set_solve!  — shared by forward and adjoint
# --------------------------------------------------------------------------- #
function _condensed_active_set_solve!(kkt::DenseCondensedActiveSetKKTSystem{T}, pd, n, ns, m) where T
    ns_a   = kkt.ns_active
    n_eq_a = kkt.n_eq_active
    wr     = kkt.work_reduced

    # Gather r_x
    copyto!(view(wr, 1:n), view(pd, 1:n))

    # Condensation: absorb active inequality constraints
    if ns_a > 0
        Σ_sa = view(kkt.pr_diag, kkt.ind_active_slack_pd)
        r_s  = pd[kkt.ind_active_slack_pd]
        r_y  = pd[kkt.ind_active_ineq_dual_pd]

        # Save r_s for back-substitution
        kkt.r_s_save .= r_s

        # buffer = D · (r_y + r_s / Σ_sa)
        kkt.buffer_ineq .= kkt.diag_buffer .* (r_y .+ r_s ./ Σ_sa)

        # r_x += J' · scatter(buffer_ineq)
        fill!(kkt.buffer_m, zero(T))
        kkt.buffer_m[kkt.active_ineq_cons] .= kkt.buffer_ineq
        mul!(view(wr, 1:n), kkt.jac', kkt.buffer_m, one(T), one(T))
    end

    # Gather active equality dual RHS
    if n_eq_a > 0
        view(wr, n+1:n+n_eq_a) .= pd[kkt.ind_active_eq_dual_pd]
    end

    # Solve condensed system
    solve_linear_system!(kkt.linear_solver, wr)

    # Scatter Δx
    copyto!(view(pd, 1:n), view(wr, 1:n))

    # Zero inactive slacks and duals
    fill!(view(pd, n+1:n+ns), zero(T))
    fill!(view(pd, n+ns+1:n+ns+m), zero(T))

    # Back-substitute active inequalities
    if ns_a > 0
        Σ_sa = view(kkt.pr_diag, kkt.ind_active_slack_pd)

        # buffer_m = J · Δx, then gather active ineq rows
        mul!(kkt.buffer_m, kkt.jac, view(wr, 1:n))
        kkt.buffer2_ineq .= kkt.buffer_m[kkt.active_ineq_cons]

        # Δy_ineq = -buffer + D · J_ineq_a · Δx
        kkt.buffer2_ineq .= .-kkt.buffer_ineq .+ kkt.diag_buffer .* kkt.buffer2_ineq

        # Scatter Δy_ineq
        pd[kkt.ind_active_ineq_dual_pd] .= kkt.buffer2_ineq

        # Δs = (r_s + Δy_ineq) / Σ_sa
        pd[kkt.ind_active_slack_pd] .= (kkt.r_s_save .+ kkt.buffer2_ineq) ./ Σ_sa
    end

    # Scatter active equality duals
    if n_eq_a > 0
        pd[kkt.ind_active_eq_dual_pd] .= view(wr, n+1:n+n_eq_a)
    end

    return
end

# --------------------------------------------------------------------------- #
#  solve_kkt!
# --------------------------------------------------------------------------- #
function MadNLP.solve_kkt!(kkt::DenseCondensedActiveSetKKTSystem{T}, w::AbstractKKTVector) where T
    n  = size(kkt.hess, 1)
    ns = length(kkt.ind_ineq)
    m  = size(kkt.jac, 1)

    MadNLP.reduce_rhs!(kkt, w)
    _condensed_active_set_solve!(kkt, primal_dual(w), n, ns, m)
    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

# --------------------------------------------------------------------------- #
#  adjoint_solve_kkt!
# --------------------------------------------------------------------------- #
function adjoint_solve_kkt!(kkt::DenseCondensedActiveSetKKTSystem{T}, w::AbstractKKTVector) where T
    n  = size(kkt.hess, 1)
    ns = length(kkt.ind_ineq)
    m  = size(kkt.jac, 1)

    _adjoint_finish_bounds!(kkt, w)
    _condensed_active_set_solve!(kkt, primal_dual(w), n, ns, m)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

# --------------------------------------------------------------------------- #
#  mul!  — unreduced active-set matrix-vector product (for iterative refinement)
# --------------------------------------------------------------------------- #
function mul!(
    w::AbstractKKTVector{T},
    kkt::DenseCondensedActiveSetKKTSystem{T},
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
        fill!(kkt.buffer_m, zero(T))
        kkt.buffer_m[kkt.active_cons] .= xy[kkt.active_cons]
        mul!(wx, kkt.jac', kkt.buffer_m, alpha, one(T))
        mul!(kkt.buffer_m, kkt.jac, xx)
        wy[kkt.active_cons] .+= alpha .* kkt.buffer_m[kkt.active_cons]
    end

    ws .= beta .* ws
    ws[kkt.active_slack_map] .-= alpha .* xy[kkt.active_ineq_cons]
    wy[kkt.active_ineq_cons]   .-= alpha .* xs[kkt.active_slack_map]
    MadNLP._kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower,
                    kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

# --------------------------------------------------------------------------- #
#  adjoint_mul!
# --------------------------------------------------------------------------- #
function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::DenseCondensedActiveSetKKTSystem{T},
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
        fill!(kkt.buffer_m, zero(T))
        kkt.buffer_m[kkt.active_cons] .= xy[kkt.active_cons]
        mul!(wx, kkt.jac', kkt.buffer_m, alpha, one(T))
        mul!(kkt.buffer_m, kkt.jac, xx)
        wy[kkt.active_cons] .+= alpha .* kkt.buffer_m[kkt.active_cons]
    end

    ws .= beta .* ws
    ws[kkt.active_slack_map] .-= alpha .* xy[kkt.active_ineq_cons]
    wy[kkt.active_ineq_cons]   .-= alpha .* xs[kkt.active_slack_map]
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower,
                     kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

# --------------------------------------------------------------------------- #
#  Constructor
# --------------------------------------------------------------------------- #
function MadNLP.create_kkt_system(
    ::Type{DenseCondensedActiveSetKKTSystem},
    cb::AbstractCallback{T, VT},
    linear_solver::Type;
    opt_linear_solver = MadNLP.default_options(linear_solver),
    solver = nothing,
    sigma = T(0.75),
    kwargs...,
) where {T, VT}
    isnothing(solver) && error("DenseCondensedActiveSetKKTSystem requires `solver` keyword argument")

    n   = cb.nvar
    m   = cb.ncon
    ns  = length(cb.ind_ineq)
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    active_cons = identify_active_set(solver; sigma)
    n_a = length(active_cons)

    active_slack_map, ind_ineq_active, ns_a = _active_slack_mapping(active_cons, cb.ind_ineq, m)
    active_eq_cons  = cb.ind_eq
    active_ineq_cons = active_cons[ind_ineq_active]
    n_eq_a = length(active_eq_cons)

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

    dim = n + n_eq_a
    aug_com = fill!(create_array(cb, dim, dim), zero(T))

    diag_buffer      = VT(undef, ns_a)
    jac_ineq_active  = fill!(create_array(cb, ns_a, n), zero(T))

    ind_active_slack_pd     = n .+ active_slack_map
    ind_active_dual_pd      = (n + ns) .+ active_cons
    ind_active_ineq_dual_pd = (n + ns) .+ active_ineq_cons
    ind_active_eq_dual_pd   = (n + ns) .+ active_eq_cons

    work_reduced  = VT(undef, dim)
    buffer_ineq   = VT(undef, ns_a)
    buffer2_ineq  = VT(undef, ns_a)
    r_s_save_buf  = VT(undef, ns_a)
    buffer_m      = VT(undef, m)

    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver)
    quasi_newton = MadNLP.ExactHessian{T, VT}()

    return DenseCondensedActiveSetKKTSystem(
        hess, jac, quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        diag_hess,
        aug_com, diag_buffer, jac_ineq_active,
        cb.ind_ineq, cb.ind_lb, cb.ind_ub,
        active_cons, active_slack_map,
        n_a, ns_a, n_eq_a,
        active_eq_cons, active_ineq_cons,
        ind_active_slack_pd, ind_active_dual_pd,
        ind_active_ineq_dual_pd, ind_active_eq_dual_pd,
        work_reduced, buffer_ineq, buffer2_ineq, r_s_save_buf,
        buffer_m,
        _linear_solver,
        Dict{Symbol, Any}(),
    )
end
