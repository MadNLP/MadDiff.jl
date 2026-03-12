struct SparseCondensedActiveSetKKTSystem{
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

    # -- Active Jacobian (n_a×n) for mul!/back-sub
    jac_active_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    jac_active_com::MT
    jac_active_csc_map::Union{Nothing, VI}
    jac_active_filter::VI

    # -- Active ineq Jacobian transpose (n×ns_a) for Schur complement
    jac_ineq_active_t_coo::SparseMatrixCOO{T, Int32, VT, VI32}
    jac_ineq_active_t_csc::MT
    jac_ineq_active_t_csc_map::Union{Nothing, VI}
    ineq_jac_active_filter::VI

    # -- Condensed augmented system (dim = n + n_eq_a)
    aug_com::MT
    diag_buffer::VT           # ns_a
    dptr::Any
    hptr::Any
    jptr::Any

    # -- Active eq Jacobian precomputed nzval indices in aug_com
    eq_jac_nzval_indices::VI  # nzval indices in aug_com for eq Jacobian entries
    eq_jac_vals_filter::VI    # indices into jac_callback for eq Jacobian entries
    eq_aug_diag_indices::VI   # nzval indices in aug_com for eq dual diagonal

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
    active_cons::VI
    active_slack_map::VI
    n_active::Int
    ns_active::Int
    n_eq_active::Int

    # -- Partition
    active_eq_cons::VI
    active_ineq_cons::VI

    # -- Gather/scatter
    ind_active_slack_pd::VI
    ind_active_dual_pd::VI
    ind_active_ineq_dual_pd::VI
    ind_active_eq_dual_pd::VI
    work_n_a::VT

    # -- Condensed solve buffers
    buffer_ineq::VT           # ns_a
    buffer2_ineq::VT          # ns_a
    r_s_save::VT              # ns_a
    work_reduced::VT          # n + n_eq_a

    _nnzh::Int

    # -- Linear solver
    linear_solver::LS
    etc::Dict{Symbol, Any}
end

num_variables(kkt::SparseCondensedActiveSetKKTSystem) = length(kkt.pr_diag)

MadNLP.get_jacobian(kkt::SparseCondensedActiveSetKKTSystem) = kkt.jac_callback

function MadNLP.compress_hessian!(kkt::SparseCondensedActiveSetKKTSystem)
    transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function MadNLP.compress_jacobian!(kkt::SparseCondensedActiveSetKKTSystem)
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
    # Active Jacobian (all active constraints)
    kkt.jac_active_raw.V .= kkt.jac_callback[kkt.jac_active_filter]
    transfer!(kkt.jac_active_com, kkt.jac_active_raw, kkt.jac_active_csc_map)
    # Active ineq Jacobian transpose
    kkt.jac_ineq_active_t_coo.V .= kkt.jac_callback[kkt.ineq_jac_active_filter]
    transfer!(kkt.jac_ineq_active_t_csc, kkt.jac_ineq_active_t_coo, kkt.jac_ineq_active_t_csc_map)
end

function MadNLP.is_inertia_correct(kkt::SparseCondensedActiveSetKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_neg == kkt.n_eq_active)
end

function MadNLP.regularize_diagonal!(kkt::SparseCondensedActiveSetKKTSystem, primal, dual)
    kkt.reg .+= primal
    kkt.pr_diag .+= primal
    kkt.du_diag .-= dual
end

MadNLP.factorize_kkt!(kkt::SparseCondensedActiveSetKKTSystem) = MadNLP.factorize!(kkt.linear_solver)

function MadNLP.initialize!(kkt::SparseCondensedActiveSetKKTSystem{T}) where T
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

# --------------------------------------------------------------------------- #
#  build_kkt!
# --------------------------------------------------------------------------- #
function MadNLP.build_kkt!(kkt::SparseCondensedActiveSetKKTSystem{T}) where T
    n      = size(kkt.hess_com, 1)
    ns_a   = kkt.ns_active
    n_eq_a = kkt.n_eq_active

    # Schur complement diagonal
    if ns_a > 0
        Σ_sa = kkt.pr_diag[kkt.ind_active_slack_pd]
        Σ_d_ineq = kkt.du_diag[kkt.active_ineq_cons]
        kkt.diag_buffer .= Σ_sa ./ (one(T) .- Σ_d_ineq .* Σ_sa)
    end

    # Build condensed n×n block via symbolic pointers
    MadNLP._build_condensed_aug_coord!(
        kkt.aug_com, kkt.pr_diag, kkt.hess_com,
        kkt.jac_ineq_active_t_csc, kkt.diag_buffer,
        kkt.dptr, kkt.hptr, kkt.jptr,
    )

    # Add equality Jacobian entries via precomputed nzval indices
    if n_eq_a > 0
        nzv = nonzeros(kkt.aug_com)
        nzv[kkt.eq_jac_nzval_indices] .+= kkt.jac_callback[kkt.eq_jac_vals_filter]
        nzv[kkt.eq_aug_diag_indices] .+= kkt.du_diag[kkt.active_eq_cons]
    end

    return
end

# --------------------------------------------------------------------------- #
#  solve_kkt!
# --------------------------------------------------------------------------- #
function MadNLP.solve_kkt!(kkt::SparseCondensedActiveSetKKTSystem{T}, w::AbstractKKTVector) where T
    n  = size(kkt.hess_com, 1)
    ns = length(kkt.ind_ineq)
    m  = length(kkt.du_diag)

    MadNLP.reduce_rhs!(kkt, w)
    _sparse_condensed_active_set_solve!(kkt, primal_dual(w), n, ns, m)
    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

# --------------------------------------------------------------------------- #
#  adjoint_solve_kkt!
# --------------------------------------------------------------------------- #
function adjoint_solve_kkt!(kkt::SparseCondensedActiveSetKKTSystem{T}, w::AbstractKKTVector) where T
    n  = size(kkt.hess_com, 1)
    ns = length(kkt.ind_ineq)
    m  = length(kkt.du_diag)

    _adjoint_finish_bounds!(kkt, w)
    _sparse_condensed_active_set_solve!(kkt, primal_dual(w), n, ns, m)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function _sparse_condensed_active_set_solve!(kkt::SparseCondensedActiveSetKKTSystem{T}, pd, n, ns, m) where T
    ns_a   = kkt.ns_active
    n_eq_a = kkt.n_eq_active
    wr     = kkt.work_reduced

    # Gather r_x
    copyto!(view(wr, 1:n), view(pd, 1:n))

    # Condensation of active inequalities
    if ns_a > 0
        Σ_sa = view(kkt.pr_diag, kkt.ind_active_slack_pd)
        r_s  = pd[kkt.ind_active_slack_pd]
        r_y  = pd[kkt.ind_active_ineq_dual_pd]

        kkt.r_s_save .= r_s
        kkt.buffer_ineq .= kkt.diag_buffer .* (r_y .+ r_s ./ Σ_sa)

        # r_x += Jt_ineq · buffer  (Jt_ineq is n×ns_a, so Jt * buffer gives n-vec)
        mul!(view(wr, 1:n), kkt.jac_ineq_active_t_csc, kkt.buffer_ineq, one(T), one(T))
    end

    # Gather active equality dual RHS
    if n_eq_a > 0
        view(wr, n+1:n+n_eq_a) .= pd[kkt.ind_active_eq_dual_pd]
    end

    # Solve
    solve_linear_system!(kkt.linear_solver, wr)

    # Scatter Δx
    copyto!(view(pd, 1:n), view(wr, 1:n))

    # Zero inactive
    fill!(view(pd, n+1:n+ns), zero(T))
    fill!(view(pd, n+ns+1:n+ns+m), zero(T))

    # Back-substitute active inequalities
    if ns_a > 0
        Σ_sa = view(kkt.pr_diag, kkt.ind_active_slack_pd)

        # buffer2 = Jt_ineq' · Δx  (= J_ineq_a · Δx)
        mul!(kkt.buffer2_ineq, kkt.jac_ineq_active_t_csc', view(wr, 1:n))

        # Δy_ineq = -buffer + D · buffer2
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
#  mul! / adjoint_mul!  — unreduced active-set KKT (same as SparseActiveSetKKT)
# --------------------------------------------------------------------------- #
function mul!(
    w::AbstractKKTVector{T},
    kkt::SparseCondensedActiveSetKKTSystem{T},
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

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)

    wy .= beta .* wy
    if n_a > 0
        kkt.work_n_a .= xy[kkt.active_cons]
        mul!(wx, kkt.jac_active_com', kkt.work_n_a, alpha, one(T))
        mul!(kkt.work_n_a, kkt.jac_active_com, xx)
        wy[kkt.active_cons] .+= alpha .* kkt.work_n_a
    end

    ws .= beta .* ws
    ws[kkt.active_slack_map] .-= alpha .* xy[kkt.active_ineq_cons]
    wy[kkt.active_ineq_cons]   .-= alpha .* xs[kkt.active_slack_map]
    MadNLP._kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower,
                    kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::SparseCondensedActiveSetKKTSystem{T},
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

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)

    wy .= beta .* wy
    if n_a > 0
        kkt.work_n_a .= xy[kkt.active_cons]
        mul!(wx, kkt.jac_active_com', kkt.work_n_a, alpha, one(T))
        mul!(kkt.work_n_a, kkt.jac_active_com, xx)
        wy[kkt.active_cons] .+= alpha .* kkt.work_n_a
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
    ::Type{SparseCondensedActiveSetKKTSystem},
    cb::AbstractCallback{T, VT},
    linear_solver::Type;
    opt_linear_solver = MadNLP.default_options(linear_solver),
    solver = nothing,
    sigma = T(0.75),
    kwargs...,
) where {T, VT}
    isnothing(solver) && error("SparseCondensedActiveSetKKTSystem requires `solver` keyword argument")

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

    # -- Active constraint → active index mapping --
    cons_to_active = fill!(create_array(cb, Int, m), 0)
    _seq = create_array(cb, Int, n_a); _seq .= 1:n_a
    cons_to_active[active_cons] .= _seq

    _jac_rows = create_array(cb, Int, nnzj)
    copyto!(_jac_rows, Vector{Int}(jac_sp_I))

    # -- All-active Jacobian filter --
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

    # -- Active ineq constraint → ineq-active index mapping --
    cons_to_ineq_active = fill!(create_array(cb, Int, m), 0)
    if ns_a > 0
        _iseq = create_array(cb, Int, ns_a); _iseq .= 1:ns_a
        cons_to_ineq_active[active_ineq_cons] .= _iseq
    end

    # -- Active ineq Jacobian transpose COO (rows=variables, cols=ineq_active) --
    jac_row_ineq_active = cons_to_ineq_active[_jac_rows]
    ineq_active_jac_mask = jac_row_ineq_active .> 0
    ineq_jac_active_filter = findall(ineq_active_jac_mask)
    nnzj_ineq_a = length(ineq_jac_active_filter)

    jt_ineq_V = fill!(VT(undef, nnzj_ineq_a), zero(T))
    # Transpose: rows = variable indices (J columns), cols = ineq active indices (J rows)
    _jtI = _jJ[ineq_jac_active_filter]   # variable indices → rows of Jt
    _jtJ = create_array(cb, Int32, nnzj_ineq_a)
    _jtJ .= Int32.(jac_row_ineq_active[ineq_jac_active_filter])  # ineq active indices → cols of Jt
    jac_ineq_active_t_coo = SparseMatrixCOO(n, ns_a, _jtI, _jtJ, jt_ineq_V)
    jac_ineq_active_t_csc, jac_ineq_active_t_csc_map = coo_to_csc(jac_ineq_active_t_coo)

    # -- Condensed augmented system via symbolic build --
    # The n×n condensed block: H + Σx + Jt_ineq · D · Jt_ineq'
    aug_com_base, dptr, hptr, jptr = MadNLP.build_condensed_aug_symbolic(
        hess_com,
        jac_ineq_active_t_csc,
    )

    # If n_eq_a > 0, we need to extend the system to (n + n_eq_a) × (n + n_eq_a)
    if n_eq_a > 0
        # Build equality Jacobian entries for the augmented system
        # Find Jacobian entries for active equality constraints
        cons_to_eq_active = fill!(create_array(cb, Int, m), 0)
        _eseq = create_array(cb, Int, n_eq_a); _eseq .= 1:n_eq_a
        cons_to_eq_active[active_eq_cons] .= _eseq

        jac_row_eq_active = cons_to_eq_active[_jac_rows]
        eq_active_jac_mask = jac_row_eq_active .> 0
        eq_jac_active_filter = findall(eq_active_jac_mask)
        nnzj_eq_a = length(eq_jac_active_filter)

        # Equality Jacobian in augmented system:
        # rows n+1:n+n_eq_a, cols 1:n (lower triangular: row > col, so these go as-is)
        eq_aug_I = create_array(cb, Int32, nnzj_eq_a)
        eq_aug_I .= Int32(n) .+ Int32.(jac_row_eq_active[eq_jac_active_filter])
        eq_aug_J = _jJ[eq_jac_active_filter]

        dim = n + n_eq_a

        # Extend aug_com_base to dim × dim
        base_nzv = nonzeros(aug_com_base)
        base_rv = aug_com_base.rowval
        base_cp = aug_com_base.colptr

        # Build the extended system as a new COO and convert
        n_base_nz = length(base_nzv)
        ext_I = Vector{Int32}(undef, n_base_nz + nnzj_eq_a + n_eq_a)
        ext_J = Vector{Int32}(undef, n_base_nz + nnzj_eq_a + n_eq_a)

        # Extract COO from base CSC (CPU loop over CSC structure)
        cnt = 0
        for c in 1:n
            for k in base_cp[c]:base_cp[c+1]-1
                cnt += 1
                ext_I[cnt] = base_rv[k]
                ext_J[cnt] = Int32(c)
            end
        end

        # Add equality Jacobian (lower triangular: row=n+i > col=j for j<=n)
        off = n_base_nz
        for k in 1:nnzj_eq_a
            ext_I[off + k] = eq_aug_I[k]
            ext_J[off + k] = eq_aug_J[k]
        end

        # Add diagonal entries for equality dual regularization
        off2 = n_base_nz + nnzj_eq_a
        for k in 1:n_eq_a
            ext_I[off2 + k] = Int32(n + k)
            ext_J[off2 + k] = Int32(n + k)
        end

        ext_V = fill!(VT(undef, length(ext_I)), zero(T))

        # Convert ext_I, ext_J to device arrays
        _extI = create_array(cb, Int32, length(ext_I)); copyto!(_extI, ext_I)
        _extJ = create_array(cb, Int32, length(ext_J)); copyto!(_extJ, ext_J)

        ext_raw = SparseMatrixCOO(dim, dim, _extI, _extJ, ext_V)
        aug_com, aug_csc_map = coo_to_csc(ext_raw)

        # Remap dptr, hptr, jptr from base to extended aug_com
        base_to_ext = Vector{Int}(undef, n_base_nz)
        for c in 1:n
            for k_base in aug_com_base.colptr[c]:aug_com_base.colptr[c+1]-1
                row_base = aug_com_base.rowval[k_base]
                for k_ext in aug_com.colptr[c]:aug_com.colptr[c+1]-1
                    if aug_com.rowval[k_ext] == row_base
                        base_to_ext[k_base] = k_ext
                        break
                    end
                end
            end
        end

        new_dptr = map(((i,j),) -> (Int32(base_to_ext[i]), j), dptr)
        new_hptr = map(((i,j),) -> (Int32(base_to_ext[i]), j), hptr)
        new_jptr = map(((i,t),) -> (Int32(base_to_ext[i]), t), jptr)

        # Find eq diagonal indices in aug_com (CPU loop over CSC structure)
        eq_aug_diag_indices_cpu = Vector{Int}(undef, n_eq_a)
        for k in 1:n_eq_a
            col = n + k
            for j in aug_com.colptr[col]:aug_com.colptr[col+1]-1
                if aug_com.rowval[j] == col
                    eq_aug_diag_indices_cpu[k] = j
                    break
                end
            end
        end
        eq_aug_diag_indices = create_array(cb, Int, n_eq_a)
        copyto!(eq_aug_diag_indices, eq_aug_diag_indices_cpu)

        # Precompute nzval indices for eq Jacobian entries in aug_com (CPU loop)
        eq_jac_nzval_indices_cpu = Vector{Int}(undef, nnzj_eq_a)
        for k in 1:nnzj_eq_a
            i, j = Int(eq_aug_I[k]), Int(eq_aug_J[k])
            r, c = i > j ? (i, j) : (j, i)  # lower triangular
            for idx in aug_com.colptr[c]:aug_com.colptr[c+1]-1
                if aug_com.rowval[idx] == r
                    eq_jac_nzval_indices_cpu[k] = idx
                    break
                end
            end
        end
        _eq_nzv_idx = create_array(cb, Int, nnzj_eq_a)
        copyto!(_eq_nzv_idx, eq_jac_nzval_indices_cpu)

        eq_jac_vals_filter_vec = Vector{Int}(eq_jac_active_filter)

        aug_com_final = aug_com
        dptr_final = new_dptr
        hptr_final = new_hptr
        jptr_final = new_jptr
    else
        aug_com_final = aug_com_base
        dptr_final = dptr
        hptr_final = hptr
        jptr_final = jptr
        _eq_nzv_idx = create_array(cb, Int, 0)
        eq_jac_active_filter = Int[]
        eq_aug_diag_indices = create_array(cb, Int, 0)
        eq_jac_vals_filter_vec = Int[]
    end

    # -- Work vectors --
    reg     = fill!(VT(undef, n + ns), zero(T))
    pr_diag = fill!(VT(undef, n + ns), zero(T))
    du_diag = fill!(VT(undef, m),      zero(T))
    l_diag  = fill!(VT(undef, nlb), one(T))
    u_diag  = fill!(VT(undef, nub), one(T))
    l_lower = fill!(VT(undef, nlb), zero(T))
    u_lower = fill!(VT(undef, nub), zero(T))

    ind_active_slack_pd     = n .+ active_slack_map
    ind_active_dual_pd      = (n + ns) .+ active_cons
    ind_active_ineq_dual_pd = (n + ns) .+ active_ineq_cons
    ind_active_eq_dual_pd   = (n + ns) .+ active_eq_cons
    work_n_a                = VT(undef, n_a)

    dim = n + n_eq_a
    work_reduced  = VT(undef, dim)
    buffer_ineq   = VT(undef, ns_a)
    buffer2_ineq  = VT(undef, ns_a)
    r_s_save_buf  = VT(undef, ns_a)
    diag_buffer   = VT(undef, ns_a)

    _linear_solver = linear_solver(aug_com_final; opt = opt_linear_solver)
    quasi_newton = MadNLP.ExactHessian{T, VT}()

    return SparseCondensedActiveSetKKTSystem(
        hess, hess_raw, hess_com, hess_csc_map,
        jac_callback, jac_raw, jac_com, jac_csc_map,
        jac_active_raw, jac_active_com, jac_active_csc_map, jac_active_filter,
        jac_ineq_active_t_coo, jac_ineq_active_t_csc, jac_ineq_active_t_csc_map,
        ineq_jac_active_filter,
        aug_com_final, diag_buffer, dptr_final, hptr_final, jptr_final,
        _eq_nzv_idx, eq_jac_vals_filter_vec, eq_aug_diag_indices,
        quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        cb.ind_ineq, cb.ind_lb, cb.ind_ub,
        active_cons, active_slack_map,
        n_a, ns_a, n_eq_a,
        active_eq_cons, active_ineq_cons,
        ind_active_slack_pd, ind_active_dual_pd,
        ind_active_ineq_dual_pd, ind_active_eq_dual_pd,
        work_n_a,
        buffer_ineq, buffer2_ineq, r_s_save_buf, work_reduced,
        nnzh,
        _linear_solver,
        Dict{Symbol, Any}(),
    )
end
