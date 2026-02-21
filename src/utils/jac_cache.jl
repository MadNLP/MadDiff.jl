function _to_index_array(cb, src::AbstractVector{<:Integer})
    out = create_array(cb, Int, length(src))
    copyto!(out, src)
    return out
end

function _build_index_map(n::Int, indices)
    map = zeros(Int, n)
    idx = collect(indices)
    map[idx] .= 1:length(idx)
    return map
end

_var_to_cb_indices(cb, n_x) = 1:min(n_x, cb.nvar)
function _var_to_cb_indices(
    cb::SparseCallback{<:Any, <:Any, <:Any, <:Any, FH},
    n_x,
) where {FH<:MakeParameter}
    return cb.fixed_handler.free
end

function _build_fixed_index_host(::AbstractCallback, ::Int)
    return Int[]
end

function _build_fixed_index_host(
    cb::SparseCallback{T, VT, VI, NLP, FH},
    n_x::Int,
) where {T, VT, VI, NLP, FH<:MakeParameter}
    fixed = collect(cb.fixed_handler.fixed)
    return [i for i in fixed if 1 <= i <= n_x]
end

struct SparseParamBlock{C, S, M}
    raw::C
    mat::S
    map::M
end

struct RHSBlock{VI}
    seg::UnitRange{Int}
    src::VI
    row::VI   # empty when not needed
    pos::VI   # empty when not needed
end

struct FixedVarBlock{C, S, M, VI, MT}
    raw::C
    mat::S
    map::M
    src::VI
    work::MT
end

function _make_sparse_param(cb, ::Type{T}, n_rows, n_p, rows_h, cols_h) where {T}
    return SparseMatrixCOO(
        n_rows, n_p,
        _to_index_array(cb, rows_h),
        _to_index_array(cb, cols_h),
        zeros_like(cb, T, length(rows_h)),
    )
end

function _make_sparse_param_csc(cb, ::Type{T}, n_rows, n_p, rows_h, cols_h) where {T}
    raw = _make_sparse_param(cb, T, n_rows, n_p, rows_h, cols_h)
    mat, map = coo_to_csc(raw)
    return SparseParamBlock(raw, mat, map)
end

function _load_param_structures(pmeta, nlp)
    empty = (Int[], Int[])
    return (
        hess = pmeta.hess_param_available ? hess_param_structure(nlp)     : empty,
        jac  = pmeta.jac_param_available  ? jac_param_structure(nlp)      : empty,
        lvar = pmeta.lvar_jac_available   ? lvar_jac_param_structure(nlp) : empty,
        uvar = pmeta.uvar_jac_available   ? uvar_jac_param_structure(nlp) : empty,
        lcon = pmeta.lcon_jac_available   ? lcon_jac_param_structure(nlp) : empty,
        ucon = pmeta.ucon_jac_available   ? ucon_jac_param_structure(nlp) : empty,
    )
end

struct JacobianCache{VT, MT, WM, VI}
    x_nlp::VT
    y_nlp::VT
    grad_x::VT
    grad_p::VT
    hess_raw
    jac_raw
    lvar_raw
    uvar_raw
    lcon_raw
    ucon_raw
    rhs_raw
    rhs_map
    blk_hess::RHSBlock{VI}
    blk_jac::RHSBlock{VI}
    blk_lcon_eq::RHSBlock{VI}
    blk_ucon_eq::RHSBlock{VI}
    blk_lvar::RHSBlock{VI}
    blk_uvar::RHSBlock{VI}
    blk_lcon_slack::RHSBlock{VI}
    blk_ucon_slack::RHSBlock{VI}
    fixed_idx::VI
    fixed_l::FixedVarBlock
    fixed_u::FixedVarBlock
    eye_p::MT
    dz_work::MT
    W::WM
end

# Filter hess rows through var_to_cb (skip fixed vars); primal block, no row/pos needed.
function _build_hess_block!(rhs_I, rhs_J, rows_h, cols_h, var_to_cb_h)
    src_h = Int[]
    start = length(rhs_I) + 1
    for k in eachindex(rows_h)
        row_cb = var_to_cb_h[rows_h[k]]
        row_cb == 0 && continue
        push!(rhs_I, row_cb); push!(rhs_J, cols_h[k]); push!(src_h, k)
    end
    return RHSBlock(start:length(rhs_I), src_h, Int[], Int[])
end

# All entries go in; offset into dual block; keep row index for scaling.
function _build_dual_block!(rhs_I, rhs_J, rows_h, cols_h, offset)
    src_h = Int[]; row_h = Int[]
    start = length(rhs_I) + 1
    for k in eachindex(rows_h)
        push!(rhs_I, offset + rows_h[k]); push!(rhs_J, cols_h[k])
        push!(src_h, k); push!(row_h, rows_h[k])
    end
    return RHSBlock(start:length(rhs_I), src_h, row_h, Int[])
end

# Filter by var_to_cb then by bound membership; offset into bound block; keep pos for scaling.
function _build_var_bound_block!(rhs_I, rhs_J, rows_h, cols_h, var_to_cb_h, pos_h, offset)
    src_h = Int[]; pos_out = Int[]
    start = length(rhs_I) + 1
    for k in eachindex(rows_h)
        row_cb = var_to_cb_h[rows_h[k]]
        row_cb == 0 && continue
        pos = pos_h[row_cb]
        pos == 0 && continue
        push!(rhs_I, offset + pos); push!(rhs_J, cols_h[k])
        push!(src_h, k); push!(pos_out, pos)
    end
    return RHSBlock(start:length(rhs_I), src_h, Int[], pos_out)
end

# Filter by ineq_to_slack then by bound membership; keep row for scaling and pos for scaling.
function _build_slack_bound_block!(rhs_I, rhs_J, rows_h, cols_h, ineq_to_slack_h, pos_h, n_var_cb, offset)
    src_h = Int[]; row_h = Int[]; pos_out = Int[]
    start = length(rhs_I) + 1
    for k in eachindex(rows_h)
        slack_pos = ineq_to_slack_h[rows_h[k]]
        slack_pos == 0 && continue
        pos = pos_h[n_var_cb + slack_pos]
        pos == 0 && continue
        push!(rhs_I, offset + pos); push!(rhs_J, cols_h[k])
        push!(src_h, k); push!(row_h, rows_h[k]); push!(pos_out, pos)
    end
    return RHSBlock(start:length(rhs_I), src_h, row_h, pos_out)
end

# Filter by fixed_pos; builds a separate sparse matrix (not part of rhs).
function _build_fixed_block!(rows_h, cols_h, fixed_pos_h)
    I_h = Int[]; J_h = Int[]; src_h = Int[]
    for k in eachindex(rows_h)
        pos = fixed_pos_h[rows_h[k]]
        pos == 0 && continue
        push!(I_h, pos); push!(J_h, cols_h[k]); push!(src_h, k)
    end
    return I_h, J_h, src_h
end

function get_jac_cache!(sens::MadDiffSolver{T}) where {T}
    if isnothing(sens.jac_cache)
        cb = sens.solver.cb
        nlp = sens.solver.nlp
        pmeta = nlp.pmeta
        n_x = get_nvar(sens.solver.nlp)
        n_con = get_ncon(sens.solver.nlp)
        n_p = sens.n_p
        n_var_cb = cb.nvar
        n_ineq = length(cb.ind_ineq)
        n_pv = n_var_cb + n_ineq
        n_primal = length(sens.kkt.pr_diag)
        n_dual = length(sens.kkt.du_diag)
        n_lb = length(sens.kkt.l_diag)
        n_rhs = length(sens.kkt.pr_diag) + length(sens.kkt.du_diag) +
            length(sens.kkt.l_diag) + length(sens.kkt.u_diag)

        st = _load_param_structures(pmeta, nlp)
        hess_rows_h, hess_cols_h = st.hess
        jac_rows_h, jac_cols_h   = st.jac
        lvar_rows_h, lvar_cols_h = st.lvar
        uvar_rows_h, uvar_cols_h = st.uvar
        lcon_rows_h, lcon_cols_h = st.lcon
        ucon_rows_h, ucon_cols_h = st.ucon

        var_to_cb_h     = _build_index_map(n_x, _var_to_cb_indices(cb, n_x))
        ineq_to_slack_h = _build_index_map(n_con, cb.ind_ineq)
        lb_pos_h        = _build_index_map(n_pv, cb.ind_lb)
        ub_pos_h        = _build_index_map(n_pv, cb.ind_ub)
        fixed_idx_h     = _build_fixed_index_host(cb, n_x)
        fixed_pos_h     = _build_index_map(n_x, fixed_idx_h)
        n_fixed         = length(fixed_idx_h)

        dual_offset = n_primal
        lb_offset   = n_primal + n_dual
        ub_offset   = n_primal + n_dual + n_lb

        rhs_I_h = Int[]; rhs_J_h = Int[]

        blk_hess_h      = _build_hess_block!(rhs_I_h, rhs_J_h, hess_rows_h, hess_cols_h, var_to_cb_h)
        blk_jac_h       = _build_dual_block!(rhs_I_h, rhs_J_h, jac_rows_h,  jac_cols_h,  dual_offset)
        blk_lcon_eq_h   = _build_dual_block!(rhs_I_h, rhs_J_h, lcon_rows_h, lcon_cols_h, dual_offset)
        blk_ucon_eq_h   = _build_dual_block!(rhs_I_h, rhs_J_h, ucon_rows_h, ucon_cols_h, dual_offset)
        blk_lvar_h      = _build_var_bound_block!(rhs_I_h, rhs_J_h, lvar_rows_h, lvar_cols_h, var_to_cb_h, lb_pos_h, lb_offset)
        blk_uvar_h      = _build_var_bound_block!(rhs_I_h, rhs_J_h, uvar_rows_h, uvar_cols_h, var_to_cb_h, ub_pos_h, ub_offset)
        fixed_l_I_h, fixed_l_J_h, fixed_l_src_h = _build_fixed_block!(lvar_rows_h, lvar_cols_h, fixed_pos_h)
        fixed_u_I_h, fixed_u_J_h, fixed_u_src_h = _build_fixed_block!(uvar_rows_h, uvar_cols_h, fixed_pos_h)
        blk_lcon_slack_h = _build_slack_bound_block!(rhs_I_h, rhs_J_h, lcon_rows_h, lcon_cols_h, ineq_to_slack_h, lb_pos_h, n_var_cb, lb_offset)
        blk_ucon_slack_h = _build_slack_bound_block!(rhs_I_h, rhs_J_h, ucon_rows_h, ucon_cols_h, ineq_to_slack_h, ub_pos_h, n_var_cb, ub_offset)

        hess_raw = _make_sparse_param(cb, T, n_x,   n_p, hess_rows_h, hess_cols_h)
        jac_raw  = _make_sparse_param(cb, T, n_con,  n_p, jac_rows_h,  jac_cols_h)
        lvar_raw = _make_sparse_param(cb, T, n_x,   n_p, lvar_rows_h, lvar_cols_h)
        uvar_raw = _make_sparse_param(cb, T, n_x,   n_p, uvar_rows_h, uvar_cols_h)
        lcon_raw = _make_sparse_param(cb, T, n_con,  n_p, lcon_rows_h, lcon_cols_h)
        ucon_raw = _make_sparse_param(cb, T, n_con,  n_p, ucon_rows_h, ucon_cols_h)

        rhs_vals = zeros_like(cb, T, length(rhs_I_h))
        rhs_raw = SparseMatrixCOO(
            n_rhs,
            n_p,
            _to_index_array(cb, rhs_I_h),
            _to_index_array(cb, rhs_J_h),
            rhs_vals,
        )
        W, rhs_map = coo_to_csc(rhs_raw)

        fixed_l_blk = _make_sparse_param_csc(cb, T, n_fixed, n_p, fixed_l_I_h, fixed_l_J_h)
        fixed_l = FixedVarBlock(fixed_l_blk.raw, fixed_l_blk.mat, fixed_l_blk.map,
                                _to_index_array(cb, fixed_l_src_h), zeros_like(cb, T, n_fixed, n_p))
        fixed_u_blk = _make_sparse_param_csc(cb, T, n_fixed, n_p, fixed_u_I_h, fixed_u_J_h)
        fixed_u = FixedVarBlock(fixed_u_blk.raw, fixed_u_blk.mat, fixed_u_blk.map,
                                _to_index_array(cb, fixed_u_src_h), zeros_like(cb, T, n_fixed, n_p))

        eye_p = zeros_like(cb, T, n_p, n_p)
        eye_h = zeros(T, n_p, n_p)
        eye_h[diagind(eye_h)] .= one(T)
        copyto!(eye_p, eye_h)

        _blk(b::RHSBlock) = RHSBlock(b.seg,
            _to_index_array(cb, b.src), _to_index_array(cb, b.row), _to_index_array(cb, b.pos))

        sens.jac_cache = JacobianCache(
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_p),
            hess_raw,
            jac_raw,
            lvar_raw,
            uvar_raw,
            lcon_raw,
            ucon_raw,
            rhs_raw,
            rhs_map,
            _blk(blk_hess_h),
            _blk(blk_jac_h),
            _blk(blk_lcon_eq_h),
            _blk(blk_ucon_eq_h),
            _blk(blk_lvar_h),
            _blk(blk_uvar_h),
            _blk(blk_lcon_slack_h),
            _blk(blk_ucon_slack_h),
            _to_index_array(cb, fixed_idx_h),
            fixed_l,
            fixed_u,
            eye_p,
            zeros_like(cb, T, n_var_cb + n_ineq, n_p),
            W,
        )
    end
    return sens.jac_cache
end

"""
    JacobianResult{MT,VT}

Container for the Jacobian of the optimal solution with respect to parameters.

Fields are Jacobian blocks with columns corresponding to parameter directions:

- `dx`: ∂x/∂p
- `dy`: ∂y/∂p
- `dzl`: ∂zl/∂p
- `dzu`: ∂zu/∂p
- `dobj`: ∂obj/∂p (objective gradient w.r.t. parameters)

Returned by [`jacobian!`](@ref).
"""
struct JacobianResult{MT, VT}
    dx::MT
    dy::MT
    dzl::MT
    dzu::MT
    dobj::VT
end

function JacobianResult(sens::MadDiffSolver{T}) where {T}
    n_x = get_nvar(sens.solver.nlp)
    n_con = get_ncon(sens.solver.nlp)
    n_p = sens.n_p
    cb = sens.solver.cb
    return JacobianResult(
        zeros_like(cb, T, n_x, n_p),
        zeros_like(cb, T, n_con, n_p),
        zeros_like(cb, T, n_x, n_p),
        zeros_like(cb, T, n_x, n_p),
        zeros_like(cb, T, n_p),
    )
end

struct JacobianTransposeCache{VT, MT, WM}
    x_nlp::VT
    y_nlp::VT
    dy_nlp::MT
    grad_x::VT
    grad_p::VT
    hess
    jac
    lvar
    uvar
    lcon
    ucon
    dx_solved::MT
    dy_solved::MT
    dzl_solved::MT
    dzu_solved::MT
    dz_work::MT
    pv_lr::MT
    pv_ur::MT
    x_lr::MT
    x_ur::MT
    y_lr::MT
    y_ur::MT
    grad_all::MT
    tmp_mul::MT
    W::WM
end

function get_jact_cache!(sens::MadDiffSolver{T}) where {T}
    if isnothing(sens.jact_cache)
        cb = sens.solver.cb
        nlp = sens.solver.nlp
        pmeta = nlp.pmeta
        n_x = get_nvar(sens.solver.nlp)
        n_con = get_ncon(sens.solver.nlp)
        n_p = sens.n_p
        n_var_cb = cb.nvar
        n_ineq = length(cb.ind_ineq)
        n_rhs = length(sens.kkt.pr_diag) + length(sens.kkt.du_diag) +
            length(sens.kkt.l_diag) + length(sens.kkt.u_diag)
        n_seed = 3 * n_x + n_con + 1

        st = _load_param_structures(pmeta, nlp)
        hess_rows_h, hess_cols_h = st.hess
        jac_rows_h, jac_cols_h   = st.jac
        lvar_rows_h, lvar_cols_h = st.lvar
        uvar_rows_h, uvar_cols_h = st.uvar
        lcon_rows_h, lcon_cols_h = st.lcon
        ucon_rows_h, ucon_cols_h = st.ucon

        hess = _make_sparse_param_csc(cb, T, n_x,  n_p, hess_rows_h, hess_cols_h)
        jac  = _make_sparse_param_csc(cb, T, n_con, n_p, jac_rows_h,  jac_cols_h)
        lvar = _make_sparse_param_csc(cb, T, n_x,  n_p, lvar_rows_h, lvar_cols_h)
        uvar = _make_sparse_param_csc(cb, T, n_x,  n_p, uvar_rows_h, uvar_cols_h)
        lcon = _make_sparse_param_csc(cb, T, n_con, n_p, lcon_rows_h, lcon_cols_h)
        ucon = _make_sparse_param_csc(cb, T, n_con, n_p, ucon_rows_h, ucon_cols_h)

        sens.jact_cache = JacobianTransposeCache(
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_con, n_seed),
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_p),
            hess,
            jac,
            lvar,
            uvar,
            lcon,
            ucon,
            zeros_like(cb, T, n_x, n_seed),
            zeros_like(cb, T, n_con, n_seed),
            zeros_like(cb, T, n_x, n_seed),
            zeros_like(cb, T, n_x, n_seed),
            zeros_like(cb, T, n_var_cb + n_ineq, n_seed),
            zeros_like(cb, T, n_var_cb + n_ineq, n_seed),
            zeros_like(cb, T, n_var_cb + n_ineq, n_seed),
            zeros_like(cb, T, n_x, n_seed),
            zeros_like(cb, T, n_x, n_seed),
            zeros_like(cb, T, n_con, n_seed),
            zeros_like(cb, T, n_con, n_seed),
            zeros_like(cb, T, n_p, n_seed),
            zeros_like(cb, T, n_p, n_seed),
            spzeros_like(cb, T, n_rhs, n_seed),
        )
    end
    return sens.jact_cache
end

"""
    JacobianTransposeResult{MT,VT}

Container for the transpose of the Jacobian of the optimal solution with respect
to parameters.

Fields represent Jacobian-transpose blocks (rows corresponding to parameters):

- `dx`: (∂x/∂p)ᵀ
- `dy`: (∂y/∂p)ᵀ
- `dzl`: (∂zl/∂p)ᵀ
- `dzu`: (∂zu/∂p)ᵀ
- `dobj`: (∂obj/∂p)ᵀ

Returned by [`jacobian_transpose!`](@ref).
"""
struct JacobianTransposeResult{MT, VT}
    dx::MT
    dy::MT
    dzl::MT
    dzu::MT
    dobj::VT
end

function JacobianTransposeResult(sens::MadDiffSolver{T}) where {T}
    n_x = get_nvar(sens.solver.nlp)
    n_con = get_ncon(sens.solver.nlp)
    n_p = sens.n_p
    cb = sens.solver.cb
    return JacobianTransposeResult(
        zeros_like(cb, T, n_p, n_x),
        zeros_like(cb, T, n_p, n_con),
        zeros_like(cb, T, n_p, n_x),
        zeros_like(cb, T, n_p, n_x),
        zeros_like(cb, T, n_p),
    )
end

has_hess_param(jcache::JacobianCache)       = !isempty(jcache.blk_hess.src)
has_jac_param(jcache::JacobianCache)        = !isempty(jcache.blk_jac.src)
has_lcon_eq_param(jcache::JacobianCache)    = !isempty(jcache.blk_lcon_eq.src)
has_ucon_eq_param(jcache::JacobianCache)    = !isempty(jcache.blk_ucon_eq.src)
has_lvar_param(jcache::JacobianCache)       = !isempty(jcache.blk_lvar.src)
has_uvar_param(jcache::JacobianCache)       = !isempty(jcache.blk_uvar.src)
has_lcon_slack_param(jcache::JacobianCache) = !isempty(jcache.blk_lcon_slack.src)
has_ucon_slack_param(jcache::JacobianCache) = !isempty(jcache.blk_ucon_slack.src)
