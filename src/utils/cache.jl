# ============================================================================
# Scratch caches and result containers for JVP/VJP kernels.
# ============================================================================

zeros_like(cb, ::Type{T}, dims::Int...) where {T} =
    fill!(create_array(cb, T, dims...), zero(T))

# ---------- JVP ----------

struct JVPCache{VT, VK, PV}
    kkt_rhs::VK
    kkt_sol::VK
    kkt_work::VK
    d2L_dxdp::VT
    dg_dp::VT
    dlvar_dp::PV
    duvar_dp::PV
    dlcon_dp::VT
    ducon_dp::VT
    x_nlp::VT
    y_nlp::VT
    hpv_nlp::VT
    jpv_nlp::VT
    dlvar_nlp::VT
    duvar_nlp::VT
    dlcon_nlp::VT
    ducon_nlp::VT
    grad_x::VT
    grad_p::VT
end

function get_jvp_cache!(sens::MadDiffSolver{T}) where {T}
    sens.jvp_cache === nothing || return sens.jvp_cache

    cb     = sens.solver.cb
    nlp    = sens.solver.nlp
    n_x    = get_nvar(nlp)
    n_con  = get_ncon(nlp)
    n_ineq = length(cb.ind_ineq)
    VT     = typeof(full(sens.solver.x))

    pv() = PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub)
    kv() = UnreducedKKTVector(sens.kkt)
    z(n) = zeros_like(cb, T, n)

    sens.jvp_cache = JVPCache(
        kv(), kv(), kv(),
        z(cb.nvar), z(n_con),
        pv(), pv(),
        z(n_con), z(n_con),
        z(n_x), z(n_con),
        z(n_x), z(n_con),
        z(n_x), z(n_x),
        z(n_con), z(n_con),
        z(n_x), z(sens.n_p),
    )
    return sens.jvp_cache
end

"""
    JVPResult

Result of [`jacobian_vector_product!`](@ref). Fields:
`dx`, `dy`, `dzl`, `dzu`, and `dobj::Base.RefValue{T}` (populated by
[`compute_objective_sensitivity!`](@ref)).
"""
struct JVPResult{VT, T}
    dx::VT
    dy::VT
    dzl::VT
    dzu::VT
    dobj::Base.RefValue{T}
end

function JVPResult(sens::MadDiffSolver{T}) where {T}
    cb    = sens.solver.cb
    n_x   = get_nvar(sens.solver.nlp)
    n_con = get_ncon(sens.solver.nlp)
    return JVPResult(
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_con),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_x),
        Ref(zero(T)),
    )
end

# ---------- VJP ----------

struct VJPCache{VT, VK, PV}
    kkt_rhs::VK
    kkt_sol::VK
    kkt_work::VK
    dzl_full::PV
    dzu_full::PV
    dL_dx::VT
    dL_dy::VT
    dL_dzl::VT
    dL_dzu::VT
    x_nlp::VT
    y_nlp::VT
    dy_scaled::VT
    tmp_p::VT
    grad_x::VT
end

function get_vjp_cache!(sens::MadDiffSolver{T}) where {T}
    sens.vjp_cache === nothing || return sens.vjp_cache

    cb     = sens.solver.cb
    nlp    = sens.solver.nlp
    n_x    = get_nvar(nlp)
    n_con  = get_ncon(nlp)
    n_ineq = length(cb.ind_ineq)
    n_lb   = length(cb.ind_lb)
    n_ub   = length(cb.ind_ub)
    VT     = typeof(full(sens.solver.x))

    pv() = PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub)
    kv() = UnreducedKKTVector(sens.kkt)
    z(n) = zeros_like(cb, T, n)

    sens.vjp_cache = VJPCache(
        kv(), kv(), kv(),
        pv(), pv(),
        z(cb.nvar), z(n_con),
        z(n_lb), z(n_ub),
        z(n_x), z(n_con), z(n_con),
        z(sens.n_p), z(cb.nvar),
    )
    return sens.vjp_cache
end

"""
    VJPResult

Result of [`vector_jacobian_product!`](@ref). `grad_p` is the parameter
gradient; `dx`, `dy`, `dzl`, `dzu` are the adjoint solve's outputs.
"""
struct VJPResult{VT, GT}
    dx::VT
    dy::VT
    dzl::VT
    dzu::VT
    grad_p::GT
end

function VJPResult(sens::MadDiffSolver{T}) where {T}
    cb    = sens.solver.cb
    n_x   = get_nvar(sens.solver.nlp)
    n_con = get_ncon(sens.solver.nlp)
    return VJPResult(
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_con),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, sens.n_p),
    )
end
