zeros_like(cb, ::Type{T}, n::Int) where {T} = fill!(create_array(cb, T, n), zero(T))
zeros_like(cb, ::Type{T}, n::Int, m::Int) where {T} = fill!(create_array(cb, T, n, m), zero(T))

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
    if isnothing(sens.jvp_cache)
        cb = sens.solver.cb
        n_x = get_nvar(sens.solver.nlp)
        n_con = get_ncon(sens.solver.nlp)
        n_p = sens.n_p
        x_array = full(sens.solver.x)
        VT = typeof(x_array)
        n_ineq = length(cb.ind_ineq)

        sens.jvp_cache = JVPCache(
            UnreducedKKTVector(sens.kkt),
            UnreducedKKTVector(sens.kkt),
            UnreducedKKTVector(sens.kkt),
            zeros_like(cb, T, cb.nvar),
            zeros_like(cb, T, n_con),
            PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub),
            PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_p),
        )
    end
    return sens.jvp_cache
end

"""
    JVPResult{VT,T}

Container for the result of a Jacobian–vector product (JVP), for the sensitivity of the
optimal solution with respect to parameters.

Fields store directional sensitivities for a parameter perturbation `Δp`:

- `dx`: direction for primal variables `x`
- `dy`: direction for constraint multipliers `y`
- `dzl`: direction for lower-bound multipliers
- `dzu`: direction for upper-bound multipliers
- `dobj`: directional derivative of the objective value along `Δp`

Returned by [`jacobian_vector_product!`](@ref).
"""
struct JVPResult{VT, T}
    dx::VT
    dy::VT
    dzl::VT
    dzu::VT
    dobj::Base.RefValue{T}
end

function JVPResult(sens::MadDiffSolver{T}) where {T}
    n_x = get_nvar(sens.solver.nlp)
    n_con = get_ncon(sens.solver.nlp)
    cb = sens.solver.cb
    return JVPResult(
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_con),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_x),
        Ref(zero(T)),
    )
end

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
    if isnothing(sens.vjp_cache)
        cb = sens.solver.cb
        n_x = get_nvar(sens.solver.nlp)
        n_con = get_ncon(sens.solver.nlp)
        n_p = sens.n_p
        x_array = full(sens.solver.x)
        VT = typeof(x_array)
        n_ineq = length(cb.ind_ineq)
        sens.vjp_cache = VJPCache(
            UnreducedKKTVector(sens.kkt),
            UnreducedKKTVector(sens.kkt),
            UnreducedKKTVector(sens.kkt),
            PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub),
            PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub),
            zeros_like(cb, T, cb.nvar),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, length(cb.ind_lb)),
            zeros_like(cb, T, length(cb.ind_ub)),
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_p),
            zeros_like(cb, T, cb.nvar),
        )
    end
    return sens.vjp_cache
end

"""
    VJPResult{VT,GT}

Container for the result of a vector–Jacobian product (VJP), for backpropagating
a scalar loss through the optimal solution.

Fields:
- `dx`, `dy`, `dzl`, `dzu`: adjoints for the solution components (the solved
  reverse sensitivities)
- `grad_p`: gradient of the loss with respect to parameters

Returned by [`vector_jacobian_product!`](@ref).
"""
struct VJPResult{VT, GT}
    dx::VT
    dy::VT
    dzl::VT
    dzu::VT
    grad_p::GT
end

function VJPResult(sens::MadDiffSolver{T}) where {T}
    n_x = get_nvar(sens.solver.nlp)
    n_con = get_ncon(sens.solver.nlp)
    cb = sens.solver.cb
    return VJPResult(
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_con),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, sens.n_p),
    )
end

struct JacobianCache{VT, MT, WM}
    x_nlp::VT
    y_nlp::VT
    grad_x::VT
    grad_p::VT
    hess_nlp::MT
    jac_nlp::MT
    dlvar_nlp::MT
    duvar_nlp::MT
    dlcon_nlp::MT
    ducon_nlp::MT
    d2L_dxdp::MT
    dg_dp::MT
    dlvar_dp::MT
    duvar_dp::MT
    dlcon_dp::MT
    ducon_dp::MT
    dz_work::MT
    W::WM
end

function get_jac_cache!(sens::MadDiffSolver{T}) where {T}
    if isnothing(sens.jac_cache)
        cb = sens.solver.cb
        n_x = get_nvar(sens.solver.nlp)
        n_con = get_ncon(sens.solver.nlp)
        n_p = sens.n_p
        n_var_cb = cb.nvar
        n_ineq = length(cb.ind_ineq)
        n_rhs = length(sens.kkt.pr_diag) + length(sens.kkt.du_diag) +
            length(sens.kkt.l_diag) + length(sens.kkt.u_diag)

        sens.jac_cache = JacobianCache(
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_p),
            zeros_like(cb, T, n_x, n_p),
            zeros_like(cb, T, n_con, n_p),
            zeros_like(cb, T, n_x, n_p),
            zeros_like(cb, T, n_x, n_p),
            zeros_like(cb, T, n_con, n_p),
            zeros_like(cb, T, n_con, n_p),
            zeros_like(cb, T, n_var_cb, n_p),
            zeros_like(cb, T, n_con, n_p),
            zeros_like(cb, T, n_var_cb + n_ineq, n_p),
            zeros_like(cb, T, n_var_cb + n_ineq, n_p),
            zeros_like(cb, T, n_con, n_p),
            zeros_like(cb, T, n_con, n_p),
            zeros_like(cb, T, n_var_cb + n_ineq, n_p),
            spzeros_like(cb, T, n_rhs, n_p),  # NOTE: we build the RHS sparse
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

struct JacobianTransposeCache{VT, MT, WM}
    x_nlp::VT
    y_nlp::VT
    dy_nlp::MT
    grad_p::VT
    hess_nlp::MT
    jac_nlp::MT
    dlvar_nlp::MT
    duvar_nlp::MT
    dlcon_nlp::MT
    ducon_nlp::MT
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
    W::WM
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

function get_jact_cache!(sens::MadDiffSolver{T}) where {T}
    if isnothing(sens.jact_cache)
        cb = sens.solver.cb
        n_x = get_nvar(sens.solver.nlp)
        n_con = get_ncon(sens.solver.nlp)
        n_p = sens.n_p
        n_var_cb = cb.nvar
        n_ineq = length(cb.ind_ineq)
        n_rhs = length(sens.kkt.pr_diag) + length(sens.kkt.du_diag) +
            length(sens.kkt.l_diag) + length(sens.kkt.u_diag)
        n_seed = 3 * n_x + n_con + 1

        sens.jact_cache = JacobianTransposeCache(
            zeros_like(cb, T, n_x),
            zeros_like(cb, T, n_con),
            zeros_like(cb, T, n_con, n_seed),
            zeros_like(cb, T, n_p),
            zeros_like(cb, T, n_x, n_p),
            zeros_like(cb, T, n_con, n_p),
            zeros_like(cb, T, n_x, n_p),
            zeros_like(cb, T, n_x, n_p),
            zeros_like(cb, T, n_con, n_p),
            zeros_like(cb, T, n_con, n_p),
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
