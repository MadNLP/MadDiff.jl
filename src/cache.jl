zeros_like(cb, ::Type{T}, n::Int) where {T} = fill!(create_array(cb, T, n), zero(T))
zeros_like(cb, ::Type{T}, n::Int, m::Int) where {T} = fill!(create_array(cb, T, n, m), zero(T))

struct ForwardCache{VT, VK, PV}
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

function get_forward_cache!(sens::MadDiffSolver{T}) where {T}
    if isnothing(sens.forward_cache)
        cb = sens.solver.cb
        n_x = get_nvar(sens.solver.nlp)
        n_con = get_ncon(sens.solver.nlp)
        n_p = sens.n_p
        x_array = full(sens.solver.x)
        VT = typeof(x_array)
        n_ineq = length(cb.ind_ineq)

        sens.forward_cache = ForwardCache(
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
    return sens.forward_cache
end

function get_jacobian_forward_cache!(sens::MadDiffSolver{T}) where {T}
    if isnothing(sens.jacobian_cache)
        cb = sens.solver.cb
        n_x = get_nvar(sens.solver.nlp)
        n_con = get_ncon(sens.solver.nlp)
        n_p = sens.n_p
        n_var_cb = cb.nvar
        n_ineq = length(cb.ind_ineq)
        n_rhs = length(sens.kkt.pr_diag) + length(sens.kkt.du_diag) +
            length(sens.kkt.l_diag) + length(sens.kkt.u_diag)

        sens.jacobian_cache = JacobianForwardCache(
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
            spzeros(T, n_rhs, n_p),
        )
    end
    return sens.jacobian_cache
end

struct ForwardResult{VT, T}
    dx::VT
    dy::VT
    dzl::VT
    dzu::VT
    dobj::Base.RefValue{T}
end

function ForwardResult(sens::MadDiffSolver{T}) where {T}
    n_x = get_nvar(sens.solver.nlp)
    n_con = get_ncon(sens.solver.nlp)
    cb = sens.solver.cb
    return ForwardResult(
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_con),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_x),
        Ref(zero(T)),
    )
end

struct ReverseCache{VT, VK, PV}
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

function get_reverse_cache!(sens::MadDiffSolver{T}) where {T}
    if isnothing(sens.reverse_cache)
        cb = sens.solver.cb
        n_x = get_nvar(sens.solver.nlp)
        n_con = get_ncon(sens.solver.nlp)
        n_p = sens.n_p
        x_array = full(sens.solver.x)
        VT = typeof(x_array)
        n_ineq = length(cb.ind_ineq)
        sens.reverse_cache = ReverseCache(
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
    return sens.reverse_cache
end

struct ReverseResult{VT, GT}
    dx::VT
    dy::VT
    dzl::VT
    dzu::VT
    grad_p::GT
end

struct JacobianResult{MT, VT}
    dx::MT
    dy::MT
    dzl::MT
    dzu::MT
    dobj::VT
end

struct JacobianTransposeResult{MT, VT}
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

function ReverseResult(sens::MadDiffSolver{T}) where {T}
    n_x = get_nvar(sens.solver.nlp)
    n_con = get_ncon(sens.solver.nlp)
    cb = sens.solver.cb
    return ReverseResult(
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_con),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, sens.n_p),
    )
end
