zeros_like(cb, ::Type{T}, n::Int) where {T} = fill!(create_array(cb, T, n), zero(T))

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
end

function get_forward_cache!(sens::MadDiffSolver{T}) where {T}
    if isnothing(sens.forward_cache)
        cb = sens.solver.cb
        n_x = NLPModels.get_nvar(sens.solver.nlp)
        n_con = NLPModels.get_ncon(sens.solver.nlp)
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
        )
    end
    return sens.forward_cache
end

struct ForwardResult{VT}
    dx::VT
    dy::VT
    dzl::VT
    dzu::VT
end

function ForwardResult(sens::MadDiffSolver{T}) where {T}
    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    cb = sens.solver.cb
    return ForwardResult(
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_con),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_x),
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
    eq_scale::VT
    dy_scaled::VT
end

function get_reverse_cache!(sens::MadDiffSolver{T}) where {T}
    if isnothing(sens.reverse_cache)
        cb = sens.solver.cb
        n_con = NLPModels.get_ncon(sens.solver.nlp)
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
            ifelse.(sens.is_eq, T(1 // 2), one(T)),
            zeros_like(cb, T, n_con),
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

function ReverseResult(sens::MadDiffSolver{T}) where {T}
    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    cb = sens.solver.cb
    return ReverseResult(
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_con),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, n_x),
        zeros_like(cb, T, sens.n_p),
    )
end
