const SUCCESSFUL_STATUSES = (SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL)
function assert_solved_and_feasible(solver::AbstractMadNLPSolver)
    solver.status ∉ SUCCESSFUL_STATUSES &&
        error("Solver did not converge successfully: $(solver.status)")
    return nothing
end

struct ForwardCache{VT, VK, PV}
    kkt_rhs::VK
    kkt_sol::VK
    kkt_work::VK
    dzl_reduced::VT
    dzu_reduced::VT
    dλ_reduced::VT
    dl_dp::PV
    du_dp::PV
    dx_reduced::VT
    d2L_dxdp::VT
    dg_dp::VT
    dlcon_dp::VT
    ducon_dp::VT
end

function _get_forward_cache!(sens::MadDiffSolver{T}) where {T}
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
            _zeros_like(cb, T, length(cb.ind_lb)),
            _zeros_like(cb, T, length(cb.ind_ub)),
            _zeros_like(cb, T, n_con),
            PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub),
            PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub),
            _zeros_like(cb, T, cb.nvar),
            _zeros_like(cb, T, n_x),
            _zeros_like(cb, T, n_con),
            _zeros_like(cb, T, n_con),
            _zeros_like(cb, T, n_con),
        )
    end
    return sens.forward_cache
end

struct ReverseCache{VT, VK, PV}
    kkt_rhs::VK
    kkt_sol::VK
    kkt_work::VK
    dzl_reduced::VT
    dzu_reduced::VT
    dλ_reduced::VT
    dl_dp::PV
    du_dp::PV
    dx_reduced::VT
    dL_dx::VT
    dL_dλ::VT
    dL_dzl::VT
    dL_dzu::VT
    eq_scale::VT
    dλ_scaled::VT
end

function _get_reverse_cache!(sens::MadDiffSolver{T}) where {T}
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
            _zeros_like(cb, T, length(cb.ind_lb)),
            _zeros_like(cb, T, length(cb.ind_ub)),
            _zeros_like(cb, T, n_con),
            PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub),
            PrimalVector(VT, cb.nvar, n_ineq, cb.ind_lb, cb.ind_ub),
            _zeros_like(cb, T, cb.nvar),
            _zeros_like(cb, T, cb.nvar),
            _zeros_like(cb, T, n_con),
            _zeros_like(cb, T, length(cb.ind_lb)),
            _zeros_like(cb, T, length(cb.ind_ub)),
            ifelse.(sens.is_eq, T(1 // 2), one(T)),
            _zeros_like(cb, T, n_con),
        )
    end
    return sens.reverse_cache
end

_zeros_like(cb, ::Type{T}, n::Int) where {T} = fill!(create_array(cb, T, n), zero(T))

function _unpack_z!(result, cb, cache)
    fill!(full(cache.dl_dp), zero(eltype(full(cache.dl_dp))))
    fill!(full(cache.du_dp), zero(eltype(full(cache.du_dp))))
    cache.dl_dp.values_lr .= dual_lb(cache.kkt_rhs)
    cache.du_dp.values_ur .= dual_ub(cache.kkt_rhs)
    unpack_z!(result.dzl, cb, variable(cache.dl_dp))
    unpack_z!(result.dzu, cb, variable(cache.du_dp))
    return nothing
end

_get_fixed_idx(cb::AbstractCallback, ::Any) = nothing
function _get_fixed_idx(cb::SparseCallback{T,VT,VI,NLP,FH}, ref_array) where {T,VT,VI,NLP,FH<:MakeParameter}
    return cb.fixed_handler.fixed
end

function _pullback_add!(out, M, v)
    isnothing(M) && return nothing
    @lencheck size(M, 1) v
    out .+= M' * v
    return nothing
end

function _pullback_sub!(out, M, v)
    isnothing(M) && return nothing
    @lencheck size(M, 1) v
    out .-= M' * v
    return nothing
end

function _has_custom_config(config)
    return !isnothing(config.kkt_system) ||
        !isnothing(config.kkt_options) ||
        !isnothing(config.linear_solver) ||
        !isnothing(config.linear_solver_options)
end

_get_wrapper_type(x) = Base.typename(typeof(x)).wrapper
