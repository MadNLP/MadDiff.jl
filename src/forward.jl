struct ForwardResult{VT}
    dx::VT
    dλ::VT
    dzl::VT
    dzu::VT
end

function ForwardResult(sens::MadDiffSolver{T}) where {T}
    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    cb = sens.solver.cb
    return ForwardResult(
        _zeros_like(cb, T, n_x),
        _zeros_like(cb, T, n_con),
        _zeros_like(cb, T, n_x),
        _zeros_like(cb, T, n_x),
    )
end

function forward_differentiate!(
    result::ForwardResult,
    sens::MadDiffSolver;
    d2L_dxdp = nothing,
    dg_dp = nothing,
    dl_dp = nothing,
    du_dp = nothing,
    dlcon_dp = nothing,
    ducon_dp = nothing,
)
    _pack_jvp!(sens; d2L_dxdp, dg_dp, dl_dp, du_dp, dlcon_dp, ducon_dp)
    _solve_jvp!(sens)
    _unpack_jvp!(result, sens; dl_dp, du_dp)
    return result
end

function _unpack_jvp!(result::ForwardResult, sens::MadDiffSolver; dl_dp=nothing, du_dp=nothing)
    cache = _get_forward_cache!(sens)
    cb = sens.solver.cb

    unpack_x_fixed_zero!(result.dx, cb, primal(cache.kkt_rhs))
    _set_fixed_sensitivity!(result.dx, dl_dp, du_dp, sens.fixed_idx)
    unpack_y!(result.dλ, cb, dual(cache.kkt_rhs))
    unpack_dzl!(result, cb, cache)
    unpack_dzu!(result, cb, cache)

    return result
end

function _pack_jvp!(
    sens::MadDiffSolver{T};
    d2L_dxdp = nothing,
    dg_dp = nothing,
    dl_dp = nothing,
    du_dp = nothing,
    dlcon_dp = nothing,
    ducon_dp = nothing,
) where {T}
    all(isnothing, (d2L_dxdp, dg_dp, dl_dp, du_dp, dlcon_dp, ducon_dp)) &&
        throw(ArgumentError("At least one of d2L_dxdp, dg_dp, dl_dp, du_dp, dlcon_dp, ducon_dp must be provided"))

    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    !isnothing(d2L_dxdp) && @lencheck n_x d2L_dxdp
    !isnothing(dg_dp) && @lencheck n_con dg_dp
    !isnothing(dl_dp) && @lencheck n_x dl_dp
    !isnothing(du_dp) && @lencheck n_x du_dp
    !isnothing(dlcon_dp) && @lencheck n_con dlcon_dp
    !isnothing(ducon_dp) && @lencheck n_con ducon_dp

    cache = _get_forward_cache!(sens)
    cb = sens.solver.cb

    fill!(full(cache.dl_dp), zero(T))
    fill!(full(cache.du_dp), zero(T))
    fill!(cache.dx_reduced, zero(T))
    fill!(cache.dg_dp, zero(T))
    fill!(cache.dlcon_dp, zero(T))
    fill!(cache.ducon_dp, zero(T))

    !isnothing(d2L_dxdp) && pack_x_obj!(cache.dx_reduced, cb, d2L_dxdp)
    !isnothing(dg_dp) && pack_cons!(cache.dg_dp, cb, dg_dp)
    !isnothing(dlcon_dp) && pack_cons!(cache.dlcon_dp, cb, dlcon_dp)
    !isnothing(ducon_dp) && pack_cons!(cache.ducon_dp, cb, ducon_dp)
    !isnothing(dl_dp) && pack_x!(variable(cache.dl_dp), cb, dl_dp)
    !isnothing(du_dp) && pack_x!(variable(cache.du_dp), cb, du_dp)
    !isnothing(dlcon_dp) && pack_slack!(slack(cache.dl_dp), cb, cache.dlcon_dp)
    !isnothing(ducon_dp) && pack_slack!(slack(cache.du_dp), cb, cache.ducon_dp)
    return nothing
end

function _solve_jvp!(sens::MadDiffSolver{T}) where {T}
    cache = _get_forward_cache!(sens)
    kkt = sens.kkt
    w = cache.kkt_rhs
    n_x = length(cache.dx_reduced)

    fill!(full(w), zero(T))
    primal(w)[1:n_x] .= .-cache.dx_reduced
    dual(w) .= .-cache.dg_dp .+ sens.is_eq .* (cache.dlcon_dp .+ cache.ducon_dp) ./ 2
    _jvp_set_bound_rhs!(kkt, w, cache.dl_dp, cache.du_dp)

    _solve_with_refine!(sens, w, cache)
    return nothing
end

function _solve_with_refine!(sens::MadDiffSolver{T}, w::AbstractKKTVector, cache) where {T}
    d = cache.kkt_sol
    work = cache.kkt_work

    copyto!(full(d), full(w))
    solver = sens.solver
    iterator = if sens.kkt === solver.kkt
        solver.iterator
    else
        MadNLP.RichardsonIterator(
            sens.kkt;
            opt=solver.iterator.opt,
            logger=solver.iterator.logger,
            cnt=solver.cnt,
        )
    end
    solver.cnt.linear_solver_time += @elapsed begin
        if MadNLP.solve_refine!(d, iterator, w, work)
            # ok
        elseif MadNLP.improve!(sens.kkt.linear_solver)
            MadNLP.solve_refine!(d, iterator, w, work)
        end
    end
    copyto!(full(w), full(d))
    return nothing
end

function _jvp_set_bound_rhs!(kkt, w, dl_dp, du_dp)
    dual_lb(w) .= kkt.l_lower .* dl_dp.values_lr
    dual_ub(w) .= .-kkt.u_lower .* du_dp.values_ur
    return nothing
end
function _jvp_set_bound_rhs!(::SparseUnreducedKKTSystem, w, dl_dp, du_dp)
    dual_lb(w) .= dl_dp.values_lr
    dual_ub(w) .= .-du_dp.values_ur
    return nothing
end

_set_fixed_sensitivity!(dx, dl_dp, du_dp, ::Nothing) = nothing
function _set_fixed_sensitivity!(dx, dl_dp, du_dp, fixed_idx)
    isempty(fixed_idx) && return nothing
    if isnothing(dl_dp) && isnothing(du_dp)
        return nothing
    elseif isnothing(du_dp)
        dx[fixed_idx] .= dl_dp[fixed_idx]
    elseif isnothing(dl_dp)
        dx[fixed_idx] .= du_dp[fixed_idx]
    else
        dx[fixed_idx] .= (dl_dp[fixed_idx] .+ du_dp[fixed_idx]) ./ 2
    end
    return nothing
end