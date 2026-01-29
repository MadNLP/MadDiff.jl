struct ReverseResult{VT, GT}
    dx::VT
    dλ::VT
    dzl::VT
    dzu::VT
    grad_p::GT
end


function ReverseResult(sens::MadDiffSolver{T}) where {T}
    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    cb = sens.solver.cb
    grad_p = sens.n_p > 0 ? _zeros_like(cb, T, sens.n_p) : nothing
    return ReverseResult(
        _zeros_like(cb, T, n_x),
        _zeros_like(cb, T, n_con),
        _zeros_like(cb, T, n_x),
        _zeros_like(cb, T, n_x),
        grad_p,
    )
end

function reverse_differentiate!(
    result::ReverseResult,
    sens::MadDiffSolver;
    dL_dx = nothing,
    dL_dλ = nothing,
    dL_dzl = nothing,
    dL_dzu = nothing,
)
    _pack_vjp!(sens; dL_dx, dL_dλ, dL_dzl, dL_dzu)
    _solve_vjp!(sens)
    _unpack_vjp!(result, sens)
    return result
end

function _unpack_vjp!(result::ReverseResult, sens::MadDiffSolver)
    cache = _get_reverse_cache!(sens)
    cb = sens.solver.cb

    unpack_x_fixed_zero!(result.dx, cb, primal(cache.kkt_rhs))
    unpack_y!(result.dλ, cb, dual(cache.kkt_rhs))
    _unpack_z!(result, cb, cache)

    if !isnothing(sens.param_pullback)
        sens.param_pullback(result.grad_p, result.dx, result.dλ, result.dzl, result.dzu, sens)
    end
    return result
end

function _pack_vjp!(
    sens::MadDiffSolver{T};
    dL_dx = nothing,
    dL_dλ = nothing,
    dL_dzl = nothing,
    dL_dzu = nothing,
) where {T}
    all(isnothing, (dL_dx, dL_dλ, dL_dzl, dL_dzu)) &&
        throw(ArgumentError("At least one of dL_dx, dL_dλ, dL_dzl, dL_dzu must be provided"))

    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    isnothing(dL_dx) || @lencheck n_x dL_dx
    isnothing(dL_dλ) || @lencheck n_con dL_dλ
    isnothing(dL_dzl) || @lencheck n_x dL_dzl
    isnothing(dL_dzu) || @lencheck n_x dL_dzu

    cache = _get_reverse_cache!(sens)
    cb = sens.solver.cb

    fill!(cache.dL_dx, zero(T))
    fill!(cache.dL_dλ, zero(T))
    fill!(cache.dL_dzl, zero(T))
    fill!(cache.dL_dzu, zero(T))
    fill!(full(cache.dl_dp), zero(T))
    fill!(full(cache.du_dp), zero(T))

    isnothing(dL_dx) || pack_x!(cache.dL_dx, cb, dL_dx)
    isnothing(dL_dλ) || pack_y!(cache.dL_dλ, cb, dL_dλ)
    if !isnothing(dL_dzl)
        pack_z!(variable(cache.dl_dp), cb, dL_dzl)
        cache.dL_dzl .= cache.dl_dp.values_lr
    end
    if !isnothing(dL_dzu)
        pack_z!(variable(cache.du_dp), cb, dL_dzu)
        cache.dL_dzu .= cache.du_dp.values_ur
    end
    return nothing
end

function _solve_vjp!(sens::MadDiffSolver{T}) where {T}
    cache = _get_reverse_cache!(sens)
    kkt = sens.kkt
    w = cache.kkt_rhs
    n_x = length(cache.dL_dx)

    fill!(full(w), zero(T))
    primal(w)[1:n_x] .= cache.dL_dx
    dual(w) .= cache.dL_dλ
    dual_lb(w) .= cache.dL_dzl
    dual_ub(w) .= cache.dL_dzu

    _adjoint_solve_with_refine!(sens, w, cache)
    return nothing
end

function _adjoint_solve_with_refine!(sens::MadDiffSolver{T}, w::AbstractKKTVector, cache) where {T}
    d = cache.kkt_sol
    work = cache.kkt_work

    copyto!(full(d), full(w))
    solver = sens.solver
    iterator = if sens.kkt === solver.kkt
        solver.iterator
    else
        RichardsonIterator(
            sens.kkt;
            opt=solver.iterator.opt,
            logger=solver.iterator.logger,
            cnt=solver.cnt,
        )
    end
    solver.cnt.linear_solver_time += @elapsed begin
        if adjoint_solve_refine!(d, iterator, w, work)
            # ok
        elseif improve!(sens.kkt.linear_solver)
            adjoint_solve_refine!(d, iterator, w, work)
        end
    end
    copyto!(full(w), full(d))
    return nothing
end
