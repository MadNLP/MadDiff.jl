function reverse_differentiate!(
    result::ReverseResult,
    sens::MadDiffSolver;
    dL_dx = nothing,
    dL_dy = nothing,
    dL_dzl = nothing,
    dL_dzu = nothing,
)
    pack_vjp!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu)
    solve_vjp!(sens)
    unpack_vjp!(result, sens)
    return result
end

function pack_vjp!(
    sens::MadDiffSolver{T};
    dL_dx = nothing,
    dL_dy = nothing,
    dL_dzl = nothing,
    dL_dzu = nothing,
) where {T}
    all(isnothing, (dL_dx, dL_dy, dL_dzl, dL_dzu)) &&
        throw(ArgumentError("At least one of dL_dx, dL_dy, dL_dzl, dL_dzu must be provided"))

    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    isnothing(dL_dx) || @lencheck n_x dL_dx
    isnothing(dL_dy) || @lencheck n_con dL_dy
    isnothing(dL_dzl) || @lencheck n_x dL_dzl
    isnothing(dL_dzu) || @lencheck n_x dL_dzu

    cache = get_reverse_cache!(sens)
    cb = sens.solver.cb

    fill!(cache.dL_dx, zero(T))
    fill!(cache.dL_dy, zero(T))
    fill!(cache.dL_dzl, zero(T))
    fill!(cache.dL_dzu, zero(T))
    fill!(full(cache.dzl_full), zero(T))
    fill!(full(cache.dzu_full), zero(T))

    isnothing(dL_dx) || pack_dx!(cache.dL_dx, cb, dL_dx)
    isnothing(dL_dy) || pack_dy!(cache.dL_dy, cb, dL_dy)
    isnothing(dL_dzl) || pack_dzl!(cache.dL_dzl, cb, dL_dzl, cache.dzl_full)
    isnothing(dL_dzu) || pack_dzu!(cache.dL_dzu, cb, dL_dzu, cache.dzu_full)
    return nothing
end

function solve_vjp!(sens::MadDiffSolver{T}) where {T}
    cache = get_reverse_cache!(sens)
    w = cache.kkt_rhs
    n_x = length(cache.dL_dx)

    fill!(full(w), zero(T))
    primal(w)[1:n_x] .= cache.dL_dx
    dual(w) .= cache.dL_dy
    dual_lb(w) .= cache.dL_dzl
    dual_ub(w) .= cache.dL_dzu

    _adjoint_solve_with_refine!(sens, w, cache)
    return nothing
end

function unpack_vjp!(result::ReverseResult, sens::MadDiffSolver)
    cache = get_reverse_cache!(sens)
    cb = sens.solver.cb

    unpack_dx!(result.dx, cb, primal(cache.kkt_rhs))
    unpack_y!(result.dy, cb, dual(cache.kkt_rhs))
    unpack_dzl!(result.dzl, cb, dual_lb(cache.kkt_rhs), cache.dzl_full)
    unpack_dzu!(result.dzu, cb, dual_ub(cache.kkt_rhs), cache.dzu_full)

    if !isnothing(sens.param_pullback)
        sens.param_pullback(result.grad_p, result.dx, result.dy, result.dzl, result.dzu, sens)
    end
    return result
end