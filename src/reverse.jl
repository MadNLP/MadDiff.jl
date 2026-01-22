"""
    ReverseResult{VT, GT}

# Fields
- `dx::VT`: Primal sensitivity vector
- `dλ::VT`: Dual sensitivity vector
- `dzl::VT`: Lower bound dual sensitivity
- `dzu::VT`: Upper bound dual sensitivity
- `grad_p::GT`: Parameter gradient vector (if `param_pullback` callback was set on solver, `nothing` otherwise)
"""
struct ReverseResult{VT, GT}
    dx::VT
    dλ::VT
    dzl::VT
    dzu::VT
    grad_p::GT
end


function ReverseResult(sens::MadDiffSolver)
    x_array = MadNLP.full(sens.solver.x)
    T = eltype(x_array)
    dims = sens.dims
    grad_p = dims.n_p > 0 ? _zeros_like(x_array, T, dims.n_p) : nothing
    return ReverseResult(
        _zeros_like(x_array, T, dims.n_x),
        _zeros_like(x_array, T, dims.n_con),
        _zeros_like(x_array, T, dims.n_x),
        _zeros_like(x_array, T, dims.n_x),
        grad_p,
    )
end

"""
    reverse_differentiate!(result::ReverseResult, sens::MadDiffSolver; kwargs...) -> ReverseResult

Compute reverse sensitivities (VJP) given loss gradients, writing to pre-allocated `result`.

# Keyword Arguments
- `dL_dx`: Gradient of loss with respect to primal variables (∂L/∂x)
- `dL_dλ`: Gradient of loss with respect to constraint duals (∂L/∂λ)
- `dL_dzl`: Gradient of loss with respect to lower bound duals (∂L/∂zl)
- `dL_dzu`: Gradient of loss with respect to upper bound duals (∂L/∂zu)

# Returns
- The same `result` object, with updated values
"""
function reverse_differentiate!(
    result::ReverseResult,
    sens::MadDiffSolver;
    dL_dx = nothing,
    dL_dλ = nothing,
    dL_dzl = nothing,
    dL_dzu = nothing,
)
    all(isnothing, (dL_dx, dL_dλ, dL_dzl, dL_dzu)) &&
        throw(ArgumentError("At least one of dL_dx, dL_dλ, dL_dzl, dL_dzu must be provided"))

    dims = sens.dims
    isnothing(dL_dx) || @lencheck dims.n_x dL_dx
    isnothing(dL_dλ) || @lencheck dims.n_con dL_dλ
    isnothing(dL_dzl) || @lencheck dims.n_x dL_dzl
    isnothing(dL_dzu) || @lencheck dims.n_x dL_dzu

    shared = _get_shared_cache!(sens)
    cache = _get_reverse_cache!(sens)
    cb = sens.solver.cb

    _pack_dL_dx!(cache.dL_dx, dL_dx, cb)
    _pack_dL_dλ!(cache.dL_dλ, dL_dλ, cb)
    _pack_dL_dz!(cache.dL_dzl, dL_dzl, cb, shared.zl_buffer, shared.zl_buffer.values_lr)
    _pack_dL_dz!(cache.dL_dzu, dL_dzu, cb, shared.zu_buffer, shared.zu_buffer.values_ur)

    _reverse_solve!(sens)

    _reverse_extract!(result, sens)

    return result
end

function _reverse_solve!(sens::MadDiffSolver)
    shared = sens.shared_cache
    cache = sens.reverse_cache

    _solve_vjp!(sens.kkt, shared.rhs, cache.dL_dx, cache.dL_dλ, cache.dL_dzl, cache.dL_dzu)

    _extract_sensitivities!(shared.dx_kkt, shared.dλ, shared.dzl_kkt, shared.dzu_kkt, shared.rhs, sens.solver)
    return nothing
end

function _reverse_extract!(result::ReverseResult, sens::MadDiffSolver)
    shared = sens.shared_cache
    cache = sens.reverse_cache
    cb = sens.solver.cb

    _unpack_primal!(shared.dx_full, cb, shared.dx_kkt)
    MadNLP.unpack_y!(shared.dλ, cb, shared.dλ)

    copyto!(result.dx, shared.dx_full)
    result.dλ .= shared.dλ .* cb.obj_scale[]  #FIXME undo obj_scale?
    _unpack_z!(result.dzl, cb, shared.dzl_kkt, shared.zl_buffer, shared.zl_buffer.values_lr)
    _unpack_z!(result.dzu, cb, shared.dzu_kkt, shared.zu_buffer, shared.zu_buffer.values_ur)

    if !isnothing(sens.param_pullback)
        sens.param_pullback(cache.grad_p, result.dx, result.dλ, result.dzl, result.dzu, sens)
        copyto!(result.grad_p, cache.grad_p)
    end
    return result
end

function _solve_vjp!(kkt, w, dL_dx, dL_dλ, dL_dzl, dL_dzu)
    _vjp_pre!(kkt, w, dL_dx, dL_dλ, dL_dzl, dL_dzu)
    _vjp_solve!(kkt, w)
    _vjp_post!(kkt, w)
    return nothing
end

# generic

_vjp_set_primal_rhs!(w, ::Nothing) = nothing
function _vjp_set_primal_rhs!(w, dL_dx)
    n_x = length(dL_dx)
    MadNLP.primal(w)[1:n_x] .= dL_dx
    return nothing
end

_vjp_set_dual_rhs!(dest, ::Nothing) = nothing
_vjp_set_dual_rhs!(dest, src) = (dest .= src;)

_vjp_set_bound_lower!(kkt, xp_lr, ::Nothing) = nothing
_vjp_set_bound_upper!(kkt, xp_ur, ::Nothing) = nothing
function _vjp_set_bound_lower!(kkt, xp_lr, dL_dzl)
    xp_lr .+= kkt.l_lower .* dL_dzl ./ kkt.l_diag
    return nothing
end
function _vjp_set_bound_upper!(kkt, xp_ur, dL_dzu)
    xp_ur .-= kkt.u_lower .* dL_dzu ./ kkt.u_diag
    return nothing
end

function _vjp_pre!(kkt, w, dL_dx, dL_dλ, dL_dzl, dL_dzu)
    fill!(MadNLP.full(w), zero(eltype(MadNLP.full(w))))
    _vjp_set_primal_rhs!(w, dL_dx)
    _vjp_set_dual_rhs!(MadNLP.dual(w), dL_dλ)
    _vjp_set_bound_lower!(kkt, w.xp_lr, dL_dzl)
    _vjp_set_bound_upper!(kkt, w.xp_ur, dL_dzu)
    return nothing
end

function _vjp_solve!(kkt::MadNLP.AbstractReducedKKTSystem, w)
    MadNLP.solve!(kkt.linear_solver, MadNLP.primal_dual(w))
    return nothing
end
function _vjp_solve!(kkt::MadNLP.AbstractCondensedKKTSystem, w)
    MadNLP.solve!(kkt, w)
    return nothing
end

function _vjp_post!(kkt, w)
    MadNLP.dual_lb(w) .= kkt.l_lower .* w.xp_lr ./ kkt.l_diag
    MadNLP.dual_ub(w) .= .-kkt.u_lower .* w.xp_ur ./ kkt.u_diag
    return nothing
end

# SparseUnreducedKKTSystem

function _vjp_set_bound_lower!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, dL_dzl)
    wz .= dL_dzl
    wz .*= .-kkt.l_lower_aug
    return nothing
end
function _vjp_set_bound_upper!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, dL_dzu)
    wz .= dL_dzu
    wz .*= kkt.u_lower_aug
    return nothing
end
function _vjp_pre!(kkt::MadNLP.SparseUnreducedKKTSystem, w, dL_dx, dL_dλ, dL_dzl, dL_dzu)
    fill!(MadNLP.full(w), zero(eltype(MadNLP.full(w))))
    _vjp_set_primal_rhs!(w, dL_dx)
    _vjp_set_dual_rhs!(MadNLP.dual(w), dL_dλ)
    _vjp_set_bound_lower!(kkt, MadNLP.dual_lb(w), dL_dzl)
    _vjp_set_bound_upper!(kkt, MadNLP.dual_ub(w), dL_dzu)
    return nothing
end
function _vjp_solve!(kkt::MadNLP.SparseUnreducedKKTSystem, w)
    MadNLP.solve!(kkt.linear_solver, MadNLP.full(w))
    return nothing
end
function _vjp_post!(kkt::MadNLP.SparseUnreducedKKTSystem, w)
    MadNLP.dual_lb(w) .*= .-kkt.l_lower_aug
    MadNLP.dual_ub(w) .*= kkt.u_lower_aug
    return nothing
end

# ScaledSparseKKTSystem

function _vjp_set_bound_lower!(kkt::MadNLP.ScaledSparseKKTSystem, xp_lr, dL_dzl)
    xp_lr .-= kkt.l_lower .* dL_dzl ./ kkt.l_diag
    return nothing
end
function _vjp_set_bound_upper!(kkt::MadNLP.ScaledSparseKKTSystem, xp_ur, dL_dzu)
    xp_ur .+= kkt.u_lower .* dL_dzu ./ kkt.u_diag
    return nothing
end
function _vjp_solve!(kkt::MadNLP.ScaledSparseKKTSystem, w)
    MadNLP.primal(w) .*= kkt.scaling_factor
    MadNLP.solve!(kkt.linear_solver, MadNLP.primal_dual(w))
    MadNLP.primal(w) .*= kkt.scaling_factor
    return nothing
end
function _vjp_post!(kkt::MadNLP.ScaledSparseKKTSystem, w)
    MadNLP.dual_lb(w) .= .-kkt.l_lower .* w.xp_lr ./ kkt.l_diag
    MadNLP.dual_ub(w) .= kkt.u_lower .* w.xp_ur ./ kkt.u_diag
    return nothing
end

_vjp_set_bound_lower!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing) = nothing  # COV_EXCL_LINE
_vjp_set_bound_upper!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing) = nothing  # COV_EXCL_LINE
_vjp_set_bound_lower!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing) = nothing  # COV_EXCL_LINE
_vjp_set_bound_upper!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing) = nothing  # COV_EXCL_LINE
