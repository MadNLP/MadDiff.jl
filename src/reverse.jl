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

At least one of `seed_x`, `seed_λ`, `seed_zl`, `seed_zu` must be provided.
Input vectors are NOT modified.

# Keyword Arguments
- `seed_x`: Gradient of loss with respect to primal variables
- `seed_λ`: Gradient of loss with respect to constraint duals
- `seed_zl`: Gradient of loss with respect to lower bound duals
- `seed_zu`: Gradient of loss with respect to upper bound duals

# Returns
- The same `result` object, with updated values
"""
function reverse_differentiate!(
    result::ReverseResult,
    sens::MadDiffSolver;
    seed_x = nothing,
    seed_λ = nothing,
    seed_zl = nothing,
    seed_zu = nothing,
)
    all(isnothing, (seed_x, seed_λ, seed_zl, seed_zu)) &&
        throw(ArgumentError("At least one of seed_x, seed_λ, seed_zl, seed_zu must be provided"))

    dims = sens.dims
    isnothing(seed_x) || @lencheck dims.n_x seed_x
    isnothing(seed_λ) || @lencheck dims.n_con seed_λ
    isnothing(seed_zl) || @lencheck dims.n_x seed_zl
    isnothing(seed_zu) || @lencheck dims.n_x seed_zu

    shared = _get_shared_cache!(sens)
    cache = _get_reverse_cache!(sens)
    cb = sens.solver.cb

    seed_x_kkt = _pack_seed_x!(cache.seed_x, seed_x, cb)
    seed_λ_scaled = _pack_seed_λ!(cache.seed_λ, seed_λ, cb)
    _pack_seed_z!(cache.seed_zl, seed_zl, cb, shared.zl_buffer, shared.zl_buffer.values_lr)
    _pack_seed_z!(cache.seed_zu, seed_zu, cb, shared.zu_buffer, shared.zu_buffer.values_ur)

    sol = _solve_vjp!(sens.kkt, shared.rhs, seed_x_kkt, seed_λ_scaled, cache.seed_zl, cache.seed_zu)

    _extract_sensitivities!(shared.dx_kkt, shared.dλ, shared.dzl_kkt, shared.dzu_kkt, sol, sens.solver)
    _unpack_primal!(shared.dx_full, cb, shared.dx_kkt)

    MadNLP.unpack_y!(shared.dλ, cb, shared.dλ)

    copyto!(result.dx, shared.dx_full)
    result.dλ .= shared.dλ .* cb.obj_scale[]  #FIXME undo obj_scale?
    _unpack_z!(result.dzl, cb, shared.dzl_kkt, shared.zl_buffer, shared.zl_buffer.values_lr)
    _unpack_z!(result.dzu, cb, shared.dzu_kkt, shared.zu_buffer, shared.zu_buffer.values_ur)

    if !isnothing(sens.param_pullback) && !isnothing(result.grad_p)
        sens.param_pullback(cache.grad_p, result.dx, result.dλ, result.dzl, result.dzu, sens)
        copyto!(result.grad_p, cache.grad_p)
    end

    return result
end

"""
    reverse_differentiate!(sens::MadDiffSolver; seed_x=nothing, seed_λ=nothing, seed_zl=nothing, seed_zu=nothing) -> ReverseResult

Compute reverse sensitivities (VJP) given loss gradients.

Allocates a new ReverseResult. For batch processing, use the pre-allocated variant
`reverse_differentiate!(result, sens; ...)` to avoid allocations.

At least one of `seed_x`, `seed_λ`, `seed_zl`, `seed_zu` must be provided.
Input vectors are NOT modified.

# Arguments
- `seed_x`: Gradient of loss with respect to primal variables
- `seed_λ`: Gradient of loss with respect to constraint duals
- `seed_zl`: Gradient of loss with respect to lower bound duals
- `seed_zu`: Gradient of loss with respect to upper bound duals

# Returns
- `ReverseResult` containing:
  - `dx`: Primal sensitivity vector
  - `dλ`: Constraint dual sensitivity vector
  - `dzl`: Lower bound dual sensitivity vector
  - `dzu`: Upper bound dual sensitivity vector
  - `grad_p`: Parameter gradient vector (if `param_pullback` was set; `nothing` otherwise)
"""
function reverse_differentiate!(sens::MadDiffSolver; seed_x = nothing, seed_λ = nothing, seed_zl = nothing, seed_zu = nothing)
    result = ReverseResult(sens)
    return reverse_differentiate!(result, sens; seed_x, seed_λ, seed_zl, seed_zu)
end

"""
    reverse_differentiate!(solver; seed_x=nothing, seed_λ=nothing, seed_zl=nothing, seed_zu=nothing, kwargs...) -> ReverseResult

Convenience function for one-shot reverse sensitivity computation.
For multiple gradient computations, use the `MadDiffSolver` API.
Input vectors are NOT modified.
"""
function reverse_differentiate!(solver::MadNLP.AbstractMadNLPSolver; seed_x = nothing, seed_λ = nothing, seed_zl = nothing, seed_zu = nothing, kwargs...)
    config = MadDiffConfig(; kwargs...)
    sens = MadDiffSolver(solver; config)
    return reverse_differentiate!(sens; seed_x, seed_λ, seed_zl, seed_zu)
end

struct ReverseCache{VT}
    seed_x::VT
    seed_λ::VT
    seed_zl::VT
    seed_zu::VT
    eq_scale::VT
    grad_p::VT
    dλ_scaled::VT
end

function _get_reverse_cache!(sens::MadDiffSolver)
    if isnothing(sens.reverse_cache)
        cb = sens.solver.cb
        dims = sens.dims
        x_array = MadNLP.full(sens.solver.x)
        T = eltype(x_array)

        sens.reverse_cache = ReverseCache(
            _zeros_like(x_array, T, cb.nvar),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, length(cb.ind_lb)),
            _zeros_like(x_array, T, length(cb.ind_ub)),
            ifelse.(dims.is_eq, T(1 // 2), one(T)),
            _zeros_like(x_array, T, dims.n_p),
            _zeros_like(x_array, T, dims.n_con),
        )
    end
    return sens.reverse_cache
end

function _solve_vjp!(kkt, w, seed_x, seed_λ, seed_zl, seed_zu)
    _vjp_pre!(kkt, w, seed_x, seed_λ, seed_zl, seed_zu)
    _vjp_solve!(kkt, w)
    _vjp_post!(kkt, w)
    return w
end

function _vjp_pre!(kkt, w, seed_x, seed_λ, seed_zl, seed_zu)
    fill!(MadNLP.full(w), zero(eltype(MadNLP.full(w))))
    _vjp_set_primal_rhs!(w, seed_x)
    _vjp_set_dual_rhs!(MadNLP.dual(w), seed_λ)
    _vjp_set_bound_lower!(kkt, w.xp_lr, seed_zl)
    _vjp_set_bound_upper!(kkt, w.xp_ur, seed_zu)
    return nothing
end
function _vjp_pre!(kkt::MadNLP.SparseUnreducedKKTSystem, w, seed_x, seed_λ, seed_zl, seed_zu)
    fill!(MadNLP.full(w), zero(eltype(MadNLP.full(w))))
    _vjp_set_primal_rhs!(w, seed_x)
    _vjp_set_dual_rhs!(MadNLP.dual(w), seed_λ)
    _vjp_set_bound_lower!(kkt, MadNLP.dual_lb(w), seed_zl)
    _vjp_set_bound_upper!(kkt, MadNLP.dual_ub(w), seed_zu)
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
function _vjp_solve!(kkt::MadNLP.SparseUnreducedKKTSystem, w)
    MadNLP.solve!(kkt.linear_solver, MadNLP.full(w))
    return nothing
end
function _vjp_solve!(kkt::MadNLP.ScaledSparseKKTSystem, w)
    MadNLP.primal(w) .*= kkt.scaling_factor
    MadNLP.solve!(kkt.linear_solver, MadNLP.primal_dual(w))
    MadNLP.primal(w) .*= kkt.scaling_factor
    return nothing
end

function _vjp_post!(kkt, w)
    MadNLP.dual_lb(w) .= kkt.l_lower .* w.xp_lr ./ kkt.l_diag
    MadNLP.dual_ub(w) .= .-kkt.u_lower .* w.xp_ur ./ kkt.u_diag
    return nothing
end
function _vjp_post!(kkt::MadNLP.ScaledSparseKKTSystem, w)
    MadNLP.dual_lb(w) .= .-kkt.l_lower .* w.xp_lr ./ kkt.l_diag
    MadNLP.dual_ub(w) .= kkt.u_lower .* w.xp_ur ./ kkt.u_diag
    return nothing
end
function _vjp_post!(kkt::MadNLP.SparseUnreducedKKTSystem, w)
    MadNLP.dual_lb(w) .*= .-kkt.l_lower_aug
    MadNLP.dual_ub(w) .*= kkt.u_lower_aug
    return nothing
end

_vjp_set_bound_lower!(kkt, xp_lr, ::Nothing) = nothing
_vjp_set_bound_upper!(kkt, xp_ur, ::Nothing) = nothing
function _vjp_set_bound_lower!(kkt, xp_lr, seed_zl)
    xp_lr .+= kkt.l_lower .* seed_zl ./ kkt.l_diag
    return nothing
end
function _vjp_set_bound_upper!(kkt, xp_ur, seed_zu)
    xp_ur .-= kkt.u_lower .* seed_zu ./ kkt.u_diag
    return nothing
end
function _vjp_set_bound_lower!(kkt::MadNLP.ScaledSparseKKTSystem, xp_lr, seed_zl)
    xp_lr .-= kkt.l_lower .* seed_zl ./ kkt.l_diag
    return nothing
end
function _vjp_set_bound_upper!(kkt::MadNLP.ScaledSparseKKTSystem, xp_ur, seed_zu)
    xp_ur .+= kkt.u_lower .* seed_zu ./ kkt.u_diag
    return nothing
end
function _vjp_set_bound_lower!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, seed_zl)
    wz .= seed_zl
    wz .*= .-kkt.l_lower_aug
    return nothing
end
function _vjp_set_bound_upper!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, seed_zu)
    wz .= seed_zu
    wz .*= kkt.u_lower_aug
    return nothing
end
_vjp_set_bound_lower!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing) = nothing  # COV_EXCL_LINE
_vjp_set_bound_upper!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing) = nothing  # COV_EXCL_LINE
_vjp_set_bound_lower!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing) = nothing  # COV_EXCL_LINE
_vjp_set_bound_upper!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing) = nothing  # COV_EXCL_LINE

_vjp_set_dual_rhs!(dest, ::Nothing) = nothing
_vjp_set_dual_rhs!(dest, src) = (dest .= src;)

_vjp_set_primal_rhs!(w, ::Nothing) = nothing
function _vjp_set_primal_rhs!(w, seed_x)
    n_x = length(seed_x)
    MadNLP.primal(w)[1:n_x] .= seed_x
    return nothing
end
