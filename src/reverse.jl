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

    cache = _get_reverse_cache!(sens)
    cb = sens.solver.cb

    seed_x_kkt = _scale_seed_x(seed_x, cb)
    seed_λ_scaled = _scale_seed_λ(seed_λ, cb, cache.seed_λ_cache)
    _pack_seed_zl!(cache.seed_zl_cache, seed_zl, cb, cache.primal_buffer)
    _pack_seed_zu!(cache.seed_zu_cache, seed_zu, cb, cache.primal_buffer)

    sol = _solve_vjp!(sens.kkt, cache.work, seed_x_kkt, seed_λ_scaled, cache.seed_zl_cache, cache.seed_zu_cache)

    _extract_sensitivities!(cache.dx_kkt, cache.dλ, cache.dzl_kkt, cache.dzu_kkt, sol, sens.solver)
    _unpack_primal!(cache.dx_full, cb, cache.dx_kkt)

    MadNLP.unpack_y!(cache.dλ, cb, cache.dλ)

    copyto!(result.dx, cache.dx_full)
    result.dλ .= cache.dλ .* cb.obj_scale[]  #FIXME undo obj_scale?
    _unpack_zl!(result.dzl, cb, cache.dzl_kkt, cache.primal_buffer)
    _unpack_zu!(result.dzu, cb, cache.dzu_kkt, cache.primal_buffer)

    if !isnothing(sens.param_pullback) && !isnothing(result.grad_p)
        sens.param_pullback(cache.grad_p_cache, result.dx, result.dλ, result.dzl, result.dzu, sens)
        copyto!(result.grad_p, cache.grad_p_cache)
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

struct ReverseCache{VT, VK, PV}
    work::VK
    dzl_kkt::VT
    dzu_kkt::VT
    dλ::VT
    dx_full::VT
    primal_buffer::PV
    seed_λ_cache::VT
    seed_zl_cache::VT
    seed_zu_cache::VT
    eq_scale::VT
    grad_p_cache::VT
    dλ_scaled_cache::VT
    dx_kkt::VT
end

function _get_reverse_cache!(sens::MadDiffSolver)
    if isnothing(sens.reverse_cache)
        dims = sens.dims
        x_array = MadNLP.full(sens.solver.x)
        T = eltype(x_array)
        VT = typeof(x_array)

        eq_scale = _zeros_like(x_array, T, dims.n_con)
        eq_scale .= ifelse.(dims.is_eq, T(1 // 2), one(T))

        sens.reverse_cache = ReverseCache(
            MadNLP.UnreducedKKTVector(sens.kkt),
            _zeros_like(x_array, T, dims.n_lb),
            _zeros_like(x_array, T, dims.n_ub),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_x),
            MadNLP.PrimalVector(VT, dims.n_x_kkt, dims.n_slack, dims.idx_lb, dims.idx_ub),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_lb),
            _zeros_like(x_array, T, dims.n_ub),
            eq_scale,
            _zeros_like(x_array, T, dims.n_p),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_x_kkt),
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
end
function _vjp_pre!(kkt::MadNLP.SparseUnreducedKKTSystem, w, seed_x, seed_λ, seed_zl, seed_zu)
    fill!(MadNLP.full(w), zero(eltype(MadNLP.full(w))))
    _vjp_set_primal_rhs!(w, seed_x)
    _vjp_set_dual_rhs!(MadNLP.dual(w), seed_λ)
    _vjp_set_bound_lower!(kkt, MadNLP.dual_lb(w), seed_zl)
    _vjp_set_bound_upper!(kkt, MadNLP.dual_ub(w), seed_zu)
end

function _vjp_solve!(kkt::MadNLP.AbstractReducedKKTSystem, w)
    MadNLP.solve!(kkt.linear_solver, MadNLP.primal_dual(w))
end
function _vjp_solve!(kkt::MadNLP.AbstractCondensedKKTSystem, w)
    MadNLP.solve!(kkt, w)
end
function _vjp_solve!(kkt::MadNLP.SparseUnreducedKKTSystem, w)
    MadNLP.solve!(kkt.linear_solver, MadNLP.full(w))
end
function _vjp_solve!(kkt::MadNLP.ScaledSparseKKTSystem, w)
    MadNLP.primal(w) .*= kkt.scaling_factor
    MadNLP.solve!(kkt.linear_solver, MadNLP.primal_dual(w))
    MadNLP.primal(w) .*= kkt.scaling_factor
end

function _vjp_post!(kkt, w)
    MadNLP.dual_lb(w) .= kkt.l_lower .* w.xp_lr ./ kkt.l_diag
    MadNLP.dual_ub(w) .= .-kkt.u_lower .* w.xp_ur ./ kkt.u_diag
end
function _vjp_post!(kkt::MadNLP.ScaledSparseKKTSystem, w)
    MadNLP.dual_lb(w) .= .-kkt.l_lower .* w.xp_lr ./ kkt.l_diag
    MadNLP.dual_ub(w) .= kkt.u_lower .* w.xp_ur ./ kkt.u_diag
end
function _vjp_post!(kkt::MadNLP.SparseUnreducedKKTSystem, w)
    MadNLP.dual_lb(w) .*= .-kkt.l_lower_aug
    MadNLP.dual_ub(w) .*= kkt.u_lower_aug
end

_vjp_set_bound_lower!(kkt, xp_lr, ::Nothing) = nothing
_vjp_set_bound_upper!(kkt, xp_ur, ::Nothing) = nothing
function _vjp_set_bound_lower!(kkt, xp_lr, seed_zl)
    xp_lr .+= kkt.l_lower .* seed_zl ./ kkt.l_diag
end
function _vjp_set_bound_upper!(kkt, xp_ur, seed_zu)
    xp_ur .-= kkt.u_lower .* seed_zu ./ kkt.u_diag
end
function _vjp_set_bound_lower!(kkt::MadNLP.ScaledSparseKKTSystem, xp_lr, seed_zl)
    xp_lr .-= kkt.l_lower .* seed_zl ./ kkt.l_diag
end
function _vjp_set_bound_upper!(kkt::MadNLP.ScaledSparseKKTSystem, xp_ur, seed_zu)
    xp_ur .+= kkt.u_lower .* seed_zu ./ kkt.u_diag
end
function _vjp_set_bound_lower!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, seed_zl)
    wz .= seed_zl
    wz .*= .-kkt.l_lower_aug
end
function _vjp_set_bound_upper!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, seed_zu)
    wz .= seed_zu
    wz .*= kkt.u_lower_aug
end
# disambiguation
_vjp_set_bound_lower!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing) = nothing  # COV_EXCL_LINE
_vjp_set_bound_upper!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing) = nothing  # COV_EXCL_LINE
_vjp_set_bound_lower!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing) = nothing  # COV_EXCL_LINE
_vjp_set_bound_upper!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing) = nothing  # COV_EXCL_LINE

_scale_seed_x(::Nothing, cb) = nothing
_scale_seed_x(seed_x, cb) = _pack_primal(seed_x, cb)

_scale_seed_λ(::Nothing, cb, cache) = nothing
function _scale_seed_λ(seed_λ, cb, cache)
    MadNLP.unpack_y!(cache, cb, seed_λ)
    return cache
end

_pack_seed_zl!(cache, ::Nothing, cb, ::MadNLP.PrimalVector) = fill!(cache, zero(eltype(cache)))
_pack_seed_zu!(cache, ::Nothing, cb, ::MadNLP.PrimalVector) = fill!(cache, zero(eltype(cache)))
function _pack_seed_zl!(cache, seed_z, cb, pv::MadNLP.PrimalVector)
    _scatter_to_primal_vector!(pv, seed_z, cb)
    cache .= pv.values_lr ./ cb.obj_scale[]
end
function _pack_seed_zu!(cache, seed_z, cb, pv::MadNLP.PrimalVector)
    _scatter_to_primal_vector!(pv, seed_z, cb)
    cache .= pv.values_ur ./ cb.obj_scale[]
end

_vjp_set_dual_rhs!(dest, ::Nothing) = nothing
_vjp_set_dual_rhs!(dest, src) = dest .= src

_vjp_set_primal_rhs!(w, ::Nothing) = nothing
function _vjp_set_primal_rhs!(w, seed_x)
    n_x = length(seed_x)
    MadNLP.primal(w)[1:n_x] .= seed_x
end
