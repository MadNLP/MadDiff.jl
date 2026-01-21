"""
    ReverseResult{VT, GT}

# Fields
- `dx::VT`: Primal sensitivity vector (length n_x)
- `dλ::VT`: Dual sensitivity vector (length n_con)
- `dzl::VT`: Lower bound dual sensitivity (length n_lb)
- `dzu::VT`: Upper bound dual sensitivity (length n_ub)
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
        _zeros_like(x_array, T, dims.n_lb),
        _zeros_like(x_array, T, dims.n_ub),
        grad_p,
    )
end

"""
    reverse_differentiate!(result::ReverseResult, sens::MadDiffSolver; kwargs...) -> ReverseResult

Compute reverse sensitivities (VJP) given loss gradients, writing to pre-allocated `result`.

At least one of `seed_x`, `seed_λ`, `seed_zl`, `seed_zu` must be provided.
Input vectors are NOT modified.

# Keyword Arguments
- `seed_x`: Gradient of loss with respect to primal variables (length n_x).
- `seed_λ`: Gradient of loss with respect to constraint duals (length n_con).
- `seed_zl`: Gradient of loss with respect to lower bound duals (length n_lb).
- `seed_zu`: Gradient of loss with respect to upper bound duals (length n_ub).

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
    isnothing(seed_zl) || @lencheck dims.n_lb seed_zl
    isnothing(seed_zu) || @lencheck dims.n_ub seed_zu

    cache = _get_reverse_cache!(sens)
    cb = sens.solver.cb

    seed_x_kkt = _scale_seed_x(seed_x, cb)
    seed_λ_scaled = _scale_seed_λ(seed_λ, cb, cache.seed_λ_cache)
    seed_zl_scaled = _scale_seed_z(seed_zl, cb, cache.seed_zl_cache)
    seed_zu_scaled = _scale_seed_z(seed_zu, cb, cache.seed_zu_cache)

    sol = _solve_vjp!(sens.kkt, cache.work, seed_x_kkt, seed_λ_scaled, seed_zl_scaled, seed_zu_scaled)

    dx_kkt = _extract_sensitivities!(cache.dλ, cache.dzl, cache.dzu, sol, sens.solver)
    _unpack_primal!(cache.dx_full, cb, dx_kkt)

    MadNLP.unpack_y!(cache.dλ, cb, cache.dλ)

    copyto!(result.dx, cache.dx_full)
    obj_scale = cb.obj_scale[]
    obj_scale_inv = inv(obj_scale)
    result.dλ .= cache.dλ .* obj_scale
    result.dzl .= cache.dzl .* obj_scale_inv
    result.dzu .= cache.dzu .* obj_scale_inv

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
- `seed_x`: Gradient of loss with respect to primal variables (length n_x).
- `seed_λ`: Gradient of loss with respect to constraint duals (length n_con).
- `seed_zl`: Gradient of loss with respect to lower bound duals (length n_lb).
- `seed_zu`: Gradient of loss with respect to upper bound duals (length n_ub).

# Returns
- `ReverseResult` containing:
  - `dx`: Primal sensitivity vector
  - `dλ`: Constraint dual sensitivity vector
  - `dzl`: Lower bound dual sensitivity
  - `dzu`: Upper bound dual sensitivity
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

struct ReverseCache{VT, VK, VB}
    work::VK
    dzl::VT
    dzu::VT
    dλ::VT
    dx_full::VT
    seed_λ_cache::VT
    x_opt::VT
    y_opt::VT
    has_lb::VB
    has_ub::VB
    eq_scale::VT
    grad_p_cache::VT
    seed_zl_cache::VT
    seed_zu_cache::VT
    dλ_scaled_cache::VT
end

function _get_reverse_cache!(sens::MadDiffSolver)
    if isnothing(sens.reverse_cache)
        dims = sens.dims
        x_array = MadNLP.full(sens.solver.x)
        T = eltype(x_array)

        x_opt = _zeros_like(x_array, T, dims.n_x)
        x_kkt = MadNLP.primal(sens.solver.x)[1:dims.n_x_kkt]
        MadNLP.unpack_x!(x_opt, sens.solver.cb, x_kkt)

        has_lb = _falses_like(x_array, dims.n_x)
        has_ub = _falses_like(x_array, dims.n_x)
        has_lb[dims.var_idx_lb] .= true
        has_ub[dims.var_idx_ub] .= true

        y_opt = _zeros_like(x_array, T, dims.n_con)
        MadNLP.unpack_y!(y_opt, sens.solver.cb, sens.solver.y)

        eq_scale = _zeros_like(x_array, T, dims.n_con)
        eq_scale .= ifelse.(dims.is_eq, T(1 // 2), one(T))

        grad_p_cache = _zeros_like(x_array, T, dims.n_p)

        sens.reverse_cache = ReverseCache(
            MadNLP.UnreducedKKTVector(sens.kkt),
            _zeros_like(x_array, T, dims.n_lb),
            _zeros_like(x_array, T, dims.n_ub),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_x),
            _zeros_like(x_array, T, dims.n_con),
            x_opt,
            y_opt,
            has_lb,
            has_ub,
            eq_scale,
            grad_p_cache,
            _zeros_like(x_array, T, dims.n_lb),
            _zeros_like(x_array, T, dims.n_ub),
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
    _vjp_set_bound!(kkt, w.xp_lr, seed_zl, Val(:lower))
    _vjp_set_bound!(kkt, w.xp_ur, seed_zu, Val(:upper))
end
function _vjp_pre!(kkt::MadNLP.SparseUnreducedKKTSystem, w, seed_x, seed_λ, seed_zl, seed_zu)
    fill!(MadNLP.full(w), zero(eltype(MadNLP.full(w))))
    _vjp_set_primal_rhs!(w, seed_x)
    _vjp_set_dual_rhs!(MadNLP.dual(w), seed_λ)
    _vjp_set_bound!(kkt, MadNLP.dual_lb(w), seed_zl, Val(:lower))
    _vjp_set_bound!(kkt, MadNLP.dual_ub(w), seed_zu, Val(:upper))
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

_vjp_set_bound!(kkt, xp_lr, ::Nothing, ::Val{:lower}) = nothing
_vjp_set_bound!(kkt, xp_lr, ::Nothing, ::Val{:upper}) = nothing
function _vjp_set_bound!(kkt, xp_lr, seed_zl, ::Val{:lower})
    xp_lr .+= kkt.l_lower .* seed_zl ./ kkt.l_diag
end
function _vjp_set_bound!(kkt, xp_ur, seed_zu, ::Val{:upper})
    xp_ur .-= kkt.u_lower .* seed_zu ./ kkt.u_diag
end
function _vjp_set_bound!(kkt::MadNLP.ScaledSparseKKTSystem, xp_lr, seed_zl, ::Val{:lower})
    xp_lr .-= kkt.l_lower .* seed_zl ./ kkt.l_diag
end
function _vjp_set_bound!(kkt::MadNLP.ScaledSparseKKTSystem, xp_ur, seed_zu, ::Val{:upper})
    xp_ur .+= kkt.u_lower .* seed_zu ./ kkt.u_diag
end
function _vjp_set_bound!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, seed_zl, ::Val{:lower})
    wz .= seed_zl
    wz .*= .-kkt.l_lower_aug
end
function _vjp_set_bound!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, seed_zu, ::Val{:upper})
    wz .= seed_zu
    wz .*= kkt.u_lower_aug
end
# disambiguation
_vjp_set_bound!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing, ::Val{:lower}) = nothing  # COV_EXCL_LINE
_vjp_set_bound!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing, ::Val{:upper}) = nothing  # COV_EXCL_LINE
_vjp_set_bound!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing, ::Val{:lower}) = nothing  # COV_EXCL_LINE
_vjp_set_bound!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing, ::Val{:upper}) = nothing  # COV_EXCL_LINE

_scale_seed_x(::Nothing, cb) = nothing
_scale_seed_x(seed_x, cb) = _pack_primal(seed_x, cb)

_scale_seed_λ(::Nothing, cb, cache) = nothing
function _scale_seed_λ(seed_λ, cb, cache)
    MadNLP.unpack_y!(cache, cb, seed_λ)
    return cache
end

_scale_seed_z(::Nothing, cb, cache) = nothing
function _scale_seed_z(seed_z, cb, cache)
    copyto!(cache, seed_z)
    cache ./= cb.obj_scale[]
    return cache
end

_vjp_set_dual_rhs!(dest, ::Nothing) = nothing
_vjp_set_dual_rhs!(dest, src) = dest .= src

_vjp_set_primal_rhs!(w, ::Nothing) = nothing
function _vjp_set_primal_rhs!(w, seed_x)
    n_x = length(seed_x)
    MadNLP.primal(w)[1:n_x] .= seed_x
end
