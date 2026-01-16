"""
    ReverseResult{VT, GT}

# Fields
- `adj_x::VT`: Primal adjoint vector (length n_x)
- `adj_λ::VT`: Dual adjoint vector (length n_con)
- `adj_zl::VT`: Lower bound dual adjoint (length n_lb)
- `adj_zu::VT`: Upper bound dual adjoint (length n_ub)
- `grad_p::GT`: Parameter gradient vector (if `param_pullback` callback was set on solver, `nothing` otherwise)
"""
struct ReverseResult{VT, GT}
    adj_x::VT
    adj_λ::VT
    adj_zl::VT
    adj_zu::VT
    grad_p::GT
end

"""
    reverse_differentiate!(sens::MadDiffSolver; dL_dx=nothing, dL_dλ=nothing, dL_dzl=nothing, dL_dzu=nothing) -> ReverseResult

Compute reverse sensitivities (VJP) given loss gradients.

At least one of `dL_dx`, `dL_dλ`, `dL_dzl`, `dL_dzu` must be provided.
Input vectors may be mutated during computation.

If a `param_pullback` callback was provided when constructing the `MadDiffSolver`,
it will be called to compute parameter gradients.

# Arguments
- `dL_dx`: Gradient of loss with respect to primal variables (length n_x).
- `dL_dλ`: Gradient of loss with respect to constraint duals (length n_con).
- `dL_dzl`: Gradient of loss with respect to lower bound duals (length n_lb).
- `dL_dzu`: Gradient of loss with respect to upper bound duals (length n_ub).

# Returns
- `ReverseResult` containing:
  - `adj_x`: Primal adjoint vector
  - `adj_λ`: Constraint dual adjoint vector
  - `adj_zl`: Lower bound dual adjoint
  - `adj_zu`: Upper bound dual adjoint
  - `grad_p`: Parameter gradient vector (if `param_pullback` was set; `nothing` otherwise)
"""
function reverse_differentiate!(sens::MadDiffSolver; dL_dx = nothing, dL_dλ = nothing, dL_dzl = nothing, dL_dzu = nothing)
    all(isnothing, (dL_dx, dL_dλ, dL_dzl, dL_dzu)) &&
        throw(ArgumentError("At least one of dL_dx, dL_dλ, dL_dzl, dL_dzu must be provided"))

    dims = sens.dims
    isnothing(dL_dx) || @lencheck dims.n_x dL_dx
    isnothing(dL_dλ) || @lencheck dims.n_con dL_dλ
    isnothing(dL_dzl) || @lencheck dims.n_lb dL_dzl
    isnothing(dL_dzu) || @lencheck dims.n_ub dL_dzu

    cache = _get_reverse_cache!(sens)
    cb = sens.solver.cb

    dL_dx_kkt = _scale_dL_dx(dL_dx, cb)
    dL_dλ_scaled = _scale_dL_dλ(dL_dλ, cb, cache.dL_dλ_cache)
    _scale_dL_dz!(dL_dzl, cb)
    _scale_dL_dz!(dL_dzu, cb)

    sol = _solve_vjp!(sens.kkt, cache.work, dL_dx_kkt, dL_dλ_scaled, dL_dzl, dL_dzu)

    adj_x_kkt = _extract_sensitivities!(cache.adj_λ, cache.dzl, cache.dzu, sol, sens.solver)
    _unpack_sensitivity!(cache.dx_full, cb, adj_x_kkt)

    adj_x = cache.dx_full .* cb.obj_scale[]
    adj_λ = cache.adj_λ .* cb.con_scale
    adj_zl = copy(cache.dzl)
    adj_zu = copy(cache.dzu)

    grad_p = if !isnothing(sens.param_pullback)
        sens.param_pullback(cache.grad_p_buffer, adj_x, adj_λ, adj_zl, adj_zu, sens)
        copy(cache.grad_p_buffer)
    else
        nothing
    end

    return ReverseResult(adj_x, adj_λ, adj_zl, adj_zu, grad_p)
end

"""
    reverse_differentiate!(solver; dL_dx=nothing, dL_dλ=nothing, dL_dzl=nothing, dL_dzu=nothing, kwargs...) -> ReverseResult

Convenience function for one-shot reverse sensitivity computation.
For multiple gradient computations, use the `MadDiffSolver` API.
Input vectors may be mutated during computation.
"""
function reverse_differentiate!(solver::MadNLP.AbstractMadNLPSolver; dL_dx = nothing, dL_dλ = nothing, dL_dzl = nothing, dL_dzu = nothing, kwargs...)
    config = MadDiffConfig(; kwargs...)
    sens = MadDiffSolver(solver; config)
    return reverse_differentiate!(sens; dL_dx, dL_dλ, dL_dzl, dL_dzu)
end

struct ReverseCache{VT, VK, VB}
    work::VK
    dzl::VT
    dzu::VT
    adj_λ::VT
    dx_full::VT
    dL_dλ_cache::VT
    hv_buffer::VT
    jv_buffer::VT
    grad_lb_buffer::VT
    grad_ub_buffer::VT
    x_opt::VT
    y_opt::VT
    has_lb::VB
    has_ub::VB
    eq_scale::VT
    grad_p_buffer::VT
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
        has_lb[dims.var_ind_lb] .= true
        has_ub[dims.var_ind_ub] .= true

        y_opt = _zeros_like(x_array, T, dims.n_con)
        MadNLP.unpack_y!(y_opt, sens.solver.cb, sens.solver.y)

        eq_scale = _zeros_like(x_array, T, dims.n_con)
        eq_scale .= ifelse.(dims.is_eq, T(1 // 2), one(T))

        grad_p_buffer = _zeros_like(x_array, T, dims.n_params)

        sens.reverse_cache = ReverseCache(
            MadNLP.UnreducedKKTVector(sens.kkt),
            _zeros_like(x_array, T, dims.n_lb),
            _zeros_like(x_array, T, dims.n_ub),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_x),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_x),
            _zeros_like(x_array, T, dims.n_x),
            _zeros_like(x_array, T, dims.n_x),
            _zeros_like(x_array, T, dims.n_x),
            x_opt,
            y_opt,
            has_lb,
            has_ub,
            eq_scale,
            grad_p_buffer,
        )
    end
    return sens.reverse_cache
end

function _solve_vjp!(kkt, w, dL_dx, dL_dλ, dL_dzl, dL_dzu)
    _vjp_pre!(kkt, w, dL_dx, dL_dλ, dL_dzl, dL_dzu)
    _vjp_solve!(kkt, w)
    _vjp_post!(kkt, w)
    return w
end

function _vjp_pre!(kkt, w, dL_dx, dL_dλ, dL_dzl, dL_dzu)
    fill!(MadNLP.full(w), zero(eltype(MadNLP.full(w))))
    _set_primal_rhs!(w, dL_dx)
    _set_rhs!(MadNLP.dual(w), dL_dλ)
    _vjp_bound!(kkt, w.xp_lr, dL_dzl, Val(:lower))
    _vjp_bound!(kkt, w.xp_ur, dL_dzu, Val(:upper))
end
function _vjp_pre!(kkt::MadNLP.SparseUnreducedKKTSystem, w, dL_dx, dL_dλ, dL_dzl, dL_dzu)
    fill!(MadNLP.full(w), zero(eltype(MadNLP.full(w))))
    _set_primal_rhs!(w, dL_dx)
    _set_rhs!(MadNLP.dual(w), dL_dλ)
    _vjp_bound!(kkt, MadNLP.dual_lb(w), dL_dzl, Val(:lower))
    _vjp_bound!(kkt, MadNLP.dual_ub(w), dL_dzu, Val(:upper))
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

_vjp_bound!(kkt, xp_lr, ::Nothing, ::Val{:lower}) = nothing
_vjp_bound!(kkt, xp_lr, ::Nothing, ::Val{:upper}) = nothing
function _vjp_bound!(kkt, xp_lr, dL_dzl, ::Val{:lower})
    xp_lr .+= kkt.l_lower .* dL_dzl ./ kkt.l_diag
end
function _vjp_bound!(kkt, xp_ur, dL_dzu, ::Val{:upper})
    xp_ur .-= kkt.u_lower .* dL_dzu ./ kkt.u_diag
end
function _vjp_bound!(kkt::MadNLP.ScaledSparseKKTSystem, xp_lr, dL_dzl, ::Val{:lower})
    xp_lr .-= kkt.l_lower .* dL_dzl ./ kkt.l_diag
end
function _vjp_bound!(kkt::MadNLP.ScaledSparseKKTSystem, xp_ur, dL_dzu, ::Val{:upper})
    xp_ur .+= kkt.u_lower .* dL_dzu ./ kkt.u_diag
end
function _vjp_bound!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, dL_dzl, ::Val{:lower})
    wz .= dL_dzl
    wz .*= .-kkt.l_lower_aug
end
function _vjp_bound!(kkt::MadNLP.SparseUnreducedKKTSystem, wz, dL_dzu, ::Val{:upper})
    wz .= dL_dzu
    wz .*= kkt.u_lower_aug
end
# disambiguation
_vjp_bound!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing, ::Val{:lower}) = nothing  # COV_EXCL_LINE
_vjp_bound!(::MadNLP.SparseUnreducedKKTSystem, wz, ::Nothing, ::Val{:upper}) = nothing  # COV_EXCL_LINE
_vjp_bound!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing, ::Val{:lower}) = nothing  # COV_EXCL_LINE
_vjp_bound!(::MadNLP.ScaledSparseKKTSystem, xp, ::Nothing, ::Val{:upper}) = nothing  # COV_EXCL_LINE

_scale_dL_dx(::Nothing, cb) = nothing
_scale_dL_dx(dL_dx, cb) = _pack_x(dL_dx, cb)

_scale_dL_dλ(::Nothing, cb, cache) = nothing
function _scale_dL_dλ(dL_dλ, cb, cache)
    MadNLP.unpack_y!(cache, cb, dL_dλ)
    return cache
end

_scale_dL_dz!(::Nothing, cb) = nothing
_scale_dL_dz!(dL_dz, cb) = (dL_dz ./= cb.obj_scale[])

_set_rhs!(dest, ::Nothing) = nothing
_set_rhs!(dest, src) = dest .= src

_set_primal_rhs!(w, ::Nothing) = nothing
function _set_primal_rhs!(w, dL_dx)
    n_x = length(dL_dx)
    MadNLP.primal(w)[1:n_x] .= dL_dx
end
