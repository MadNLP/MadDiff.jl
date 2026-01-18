"""
    ForwardResult{VT}

# Fields
- `dx::VT`: Primal variable sensitivities
- `dλ::VT`: Constraint dual sensitivities
- `dzl::VT`: Lower bound dual sensitivities
- `dzu::VT`: Upper bound dual sensitivities
"""
struct ForwardResult{VT}
    dx::VT
    dλ::VT
    dzl::VT
    dzu::VT
end

function ForwardResult(sens::MadDiffSolver)
    x_array = MadNLP.full(sens.solver.x)
    T = eltype(x_array)
    dims = sens.dims
    return ForwardResult(
        _zeros_like(x_array, T, dims.n_x),
        _zeros_like(x_array, T, dims.n_con),
        _zeros_like(x_array, T, dims.n_lb),
        _zeros_like(x_array, T, dims.n_ub),
    )
end

"""
    forward_differentiate!(result::ForwardResult, sens::MadDiffSolver; kwargs...) -> ForwardResult

Compute forward sensitivities (JVP) for a parameter perturbation, writing to pre-allocated `result`.

At least one of the parameter-direction inputs must be provided.
Input vectors are NOT modified.

# Keyword Arguments
- `d2L_dxdp`: Parameter-Lagrangian cross derivative times Δp: (∂²L/∂x∂p) * Δp (length = n_x)
- `dg_dp`: Parameter-constraint LHS Jacobian times Δp: (∂g/∂p) * Δp (length = n_con)
- `dl_dp`: Variable lower bound perturbation times Δp: (∂l/∂p) * Δp (length = n_lb)
- `du_dp`: Variable upper bound perturbation times Δp: (∂u/∂p) * Δp (length = n_ub)
- `dlcon_dp`: Constraint lower bound perturbation times Δp: (∂lcon/∂p) * Δp (length = n_con)
- `ducon_dp`: Constraint upper bound perturbation times Δp: (∂ucon/∂p) * Δp (length = n_con)

# Notes
For equality constraints (lcon[i] == ucon[i]), provide the same perturbation for both
dlcon_dp[i] and ducon_dp[i]. The implementation uses (dlcon_dp + ducon_dp)/2 for equality constraints.
For entries where lcon or ucon is ±Inf, the corresponding dlcon_dp/ducon_dp value is ignored.

# Returns
- The same `result` object, with updated values
"""
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
    all(isnothing, (d2L_dxdp, dg_dp, dl_dp, du_dp, dlcon_dp, ducon_dp)) &&
        throw(ArgumentError("At least one of d2L_dxdp, dg_dp, dl_dp, du_dp, dlcon_dp, ducon_dp must be provided"))

    dims = sens.dims
    !isnothing(d2L_dxdp) && @lencheck dims.n_x d2L_dxdp
    !isnothing(dg_dp) && @lencheck dims.n_con dg_dp
    !isnothing(dl_dp) && @lencheck dims.n_lb dl_dp
    !isnothing(du_dp) && @lencheck dims.n_ub du_dp
    !isnothing(dlcon_dp) && @lencheck dims.n_con dlcon_dp
    !isnothing(ducon_dp) && @lencheck dims.n_con ducon_dp

    cache = _get_forward_cache!(sens)
    cb = sens.solver.cb

    d2L_dxdp_kkt = _copy_and_scale_d2L_dxdp(d2L_dxdp, cb, cache)
    dg_dp_scaled = _copy_and_scale_dg_dp(dg_dp, cb, cache)
    dlcon_dp_scaled = _copy_and_scale_dlcon_dp(dlcon_dp, cb, cache)
    ducon_dp_scaled = _copy_and_scale_ducon_dp(ducon_dp, cb, cache)

    sol = _solve_jvp!(sens.kkt, cache.work, d2L_dxdp_kkt, dg_dp_scaled, dl_dp, du_dp, dlcon_dp_scaled, ducon_dp_scaled, dims)

    dx_kkt = _extract_sensitivities!(cache.dλ, cache.dzl, cache.dzu, sol, sens.solver)

    _unpack_primal!(cache.dx_full, cb, dx_kkt)
    MadNLP.unpack_y!(cache.dλ, cb, cache.dλ)

    copyto!(result.dx, cache.dx_full)
    copyto!(result.dλ, cache.dλ)
    obj_scale_inv = inv(cb.obj_scale[])
    result.dzl .= cache.dzl .* obj_scale_inv
    result.dzu .= cache.dzu .* obj_scale_inv

    return result
end

"""
    forward_differentiate!(sens::MadDiffSolver; kwargs...) -> ForwardResult

Compute forward sensitivities (JVP) for a parameter perturbation.

Allocates a new ForwardResult. For batch processing, use the pre-allocated variant
`forward_differentiate!(result, sens; ...)` to avoid allocations.

At least one of the parameter-direction inputs must be provided.
Input vectors are NOT modified.

# Keyword Arguments
- `d2L_dxdp`: Parameter-Lagrangian cross derivative times Δp: (∂²L/∂x∂p) * Δp (length = n_x)
- `dg_dp`: Parameter-constraint LHS Jacobian times Δp: (∂g/∂p) * Δp (length = n_con)
- `dl_dp`: Variable lower bound perturbation times Δp: (∂l/∂p) * Δp (length = n_lb)
- `du_dp`: Variable upper bound perturbation times Δp: (∂u/∂p) * Δp (length = n_ub)
- `dlcon_dp`: Constraint lower bound perturbation times Δp: (∂lcon/∂p) * Δp (length = n_con)
- `ducon_dp`: Constraint upper bound perturbation times Δp: (∂ucon/∂p) * Δp (length = n_con)

# Notes
For equality constraints (lcon[i] == ucon[i]), provide the same perturbation for both
dlcon_dp[i] and ducon_dp[i]. The implementation uses (dlcon_dp + ducon_dp)/2 for equality constraints.
For entries where lcon or ucon is ±Inf, the corresponding dlcon_dp/ducon_dp value is ignored.
To obtain a full Jacobian w.r.t. parameters, call this once per parameter direction.

# Returns
- `ForwardResult` containing dx, dλ, dzl, dzu sensitivity vectors
"""
function forward_differentiate!(
    sens::MadDiffSolver;
    d2L_dxdp = nothing,
    dg_dp = nothing,
    dl_dp = nothing,
    du_dp = nothing,
    dlcon_dp = nothing,
    ducon_dp = nothing,
)
    result = ForwardResult(sens)
    return forward_differentiate!(result, sens; d2L_dxdp, dg_dp, dl_dp, du_dp, dlcon_dp, ducon_dp)
end

"""
    forward_differentiate!(solver; kwargs...) -> ForwardResult

Convenience function for one-shot forward sensitivity computation.
For multiple perturbations, use `MadDiffSolver`.

# Arguments
- `solver`: A solved `MadNLP.AbstractMadNLPSolver`
- `kwargs...`: Passed to `forward_differentiate!` and `MadDiffConfig`
"""
function forward_differentiate!(solver::MadNLP.AbstractMadNLPSolver;
    d2L_dxdp = nothing, dg_dp = nothing, dl_dp = nothing, du_dp = nothing,
    dlcon_dp = nothing, ducon_dp = nothing, kwargs...
)
    config = MadDiffConfig(; kwargs...)
    sens = MadDiffSolver(solver; config)
    return forward_differentiate!(sens; d2L_dxdp, dg_dp, dl_dp, du_dp, dlcon_dp, ducon_dp)
end

struct ForwardCache{VT, VK}
    work::VK
    dzl::VT
    dzu::VT
    dλ::VT
    dx_full::VT
    d2L_dxdp_cache::VT
    dg_dp_cache::VT
    dlcon_dp_cache::VT
    ducon_dp_cache::VT
end

function _get_forward_cache!(sens::MadDiffSolver)
    if isnothing(sens.forward_cache)
        dims = sens.dims
        x_array = MadNLP.full(sens.solver.x)
        T = eltype(x_array)
        sens.forward_cache = ForwardCache(
            MadNLP.UnreducedKKTVector(sens.kkt),
            _zeros_like(x_array, T, dims.n_lb),
            _zeros_like(x_array, T, dims.n_ub),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_x),
            _zeros_like(x_array, T, dims.n_x),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_con),
        )
    end
    return sens.forward_cache
end

_copy_and_scale_d2L_dxdp(::Nothing, cb, cache) = nothing
function _copy_and_scale_d2L_dxdp(d2L_dxdp, cb, cache)
    copyto!(cache.d2L_dxdp_cache, d2L_dxdp)
    v = _pack_primal(cache.d2L_dxdp_cache, cb)
    v .*= cb.obj_scale[]
    return v
end

_copy_and_scale_dg_dp(::Nothing, cb, cache) = nothing
function _copy_and_scale_dg_dp(dg_dp, cb, cache)
    copyto!(cache.dg_dp_cache, dg_dp)
    cache.dg_dp_cache .*= cb.con_scale
    return cache.dg_dp_cache
end

_copy_and_scale_dlcon_dp(::Nothing, cb, cache) = nothing
function _copy_and_scale_dlcon_dp(dlcon_dp, cb, cache)
    copyto!(cache.dlcon_dp_cache, dlcon_dp)
    cache.dlcon_dp_cache .*= cb.con_scale
    return cache.dlcon_dp_cache
end

_copy_and_scale_ducon_dp(::Nothing, cb, cache) = nothing
function _copy_and_scale_ducon_dp(ducon_dp, cb, cache)
    copyto!(cache.ducon_dp_cache, ducon_dp)
    cache.ducon_dp_cache .*= cb.con_scale
    return cache.ducon_dp_cache
end

_get_bound_scale(kkt::MadNLP.AbstractReducedKKTSystem) = (kkt.l_lower, kkt.u_lower)
_get_bound_scale(kkt::MadNLP.AbstractCondensedKKTSystem) = (kkt.l_lower, kkt.u_lower)
_get_bound_scale(::MadNLP.SparseUnreducedKKTSystem) = (nothing, nothing)

function _solve_jvp!(kkt, w, d2L_dxdp_vec, dg_dp_vec, dl_dp, du_dp, dlcon_dp, ducon_dp, dims)
    _jvp_pre!(kkt, w, d2L_dxdp_vec, dg_dp_vec, dl_dp, du_dp, dlcon_dp, ducon_dp, dims)
    _jvp_solve!(kkt, w)
    _jvp_post!(kkt, w)
    return w
end

function _jvp_pre!(kkt, w, d2L_dxdp_vec, dg_dp_vec, dl_dp, du_dp, dlcon_dp, ducon_dp, dims)
    T = eltype(MadNLP.full(w))
    fill!(MadNLP.full(w), zero(T))

    _jvp_set_primal_lag!(MadNLP.primal(w), d2L_dxdp_vec)
    _jvp_set_dual_lhs!(MadNLP.dual(w), dg_dp_vec)
    _jvp_set_dual_rhs!(MadNLP.dual(w), dlcon_dp, ducon_dp, dims.is_eq)

    l_scale, u_scale = _get_bound_scale(kkt)
    _jvp_set_dual_bound_var!(MadNLP.dual_lb(w), l_scale, dl_dp, Val(:lower))
    _jvp_set_dual_bound_var!(MadNLP.dual_ub(w), u_scale, du_dp, Val(:upper))
    _jvp_set_dual_bound_con!(MadNLP.dual_lb(w), l_scale, dlcon_dp, dims.slack_lb_pos, dims.slack_lb_con, Val(:lower))
    _jvp_set_dual_bound_con!(MadNLP.dual_ub(w), u_scale, ducon_dp, dims.slack_ub_pos, dims.slack_ub_con, Val(:upper))

    return w
end

_jvp_solve!(kkt, w) = MadNLP.solve!(kkt, w)
_jvp_post!(kkt, w) = nothing

_jvp_set_primal_lag!(primal, ::Nothing) = nothing
function _jvp_set_primal_lag!(primal, d2L_dxdp_vec)
    n_x = length(d2L_dxdp_vec)
    @views primal[1:n_x] .= .-d2L_dxdp_vec
end

_jvp_set_dual_lhs!(dual, ::Nothing) = nothing
_jvp_set_dual_lhs!(dual, dg_dp_vec) = (dual .= .-dg_dp_vec)

function _jvp_set_dual_rhs!(dual, dlcon_dp, ::Nothing, is_eq)
    dual .+= is_eq .* dlcon_dp
end
function _jvp_set_dual_rhs!(dual, ::Nothing, ducon_dp, is_eq)
    dual .+= is_eq .* ducon_dp
end
function _jvp_set_dual_rhs!(dual, dlcon_dp, ducon_dp, is_eq)
    dual .+= is_eq .* (dlcon_dp .+ ducon_dp) ./ 2
end

_jvp_set_dual_bound_var!(dual, scale, ::Nothing, ::Val{:lower}) = nothing
_jvp_set_dual_bound_var!(dual, scale, ::Nothing, ::Val{:upper}) = nothing
_jvp_set_dual_bound_var!(dual, ::Nothing, dp, ::Val{:lower}) = (dual .= dp)
_jvp_set_dual_bound_var!(dual, ::Nothing, dp, ::Val{:upper}) = (dual .= .-dp)
_jvp_set_dual_bound_var!(dual, scale, dp, ::Val{:lower}) = (dual .= scale .* dp)
_jvp_set_dual_bound_var!(dual, scale, dp, ::Val{:upper}) = (dual .= .-scale .* dp)

_jvp_set_dual_bound_con!(dual, scale, ::Nothing, pos, con, ::Val{:lower}) = nothing
_jvp_set_dual_bound_con!(dual, scale, ::Nothing, pos, con, ::Val{:upper}) = nothing
function _jvp_set_dual_bound_con!(dual, ::Nothing, dcon_dp, pos, con, ::Val{:lower})
    @views dual[pos] .+= dcon_dp[con]
    return nothing
end
function _jvp_set_dual_bound_con!(dual, ::Nothing, dcon_dp, pos, con, ::Val{:upper})
    @views dual[pos] .-= dcon_dp[con]
    return nothing
end
function _jvp_set_dual_bound_con!(dual, scale, dcon_dp, pos, con, ::Val{:lower})
    @views dual[pos] .+= scale[pos] .* dcon_dp[con]
    return nothing
end
function _jvp_set_dual_bound_con!(dual, scale, dcon_dp, pos, con, ::Val{:upper})
    @views dual[pos] .-= scale[pos] .* dcon_dp[con]
    return nothing
end

# disambiguation
_jvp_set_dual_rhs!(dual, ::Nothing, ::Nothing, is_eq) = nothing
_jvp_set_dual_bound_var!(dual, ::Nothing, ::Nothing, ::Val{:lower}) = nothing
_jvp_set_dual_bound_var!(dual, ::Nothing, ::Nothing, ::Val{:upper}) = nothing
_jvp_set_dual_bound_con!(dual, ::Nothing, ::Nothing, pos, con, ::Val{:lower}) = nothing
_jvp_set_dual_bound_con!(dual, ::Nothing, ::Nothing, pos, con, ::Val{:upper}) = nothing
