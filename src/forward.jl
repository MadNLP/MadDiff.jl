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
        _zeros_like(x_array, T, dims.n_x),
        _zeros_like(x_array, T, dims.n_x),
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
- `dl_dp`: Variable lower bound perturbation times Δp: (∂l/∂p) * Δp (length = n_x)
- `du_dp`: Variable upper bound perturbation times Δp: (∂u/∂p) * Δp (length = n_x)
- `dlcon_dp`: Constraint lower bound perturbation times Δp: (∂lcon/∂p) * Δp (length = n_con)
- `ducon_dp`: Constraint upper bound perturbation times Δp: (∂ucon/∂p) * Δp (length = n_con)

# Notes
For equality constraints (lcon[i] == ucon[i]), provide the same perturbation for both
dlcon_dp[i] and ducon_dp[i]. The implementation uses (dlcon_dp + ducon_dp)/2 for equality constraints.
For entries where lcon or ucon is ±Inf, the corresponding dlcon_dp/ducon_dp value is ignored.
For fixed variables (lvar[i] == uvar[i]) with MakeParameter, the sensitivity dx[i] is set to dl_dp[i].

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
    !isnothing(dl_dp) && @lencheck dims.n_x dl_dp
    !isnothing(du_dp) && @lencheck dims.n_x du_dp
    !isnothing(dlcon_dp) && @lencheck dims.n_con dlcon_dp
    !isnothing(ducon_dp) && @lencheck dims.n_con ducon_dp

    shared = _get_shared_cache!(sens)
    cache = _get_forward_cache!(sens)
    cb = sens.solver.cb

    _lag = _copy_and_scale_lag!(shared.dx_kkt, d2L_dxdp, cb, cache)
    _lhs = _copy_and_scale_con!(cache.dg_dp, dg_dp, cb)
    _lcon = _copy_and_scale_con!(cache.dlcon_dp, dlcon_dp, cb)
    _ucon = _copy_and_scale_con!(cache.ducon_dp, ducon_dp, cb)

    _build_bound_pert!(shared.zl_buffer, dl_dp, _lcon, cb)
    _build_bound_pert!(shared.zu_buffer, du_dp, _ucon, cb)

    sol = _solve_jvp!(sens.kkt, shared.rhs, _lag, _lhs, shared.zl_buffer, shared.zu_buffer, _lcon, _ucon, dims)

    _extract_sensitivities!(shared.dx_kkt, shared.dλ, shared.dzl_kkt, shared.dzu_kkt, sol, sens.solver)

    _unpack_primal!(shared.dx_full, cb, shared.dx_kkt)
    _set_fixed_sensitivity!(shared.dx_full, dl_dp, du_dp, dims)
    MadNLP.unpack_y!(shared.dλ, cb, shared.dλ)

    copyto!(result.dx, shared.dx_full)
    copyto!(result.dλ, shared.dλ)
    _unpack_z!(result.dzl, cb, shared.dzl_kkt, shared.zl_buffer, shared.zl_buffer.values_lr)
    _unpack_z!(result.dzu, cb, shared.dzu_kkt, shared.zu_buffer, shared.zu_buffer.values_ur)

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
- `dl_dp`: Variable lower bound perturbation times Δp: (∂l/∂p) * Δp (length = n_x)
- `du_dp`: Variable upper bound perturbation times Δp: (∂u/∂p) * Δp (length = n_x)
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

struct ForwardCache{VT}
    d2L_dxdp::VT
    dg_dp::VT
    dlcon_dp::VT
    ducon_dp::VT
end

function _get_forward_cache!(sens::MadDiffSolver)
    if isnothing(sens.forward_cache)
        dims = sens.dims
        x_array = MadNLP.full(sens.solver.x)
        T = eltype(x_array)
        sens.forward_cache = ForwardCache(
            _zeros_like(x_array, T, dims.n_x),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_con),
        )
    end
    return sens.forward_cache
end

function _solve_jvp!(kkt, w, d2L_dxdp_vec, dg_dp_vec, zl_buffer, zu_buffer, dlcon_dp, ducon_dp, dims)
    _jvp_pre!(kkt, w, d2L_dxdp_vec, dg_dp_vec, zl_buffer, zu_buffer, dlcon_dp, ducon_dp, dims)
    _jvp_solve!(kkt, w)
    _jvp_post!(kkt, w)
    return w
end

function _jvp_pre!(kkt, w, d2L_dxdp_vec, dg_dp_vec, zl_buffer, zu_buffer, dlcon_dp, ducon_dp, dims)
    T = eltype(MadNLP.full(w))
    fill!(MadNLP.full(w), zero(T))

    _jvp_set_primal_lag!(MadNLP.primal(w), d2L_dxdp_vec)
    _jvp_set_dual_lhs!(MadNLP.dual(w), dg_dp_vec)
    _jvp_set_dual_rhs!(MadNLP.dual(w), dlcon_dp, ducon_dp, dims.is_eq)

    l_scale, u_scale = _get_bound_scale(kkt)
    _jvp_set_dual_lb!(MadNLP.dual_lb(w), l_scale, zl_buffer)
    _jvp_set_dual_ub!(MadNLP.dual_ub(w), u_scale, zu_buffer)

    return w
end

_jvp_solve!(kkt, w) = MadNLP.solve!(kkt, w)
_jvp_post!(kkt, w) = nothing

_jvp_set_primal_lag!(primal, ::Nothing) = nothing
function _jvp_set_primal_lag!(primal, d2L_dxdp_vec)
    n_x = length(d2L_dxdp_vec)
    @views primal[1:n_x] .= .-d2L_dxdp_vec
    return nothing
end

_jvp_set_dual_lhs!(dual, ::Nothing) = nothing
_jvp_set_dual_lhs!(dual, dg_dp_vec) = (dual .= .-dg_dp_vec;)

function _jvp_set_dual_rhs!(dual, dlcon_dp, ::Nothing, is_eq)
    dual .+= is_eq .* dlcon_dp
    return nothing
end
function _jvp_set_dual_rhs!(dual, ::Nothing, ducon_dp, is_eq)
    dual .+= is_eq .* ducon_dp
    return nothing
end
function _jvp_set_dual_rhs!(dual, dlcon_dp, ducon_dp, is_eq)
    dual .+= is_eq .* (dlcon_dp .+ ducon_dp) ./ 2
    return nothing
end

_jvp_set_dual_lb!(dual, ::Nothing, pv::MadNLP.PrimalVector) = (dual .= pv.values_lr;)
_jvp_set_dual_lb!(dual, scale, pv::MadNLP.PrimalVector) = (dual .= scale .* pv.values_lr;)
_jvp_set_dual_ub!(dual, ::Nothing, pv::MadNLP.PrimalVector) = (dual .= .-pv.values_ur;)
_jvp_set_dual_ub!(dual, scale, pv::MadNLP.PrimalVector) = (dual .= .-scale .* pv.values_ur;)
_jvp_set_dual_rhs!(dual, ::Nothing, ::Nothing, is_eq) = nothing
