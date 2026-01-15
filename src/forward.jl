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

"""
    forward_differentiate!(sens::MadDiffSolver; kwargs...) -> ForwardResult

Compute forward sensitivities (JVP) for a parameter perturbation.

At least one of the parameter-direction inputs must be provided.
Input vectors may be mutated during computation.

# Keyword Arguments
- `Dxp_L`: Parameter-Lagrangian cross derivative times Δp: (∂²L/∂x∂p) * Δp (length = n_x)
- `Dp_g`: Parameter-constraint LHS Jacobian times Δp: (∂g/∂p) * Δp (length = n_con)
- `Dp_l`: Variable lower bound perturbation times Δp: (∂l/∂p) * Δp (length = n_lb)
- `Dp_u`: Variable upper bound perturbation times Δp: (∂u/∂p) * Δp (length = n_ub)
- `Dp_lcon`: Constraint lower bound perturbation times Δp: (∂lcon/∂p) * Δp (length = n_con)
- `Dp_ucon`: Constraint upper bound perturbation times Δp: (∂ucon/∂p) * Δp (length = n_con)

# Notes
For equality constraints (lcon[i] == ucon[i]), provide the same perturbation for both
Dp_lcon[i] and Dp_ucon[i]. The implementation uses (Dp_lcon + Dp_ucon)/2 for equality constraints.
For entries where lcon or ucon is ±Inf, the corresponding Dp_lcon/Dp_ucon value is ignored.
`Dp_lcon` and `Dp_ucon` must not alias each other (pass copies if needed).
To obtain a full Jacobian w.r.t. parameters, call this once per parameter direction.

# Returns
- `ForwardResult` containing dx, dλ, dzl, dzu sensitivity vectors
"""
function forward_differentiate!(
    sens::MadDiffSolver;
    Dxp_L = nothing,
    Dp_g = nothing,
    Dp_l = nothing,
    Dp_u = nothing,
    Dp_lcon = nothing,
    Dp_ucon = nothing,
)
    all(isnothing, (Dxp_L, Dp_g, Dp_l, Dp_u, Dp_lcon, Dp_ucon)) &&
        throw(ArgumentError("At least one of Dxp_L, Dp_g, Dp_l, Dp_u, Dp_lcon, Dp_ucon must be provided"))

    dims = sens.dims
    !isnothing(Dxp_L) && @lencheck dims.n_x Dxp_L
    !isnothing(Dp_g) && @lencheck dims.n_con Dp_g
    !isnothing(Dp_l) && @lencheck dims.n_lb Dp_l
    !isnothing(Dp_u) && @lencheck dims.n_ub Dp_u
    !isnothing(Dp_lcon) && @lencheck dims.n_con Dp_lcon
    !isnothing(Dp_ucon) && @lencheck dims.n_con Dp_ucon

    cache = _get_forward_cache!(sens)
    cb = sens.solver.cb

    Dxp_L_kkt = _scale_Dxp_L!(Dxp_L, cb)
    _scale_Dp_g!(Dp_g, cb)
    !isnothing(Dp_lcon) && !isnothing(Dp_ucon) && Dp_lcon === Dp_ucon &&
        throw(ArgumentError("Dp_lcon and Dp_ucon must not alias each other"))
    _scale_Dp_con_bound!(Dp_lcon, cb)
    _scale_Dp_con_bound!(Dp_ucon, cb)

    sol = _solve_jvp!(sens.kkt, cache.work, Dxp_L_kkt, Dp_g, Dp_l, Dp_u, Dp_lcon, Dp_ucon, dims)

    dx_kkt = _extract_sensitivities!(cache.adj_λ, cache.dzl, cache.dzu, sol, sens.solver)

    _unpack_sensitivity!(cache.dx_full, cb, dx_kkt)
    MadNLP.unpack_y!(cache.adj_λ, cb, cache.adj_λ)

    return ForwardResult(
        copy(cache.dx_full),
        copy(cache.adj_λ),
        cache.dzl ./ cb.obj_scale[],
        cache.dzu ./ cb.obj_scale[],
    )
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
    Dxp_L = nothing, Dp_g = nothing, Dp_l = nothing, Dp_u = nothing,
    Dp_lcon = nothing, Dp_ucon = nothing, kwargs...
)
    config = MadDiffConfig(; kwargs...)
    sens = MadDiffSolver(solver; config)
    return forward_differentiate!(sens; Dxp_L, Dp_g, Dp_l, Dp_u, Dp_lcon, Dp_ucon)
end

struct ForwardCache{VT, VK}
    work::VK
    dzl::VT
    dzu::VT
    adj_λ::VT
    dx_full::VT
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
        )
    end
    return sens.forward_cache
end


_scale_Dxp_L!(::Nothing, cb) = nothing
function _scale_Dxp_L!(Dxp_L, cb)
    v = _pack_x(Dxp_L, cb)
    v .*= cb.obj_scale[]
    return v
end

_scale_Dp_g!(::Nothing, cb) = nothing
_scale_Dp_g!(Dp_g, cb) = (Dp_g .*= cb.con_scale)

_scale_Dp_con_bound!(::Nothing, cb) = nothing
_scale_Dp_con_bound!(Dp_con_bound, cb) = (Dp_con_bound .*= cb.con_scale)

_get_bound_scale(kkt::MadNLP.AbstractReducedKKTSystem) = (kkt.l_lower, kkt.u_lower)
_get_bound_scale(kkt::MadNLP.AbstractCondensedKKTSystem) = (kkt.l_lower, kkt.u_lower)
_get_bound_scale(::MadNLP.SparseUnreducedKKTSystem) = (nothing, nothing)

function _solve_jvp!(kkt, w, Dxp_L_vec, Dp_g_vec, Dp_l, Dp_u, Dp_lcon, Dp_ucon, dims)
    _jvp_pre!(kkt, w, Dxp_L_vec, Dp_g_vec, Dp_l, Dp_u, Dp_lcon, Dp_ucon, dims)
    _jvp_solve!(kkt, w)
    _jvp_post!(kkt, w)
    return w
end

function _jvp_pre!(kkt, w, Dxp_L_vec, Dp_g_vec, Dp_l, Dp_u, Dp_lcon, Dp_ucon, dims)
    T = eltype(MadNLP.full(w))
    fill!(MadNLP.full(w), zero(T))

    _fwd_primal_lag!(MadNLP.primal(w), Dxp_L_vec)
    _fwd_dual_lhs!(MadNLP.dual(w), Dp_g_vec)
    _fwd_dual_rhs!(MadNLP.dual(w), Dp_lcon, Dp_ucon, dims.is_eq)

    l_scale, u_scale = _get_bound_scale(kkt)
    _fwd_dual_bound_var!(MadNLP.dual_lb(w), l_scale, Dp_l, Val(:lower))
    _fwd_dual_bound_var!(MadNLP.dual_ub(w), u_scale, Dp_u, Val(:upper))
    _fwd_dual_bound_con!(MadNLP.dual_lb(w), l_scale, Dp_lcon, dims.slack_lb_pos, dims.slack_lb_con, Val(:lower))
    _fwd_dual_bound_con!(MadNLP.dual_ub(w), u_scale, Dp_ucon, dims.slack_ub_pos, dims.slack_ub_con, Val(:upper))

    return w
end

_jvp_solve!(kkt, w) = MadNLP.solve!(kkt, w)
_jvp_post!(kkt, w) = nothing

_fwd_primal_lag!(primal, ::Nothing) = nothing
function _fwd_primal_lag!(primal, Dxp_L_vec)
    n_x = length(Dxp_L_vec)
    @views primal[1:n_x] .= .-Dxp_L_vec
end

_fwd_dual_lhs!(dual, ::Nothing) = nothing
_fwd_dual_lhs!(dual, Dp_g_vec) = (dual .= .-Dp_g_vec)

function _fwd_dual_rhs!(dual, Dp_lcon, ::Nothing, is_eq)
    dual .+= is_eq .* Dp_lcon
end
function _fwd_dual_rhs!(dual, ::Nothing, Dp_ucon, is_eq)
    dual .+= is_eq .* Dp_ucon
end
function _fwd_dual_rhs!(dual, Dp_lcon, Dp_ucon, is_eq)
    dual .+= is_eq .* (Dp_lcon .+ Dp_ucon) ./ 2
end

_fwd_dual_bound_var!(dual, scale, ::Nothing, ::Val{:lower}) = nothing
_fwd_dual_bound_var!(dual, scale, ::Nothing, ::Val{:upper}) = nothing
_fwd_dual_bound_var!(dual, ::Nothing, Dp, ::Val{:lower}) = (dual .= Dp)
_fwd_dual_bound_var!(dual, ::Nothing, Dp, ::Val{:upper}) = (dual .= .-Dp)
_fwd_dual_bound_var!(dual, scale, Dp, ::Val{:lower}) = (dual .= scale .* Dp)
_fwd_dual_bound_var!(dual, scale, Dp, ::Val{:upper}) = (dual .= .-scale .* Dp)

_fwd_dual_bound_con!(dual, scale, ::Nothing, pos, con, ::Val{:lower}) = nothing
_fwd_dual_bound_con!(dual, scale, ::Nothing, pos, con, ::Val{:upper}) = nothing
function _fwd_dual_bound_con!(dual, ::Nothing, Dp_con, pos, con, ::Val{:lower})
    @views dual[pos] .+= Dp_con[con]
    return nothing
end
function _fwd_dual_bound_con!(dual, ::Nothing, Dp_con, pos, con, ::Val{:upper})
    @views dual[pos] .-= Dp_con[con]
    return nothing
end
function _fwd_dual_bound_con!(dual, scale, Dp_con, pos, con, ::Val{:lower})
    @views dual[pos] .+= scale[pos] .* Dp_con[con]
    return nothing
end
function _fwd_dual_bound_con!(dual, scale, Dp_con, pos, con, ::Val{:upper})
    @views dual[pos] .-= scale[pos] .* Dp_con[con]
    return nothing
end

# disambiguation
_fwd_dual_rhs!(dual, ::Nothing, ::Nothing, is_eq) = nothing
_fwd_dual_bound_var!(dual, ::Nothing, ::Nothing, ::Val{:lower}) = nothing
_fwd_dual_bound_var!(dual, ::Nothing, ::Nothing, ::Val{:upper}) = nothing
_fwd_dual_bound_con!(dual, ::Nothing, ::Nothing, pos, con, ::Val{:lower}) = nothing
_fwd_dual_bound_con!(dual, ::Nothing, ::Nothing, pos, con, ::Val{:upper}) = nothing