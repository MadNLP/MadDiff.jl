# TODO: ask for this from madnlp
_pack_primal(v, cb::MadNLP.AbstractCallback) = copy(v)
_pack_primal(v, cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter} = v[cb.fixed_handler.free]
_kkt_to_full_idx(cb::MadNLP.AbstractCallback, kkt_idx) = kkt_idx
_kkt_to_full_idx(cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}, kkt_idx) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter} = cb.fixed_handler.free[kkt_idx]
_zeros_like(x_array, ::Type{T}, n::Int) where {T} = fill!(similar(x_array, T, n), zero(T))
_falses_like(x_array, n::Int) = fill!(similar(x_array, Bool, n), false)
_to_bool_array(x_array, v::BitVector) = convert(Vector{Bool}, v)
_to_bool_array(x_array, v) = v
function _unpack_primal!(v_full, cb::MadNLP.AbstractCallback, v_kkt)
    v_full .= v_kkt
end
function _unpack_primal!(v_full, cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}, v_kkt) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter}
    fill!(v_full, zero(eltype(v_full)))
    v_full[cb.fixed_handler.free] .= v_kkt
end

"""
    MadDiffConfig(; kwargs...)

Configuration options for `MadDiffSolver`.

# Keyword Arguments
- `kkt_system::Type`: KKT system type for sensitivity analysis. Defaults to re-using the solver's KKT system.
- `kkt_options::Dict`: Options passed to KKT system constructor.
- `linear_solver::Type`: Custom linear solver type for KKT system.
- `linear_solver_options::Dict`: Options passed to linear solver constructor.
- `reuse_kkt::Bool`: Whether to reuse the solver's KKT system (default: `true`). Ignored if `kkt_system`, `kkt_options`,
                      `linear_solver`, or `linear_solver_options` are provided.
- `regularization::Symbol`: Regularization strategy (`:solver`, `:none`, or `:inertia`. default: `:solver`).
- `inertia_shift_step::Float64`: Step size for inertia-based regularization (default: `1.0e-6`).
- `inertia_max_corrections::Int`: Maximum number of inertia correction attempts (default: `50`).
- `should_warn_condensed::Bool`: Whether to warn when using condensed KKT systems (default: `false`).
"""
Base.@kwdef mutable struct MadDiffConfig
    kkt_system::Union{Nothing, Type} = nothing
    kkt_options::Union{Nothing, Dict} = nothing
    linear_solver::Union{Nothing, Type} = nothing
    linear_solver_options::Union{Nothing, Dict} = nothing
    reuse_kkt::Bool = true
    regularization::Symbol = :solver
    inertia_shift_step::Float64 = 1.0e-6
    inertia_max_corrections::Int = 50
    should_warn_condensed::Bool = false
end

struct SensitivityDims{VI, VB}
    n_x::Int
    n_con::Int
    n_x_kkt::Int
    n_lb::Int
    n_ub::Int
    n_p::Int
    idx_lb::VI
    idx_ub::VI
    var_idx_lb::VI
    var_idx_ub::VI
    is_eq::VB
    slack_lb_pos::VI
    slack_lb_con::VI
    slack_ub_pos::VI
    slack_ub_con::VI
end


"""
    MadDiffSolver(solver; config=MadDiffConfig(), param_pullback=nothing, n_p=0)

Create a sensitivity solver from a solved MadNLP solver.

# Arguments
- `solver`: A solved `MadNLP.AbstractMadNLPSolver`
- `config`: Optional `MadDiffConfig` controlling KKT reuse and regularization.
            If `reuse_kkt=true` and no custom KKT options are provided, we
            reuse the solver's KKT system; otherwise we build a new one.
- `param_pullback`: Optional callback to compute parameter gradients in reverse mode.
               Signature: `param_pullback(out, adj_x, adj_λ, adj_zl, adj_zu, sens) -> out`.
               The callback should fill `out` with ∂L/∂p for each parameter.
- `n_p`: Number of parameters (required if `param_pullback` is provided).
              Used to pre-allocate the gradient buffer.

# Example
```julia
solver = MadNLPSolver(nlp)

# Forward mode:
sens = MadDiffSolver(solver)
fwd_result = forward_differentiate!(sens; Dxp_L, Dp_g)

# Reverse mode:
sens = MadDiffSolver(solver; param_pullback, n_p)
rev_result = reverse_differentiate!(sens; dL_dx)
```
"""
mutable struct MadDiffSolver{
    KKT <: MadNLP.AbstractKKTSystem,
    Solver <: MadNLP.AbstractMadNLPSolver,
    VI, VB, FC, RC, F,
}
    solver::Solver
    config::MadDiffConfig
    kkt::KKT
    dims::SensitivityDims{VI, VB}
    forward_cache::Union{Nothing, FC}
    reverse_cache::Union{Nothing, RC}
    param_pullback::F
end

const SUCCESSFUL_STATUSES = (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)
function assert_solved_and_feasible(solver::MadNLP.AbstractMadNLPSolver)
    solver.status ∉ SUCCESSFUL_STATUSES &&
        error("Solver did not converge successfully: $(solver.status)")
    return nothing
end

function MadDiffSolver(solver::MadNLP.AbstractMadNLPSolver; config::MadDiffConfig = MadDiffConfig(), param_pullback = nothing, n_p = 0)
    assert_solved_and_feasible(solver)

    if !isnothing(param_pullback) && iszero(n_p)
        throw(ArgumentError("n_p must be provided when param_pullback is given"))
    end

    n_x = NLPModels.get_nvar(solver.nlp)
    n_con = NLPModels.get_ncon(solver.nlp)
    n_x_kkt = solver.cb.nvar
    idx_lb = solver.cb.ind_lb
    idx_ub = solver.cb.ind_ub
    n_lb = length(idx_lb)
    n_ub = length(idx_ub)
    var_idx_lb_kkt = idx_lb[idx_lb .<= n_x_kkt]
    var_idx_ub_kkt = idx_ub[idx_ub .<= n_x_kkt]
    var_idx_lb = _kkt_to_full_idx(solver.cb, var_idx_lb_kkt)
    var_idx_ub = _kkt_to_full_idx(solver.cb, var_idx_ub_kkt)

    lcon = NLPModels.get_lcon(solver.nlp)
    ucon = NLPModels.get_ucon(solver.nlp)
    x_array = MadNLP.full(solver.x)
    is_eq = _to_bool_array(x_array, isfinite.(lcon) .& (lcon .== ucon))

    slack_in_lb = idx_lb .> n_x_kkt
    slack_in_ub = idx_ub .> n_x_kkt
    slack_lb_pos = findall(slack_in_lb)
    slack_ub_pos = findall(slack_in_ub)
    slack_lb_con = idx_lb[slack_lb_pos] .- n_x_kkt
    slack_ub_con = idx_ub[slack_ub_pos] .- n_x_kkt

    dims = SensitivityDims(n_x, n_con, n_x_kkt, n_lb, n_ub, n_p, idx_lb, idx_ub, var_idx_lb, var_idx_ub,
                       is_eq, slack_lb_pos, slack_lb_con, slack_ub_pos, slack_ub_con)
    kkt = _prepare_sensitivity_kkt(solver, config)

    T = eltype(x_array)
    KKT = typeof(kkt)
    Solver = typeof(solver)
    VI = typeof(idx_lb)
    VB = typeof(is_eq)
    VT = typeof(x_array)
    VK = MadNLP.UnreducedKKTVector{T,VT,VI}
    FC = ForwardCache{VT, VK}
    RC = ReverseCache{VT, VK, VB}
    F = typeof(param_pullback)
    return MadDiffSolver{KKT, Solver, VI, VB, FC, RC, F}(
        solver, config, kkt, dims,
        nothing, nothing,
        param_pullback,
    )
end

"""
    reset_sensitivity_cache!(sens::MadDiffSolver) -> MadDiffSolver

Clear cached forward/reverse work buffers and refresh the sensitivity KKT
factorization. Use after re-solving or modifying solver so that
sensitivities reflect the latest solution/problem.

If not called, sensitivities will be incorrect!
"""
function reset_sensitivity_cache!(sens::MadDiffSolver)
    sens.forward_cache = nothing
    sens.reverse_cache = nothing
    sens.kkt = _prepare_sensitivity_kkt(sens.solver, sens.config)
    return sens
end

_pullback_sub!(out, ::Nothing, v) = nothing
function _pullback_sub!(out, M, v)
    @lencheck size(M, 1) v
    out .-= M' * v
end

_pullback_add!(out, ::Nothing, v) = nothing
function _pullback_add!(out, M, v)
    @lencheck size(M, 1) v
    out .+= M' * v
end

"""
    make_param_pullback(; d2L_dxdp=nothing, dg_dp=nothing, dlcon_dp=nothing, ducon_dp=nothing, dl_dp=nothing, du_dp=nothing)

Create a `param_pullback` callback from parameter derivative matrices.

The returned callback computes `grad_p = ∂L/∂p` using:

    grad_p = -d2L_dxdp' * dx - dg_dp' * dλ + dlcon_dp' * dλ - dl_dp' * dzl + du_dp' * dzu

# Arguments
- `d2L_dxdp`: `∂²L/∂x∂p` - cross derivative of Lagrangian w.r.t. x and p (n_x × n_p)
- `dg_dp`: `∂g/∂p` - derivative of constraint function w.r.t. p (n_con × n_p)
- `dlcon_dp`: `∂lcon/∂p` - derivative of constraint lower bounds (n_con × n_p)
- `ducon_dp`: `∂ucon/∂p` - derivative of constraint upper bounds (n_con × n_p)
- `dl_dp`: `∂l/∂p` - derivative of variable lower bounds (n_lb × n_p)
- `du_dp`: `∂u/∂p` - derivative of variable upper bounds (n_ub × n_p)

# Equality Constraint Handling
For equality constraints (`lcon == ucon`), the contributions from `dlcon_dp` and `ducon_dp`
are scaled by 0.5 to handle AD through problem construction naturally giving equal values
for both.

# Example
```julia
# Parameter p affects constraint RHS: g(x) >= p
param_pullback = make_param_pullback(dlcon_dp=dlcon_dp)
sens = MadDiffSolver(solver; param_pullback, n_p)
result = reverse_differentiate!(sens; dL_dx)
```
"""
function make_param_pullback(; d2L_dxdp=nothing, dg_dp=nothing, dlcon_dp=nothing, ducon_dp=nothing, dl_dp=nothing, du_dp=nothing)
    return function(out, dx, dλ, dzl, dzu, sens)
        fill!(out, zero(eltype(out)))
        _pullback_sub!(out, d2L_dxdp, dx)
        _pullback_sub!(out, dg_dp, dλ)
        dλ_scaled = sens.reverse_cache.dλ_scaled_cache
        dλ_scaled .= dλ .* sens.reverse_cache.eq_scale
        _pullback_add!(out, dlcon_dp, dλ_scaled)
        _pullback_add!(out, ducon_dp, dλ_scaled)
        _pullback_sub!(out, dl_dp, dzl)
        _pullback_add!(out, du_dp, dzu)
        return out
    end
end