
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
    n_p::Int
    fixed_idx::VI
    is_eq::VB
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
    VI, VB, SC, FC, RC, F,
}
    solver::Solver
    config::MadDiffConfig
    kkt::KKT
    dims::SensitivityDims{VI, VB}
    shared_cache::Union{Nothing, SC}
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

    cb = solver.cb
    n_x = NLPModels.get_nvar(solver.nlp)
    n_con = NLPModels.get_ncon(solver.nlp)

    lcon = NLPModels.get_lcon(solver.nlp)
    ucon = NLPModels.get_ucon(solver.nlp)
    x_array = MadNLP.full(solver.x)
    is_eq = _to_bool_array(x_array, isfinite.(lcon) .& (lcon .== ucon))
    fixed_idx = _get_fixed_idx(cb, cb.ind_lb)

    dims = SensitivityDims(n_x, n_con, n_p, fixed_idx, is_eq)
    kkt = _prepare_sensitivity_kkt(solver, config)

    T = eltype(x_array)
    KKT = typeof(kkt)
    Solver = typeof(solver)
    VI = typeof(cb.ind_lb)
    VB = typeof(is_eq)
    VT = typeof(x_array)
    VK = MadNLP.UnreducedKKTVector{T,VT,VI}
    PV = MadNLP.PrimalVector{T,VT,VI}
    SC = SharedCache{VT, VK, PV}
    FC = ForwardCache{VT}
    RC = ReverseCache{VT}
    F = typeof(param_pullback)
    return MadDiffSolver{KKT, Solver, VI, VB, SC, FC, RC, F}(
        solver, config, kkt, dims,
        nothing, nothing, nothing,
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
    sens.shared_cache = nothing
    sens.forward_cache = nothing
    sens.reverse_cache = nothing
    sens.kkt = _prepare_sensitivity_kkt(sens.solver, sens.config)
    return sens
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
- `dl_dp`: `∂l/∂p` - derivative of variable lower bounds (n_x × n_p)
- `du_dp`: `∂u/∂p` - derivative of variable upper bounds (n_x × n_p)

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
        dλ_scaled = sens.reverse_cache.dλ_scaled
        dλ_scaled .= dλ .* sens.reverse_cache.eq_scale
        _pullback_add!(out, dlcon_dp, dλ_scaled)
        _pullback_add!(out, ducon_dp, dλ_scaled)
        _pullback_sub!(out, dl_dp, dzl)
        _pullback_add!(out, du_dp, dzu)
        return out
    end
end
