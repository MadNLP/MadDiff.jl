# TODO: ask/look for this in madnlp
_pack_x(v, cb::MadNLP.AbstractCallback) = copy(v)
_pack_x(v, cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter} = v[cb.fixed_handler.free]
_kkt_to_full_idx(cb::MadNLP.AbstractCallback, kkt_idx) = kkt_idx
_kkt_to_full_idx(cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}, kkt_idx) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter} = cb.fixed_handler.free[kkt_idx]
_zeros_like(x_array, ::Type{T}, n::Int) where {T} = fill!(similar(x_array, T, n), zero(T))
_falses_like(x_array, n::Int) = fill!(similar(x_array, Bool, n), false)
_to_bool_array(x_array, v::BitVector) = convert(Vector{Bool}, v)
_to_bool_array(x_array, v) = v
function _unpack_sensitivity!(v_full, cb::MadNLP.AbstractCallback, v_kkt)
    v_full .= v_kkt
end
function _unpack_sensitivity!(v_full, cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}, v_kkt) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter}
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
- `warn_condensed_kkt::Bool`: Whether to warn when using condensed KKT systems (default: `false`).
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
    warn_condensed_kkt::Bool = false
end

struct ProblemDims{VI, VB}
    n_x::Int
    n_con::Int
    n_x_kkt::Int
    n_lb::Int
    n_ub::Int
    n_params::Int
    ind_lb::VI
    ind_ub::VI
    var_ind_lb::VI
    var_ind_ub::VI
    is_eq::VB
    slack_lb_pos::VI
    slack_lb_con::VI
    slack_ub_pos::VI
    slack_ub_con::VI
end


"""
    MadDiffSolver(solver; config=MadDiffConfig(), param_pullback=nothing, n_params=0)

Create a sensitivity solver from a solved MadNLP solver.

# Arguments
- `solver`: A solved `MadNLP.AbstractMadNLPSolver`
- `config`: Optional `MadDiffConfig` controlling KKT reuse and regularization.
            If `reuse_kkt=true` and no custom KKT options are provided, we
            reuse the solver's KKT system; otherwise we build a new one.
- `param_pullback`: Optional callback to compute parameter gradients in reverse mode.
               Signature: `param_pullback(out, adj_x, adj_λ, adj_zl, adj_zu, sens) -> out`.
               The callback should fill `out` with ∂L/∂p for each parameter.
- `n_params`: Number of parameters (required if `param_pullback` is provided).
              Used to pre-allocate the gradient buffer.

# Example
```julia
solver = MadNLPSolver(nlp)

# Forward mode:
sens = MadDiffSolver(solver)
fwd_result = forward_differentiate!(sens; Dxp_L, Dp_g)

# Reverse mode:
sens = MadDiffSolver(solver; param_pullback, n_params)
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
    dims::ProblemDims{VI, VB}
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

function MadDiffSolver(solver::MadNLP.AbstractMadNLPSolver; config::MadDiffConfig = MadDiffConfig(), param_pullback = nothing, n_params = 0)
    assert_solved_and_feasible(solver)

    if !isnothing(param_pullback) && iszero(n_params)
        throw(ArgumentError("n_params must be provided when param_pullback is given"))
    end

    n_x = NLPModels.get_nvar(solver.nlp)
    n_con = NLPModels.get_ncon(solver.nlp)
    n_x_kkt = solver.cb.nvar
    ind_lb = solver.cb.ind_lb
    ind_ub = solver.cb.ind_ub
    n_lb = length(ind_lb)
    n_ub = length(ind_ub)
    var_ind_lb_kkt = ind_lb[ind_lb .<= n_x_kkt]
    var_ind_ub_kkt = ind_ub[ind_ub .<= n_x_kkt]
    var_ind_lb = map(i -> _kkt_to_full_idx(solver.cb, i), var_ind_lb_kkt)
    var_ind_ub = map(i -> _kkt_to_full_idx(solver.cb, i), var_ind_ub_kkt)

    lcon = NLPModels.get_lcon(solver.nlp)
    ucon = NLPModels.get_ucon(solver.nlp)
    x_array = MadNLP.full(solver.x)
    is_eq = _to_bool_array(x_array, isfinite.(lcon) .& (lcon .== ucon))

    slack_in_lb = ind_lb .> n_x_kkt
    slack_in_ub = ind_ub .> n_x_kkt
    slack_lb_pos = findall(slack_in_lb)
    slack_ub_pos = findall(slack_in_ub)
    slack_lb_con = ind_lb[slack_lb_pos] .- n_x_kkt
    slack_ub_con = ind_ub[slack_ub_pos] .- n_x_kkt

    dims = ProblemDims(n_x, n_con, n_x_kkt, n_lb, n_ub, n_params, ind_lb, ind_ub, var_ind_lb, var_ind_ub,
                       is_eq, slack_lb_pos, slack_lb_con, slack_ub_pos, slack_ub_con)
    kkt = _prepare_sensitivity_kkt(solver, config)

    T = eltype(x_array)
    KKT = typeof(kkt)
    Solver = typeof(solver)
    VI = typeof(ind_lb)
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

"""
    make_param_pullback(; Dxp_L=nothing, Dp_g=nothing, Dp_lcon=nothing, Dp_ucon=nothing, Dp_l=nothing, Dp_u=nothing)

Create a `param_pullback` callback from parameter derivative matrices.

The returned callback computes `grad_p = ∂L/∂p` using:

    grad_p = -Dxp_L' * adj_x - Dp_g' * adj_λ + Dp_con' * adj_λ - Dp_l' * adj_zl + Dp_u' * adj_zu

# Arguments
- `Dxp_L`: `∂²L/∂x∂p` - cross derivative of Lagrangian w.r.t. x and p (n_x × n_params)
- `Dp_g`: `∂g/∂p` - derivative of constraint function w.r.t. p (n_con × n_params)
- `Dp_lcon`: `∂lcon/∂p` - derivative of constraint lower bounds (n_con × n_params)
- `Dp_ucon`: `∂ucon/∂p` - derivative of constraint upper bounds (n_con × n_params)
- `Dp_l`: `∂l/∂p` - derivative of variable lower bounds (n_lb × n_params)
- `Dp_u`: `∂u/∂p` - derivative of variable upper bounds (n_ub × n_params)

# Equality Constraint Handling
For equality constraints (`lcon == ucon`), the contributions from `Dp_lcon` and `Dp_ucon`
are scaled by 0.5 to handle AD through problem construction naturally giving equal values
for both.

# Example
```julia
# Parameter p affects constraint RHS: g(x) >= p
param_pullback = make_param_pullback(Dp_lcon=Dp_lcon)
sens = MadDiffSolver(solver; param_pullback, n_params)
result = reverse_differentiate!(sens; dL_dx)
```
"""
function make_param_pullback(; Dxp_L=nothing, Dp_g=nothing, Dp_lcon=nothing, Dp_ucon=nothing, Dp_l=nothing, Dp_u=nothing)
    return function(out, adj_x, adj_λ, adj_zl, adj_zu, sens)
        fill!(out, zero(eltype(out)))

        if !isnothing(Dxp_L)
            size(Dxp_L, 1) == length(adj_x) || throw(DimensionMismatch("Dxp_L has $(size(Dxp_L, 1)) rows, expected $(length(adj_x))"))
            out .-= Dxp_L' * adj_x
        end
        if !isnothing(Dp_g)
            size(Dp_g, 1) == length(adj_λ) || throw(DimensionMismatch("Dp_g has $(size(Dp_g, 1)) rows, expected $(length(adj_λ))"))
            out .-= Dp_g' * adj_λ
        end

        adj_λ_scaled = adj_λ .* sens.reverse_cache.eq_scale
        if !isnothing(Dp_lcon)
            size(Dp_lcon, 1) == length(adj_λ) || throw(DimensionMismatch("Dp_lcon has $(size(Dp_lcon, 1)) rows, expected $(length(adj_λ))"))
            out .+= Dp_lcon' * adj_λ_scaled
        end
        if !isnothing(Dp_ucon)
            size(Dp_ucon, 1) == length(adj_λ) || throw(DimensionMismatch("Dp_ucon has $(size(Dp_ucon, 1)) rows, expected $(length(adj_λ))"))
            out .+= Dp_ucon' * adj_λ_scaled
        end

        if !isnothing(Dp_l)
            size(Dp_l, 1) == length(adj_zl) || throw(DimensionMismatch("Dp_l has $(size(Dp_l, 1)) rows, expected $(length(adj_zl))"))
            out .-= Dp_l' * adj_zl
        end
        if !isnothing(Dp_u)
            size(Dp_u, 1) == length(adj_zu) || throw(DimensionMismatch("Dp_u has $(size(Dp_u, 1)) rows, expected $(length(adj_zu))"))
            out .+= Dp_u' * adj_zu
        end
        return out
    end
end