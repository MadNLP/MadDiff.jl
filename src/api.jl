"""
    MadDiffConfig(; kwargs...)

Configuration options for MadDiff sensitivity solves.
"""
Base.@kwdef mutable struct MadDiffConfig
    kkt_system::Union{Nothing, Type} = nothing
    kkt_options::Union{Nothing, Dict} = nothing
    linear_solver::Union{Nothing, Type} = nothing
    linear_solver_options::Union{Nothing, Dict} = nothing
    reuse_kkt::Bool = true
    skip_kkt_refactorization::Bool = false
end

function _needs_new_kkt(config)
    return !isnothing(config.kkt_system) ||
        !isnothing(config.kkt_options) ||
        !isnothing(config.linear_solver) ||
        !isnothing(config.linear_solver_options)
end

"""
    MadDiffSolver(solver::AbstractMadNLPSolver; config=MadDiffConfig(), param_pullback=nothing, n_p=0)

Create a sensitivity solver from a `MadNLP` solver.
"""
mutable struct MadDiffSolver{
    T,
    KKT <: AbstractKKTSystem{T},
    Solver <: AbstractMadNLPSolver{T},
    VB, FC, RC, F, E
}
    solver::Solver
    config::MadDiffConfig
    kkt::KKT
    n_p::Int
    is_eq::VB
    forward_cache::Union{Nothing, FC}
    reverse_cache::Union{Nothing, RC}
    param_pullback::F
    ext::E
end

function MadDiffSolver(solver::AbstractMadNLPSolver{T}; config::MadDiffConfig = MadDiffConfig(), param_pullback = nothing, n_p = 0, extension = nothing) where {T}
    assert_solved_and_feasible(solver)

    if !isnothing(param_pullback) && iszero(n_p)
        throw(ArgumentError("n_p must be provided when param_pullback is given"))
    end

    cb = solver.cb
    n_con = NLPModels.get_ncon(solver.nlp)

    x_array = full(solver.x)
    n_con = NLPModels.get_ncon(solver.nlp)
    is_eq = fill!(similar(x_array, Bool, n_con), false)
    is_eq[solver.cb.ind_eq] .= true

    kkt = get_sensitivity_kkt(solver, config)

    KKT = typeof(kkt)
    Solver = typeof(solver)
    VI = typeof(cb.ind_lb)
    VB = typeof(is_eq)
    VT = typeof(x_array)
    VK = UnreducedKKTVector{T,VT,VI}
    PV = PrimalVector{T,VT,VI}
    FC = ForwardCache{VT, VK, PV}
    RC = ReverseCache{VT, VK, PV}
    F = typeof(param_pullback)
    E = typeof(extension)
    return MadDiffSolver{T, KKT, Solver, VB, FC, RC, F, E}(
        solver, config, kkt, n_p, is_eq,
        nothing, nothing,
        param_pullback,
        extension,
    )
end

"""
    reset_sensitivity_cache!(sens::MadDiffSolver)

Clear cached buffers and rebuild the sensitivity KKT factorization.
Call this after changing solver/KKT options or if the underlying model changed
in-place (with the same dimensions).
"""
function reset_sensitivity_cache!(sens::MadDiffSolver)
    sens.forward_cache = nothing
    sens.reverse_cache = nothing
    sens.kkt = get_sensitivity_kkt(sens.solver, sens.config)
    return sens
end

"""
    forward_differentiate!(sens::MadDiffSolver; d2L_dxdp=nothing, dg_dp=nothing,
                           dlvar_dp=nothing, duvar_dp=nothing, dlcon_dp=nothing,
                           ducon_dp=nothing)

Compute forward sensitivities (JVP). Returns a `ForwardResult` with fields
`dx`, `dy`, `dzl`, and `dzu`.
"""
function forward_differentiate!(
    sens::MadDiffSolver;
    d2L_dxdp = nothing,
    dg_dp = nothing,
    dlvar_dp = nothing,
    duvar_dp = nothing,
    dlcon_dp = nothing,
    ducon_dp = nothing,
)
    result = ForwardResult(sens)
    return forward_differentiate!(result, sens; d2L_dxdp, dg_dp, dlvar_dp, duvar_dp, dlcon_dp, ducon_dp)
end

function forward_differentiate!(solver::AbstractMadNLPSolver;
    d2L_dxdp = nothing, dg_dp = nothing, dlvar_dp = nothing, duvar_dp = nothing,
    dlcon_dp = nothing, ducon_dp = nothing, kwargs...
)
    config = MadDiffConfig(; kwargs...)
    sens = MadDiffSolver(solver; config)
    return forward_differentiate!(sens; d2L_dxdp, dg_dp, dlvar_dp, duvar_dp, dlcon_dp, ducon_dp)
end

function make_param_pullback(; d2L_dxdp=nothing, dg_dp=nothing, dlcon_dp=nothing, ducon_dp=nothing, dlvar_dp=nothing, duvar_dp=nothing)
    return function(out, dx, dy, dzl, dzu, sens)
        fill!(out, zero(eltype(out)))
        _pullback_sub!(out, d2L_dxdp, dx)  # FIXME: it needs to compute d2L_dxdp at y * obj_scale...
        dy_scaled = sens.reverse_cache.dy_scaled
        dy_scaled .= dy .* sens.solver.cb.obj_scale[]
        _pullback_sub!(out, dg_dp, dy_scaled)
        dy_scaled .*= sens.reverse_cache.eq_scale
        _pullback_add!(out, dlcon_dp, dy_scaled)
        _pullback_add!(out, ducon_dp, dy_scaled)
        _pullback_sub!(out, dlvar_dp, dzl)
        _pullback_add!(out, duvar_dp, dzu)
        return out
    end
end

"""
    reverse_differentiate!(sens::MadDiffSolver; dL_dx=nothing, dL_dy=nothing,
                           dL_dzl=nothing, dL_dzu=nothing)

Compute reverse sensitivities (VJP). Returns a `ReverseResult` with fields
`dx`, `dy`, `dzl`, `dzu`, if a pullback was provided to MadDiffSolver, `grad_p`.
"""
function reverse_differentiate!(sens::MadDiffSolver; dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing)
    result = ReverseResult(sens)
    return reverse_differentiate!(result, sens; dL_dx, dL_dy, dL_dzl, dL_dzu)
end

function reverse_differentiate!(solver::AbstractMadNLPSolver; param_pullback = nothing, dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing, kwargs...)
    config = MadDiffConfig(; kwargs...)
    sens = MadDiffSolver(solver; config, param_pullback)
    return reverse_differentiate!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu)
end