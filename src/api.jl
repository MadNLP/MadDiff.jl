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

mutable struct MadDiffSolver{
    T,
    KKT <: AbstractKKTSystem{T},
    Solver <: AbstractMadNLPSolver{T},
    VB, FC, RC
}
    solver::Solver
    config::MadDiffConfig
    kkt::KKT
    n_p::Int
    is_eq::VB
    forward_cache::Union{Nothing, FC}
    reverse_cache::Union{Nothing, RC}
end

function MadDiffSolver(solver::AbstractMadNLPSolver{T}; config::MadDiffConfig = MadDiffConfig()) where {T}
    assert_solved_and_feasible(solver)

    n_p = solver.nlp.pmeta.nparam

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
    return MadDiffSolver{T, KKT, Solver, VB, FC, RC}(
        solver, config, kkt, n_p, is_eq,
        nothing, nothing,
    )
end

function reset_sensitivity_cache!(sens::MadDiffSolver)
    sens.forward_cache = nothing
    sens.reverse_cache = nothing
    sens.kkt = get_sensitivity_kkt(sens.solver, sens.config)
    return sens
end

function forward_differentiate!(sens::MadDiffSolver, Δp::AbstractVector)
    return forward_differentiate!(ForwardResult(sens), sens, Δp)
end

function reverse_differentiate!(
    sens::MadDiffSolver;
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing, dobj = nothing,
)
    return reverse_differentiate!(ReverseResult(sens), sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
end