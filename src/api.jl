Base.@kwdef mutable struct MadDiffConfig
    kkt_system::Union{Nothing, Type} = nothing
    kkt_options::Union{Nothing, Dict} = nothing
    linear_solver::Union{Nothing, Type} = nothing
    linear_solver_options::Any = nothing
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
    VB, FC, RC, JC, TC
}
    solver::Solver
    config::MadDiffConfig
    kkt::KKT
    n_p::Int
    is_eq::VB
    jvp_cache::Union{Nothing, FC}
    vjp_cache::Union{Nothing, RC}
    jac_cache::Union{Nothing, JC}
    jact_cache::Union{Nothing, TC}
end

function MadDiffSolver(solver::AbstractMadNLPSolver{T}; config::MadDiffConfig = MadDiffConfig()) where {T}
    assert_solved_and_feasible(solver)

    n_p = solver.nlp.pmeta.nparam

    cb = solver.cb
    n_con = get_ncon(solver.nlp)

    x_array = full(solver.x)
    n_con = get_ncon(solver.nlp)
    is_eq = fill!(similar(x_array, Bool, n_con), false)
    is_eq[solver.cb.ind_eq] .= true

    kkt = get_sensitivity_kkt(solver, config)

    KKT = typeof(kkt)
    Solver = typeof(solver)
    VI = typeof(cb.ind_lb)
    VB = typeof(is_eq)
    VT = typeof(x_array)
    MT = typeof(create_array(cb, T, get_nvar(solver.nlp), n_p))
    WM = typeof(spzeros_like(cb, T, 0, 0))
    VK = UnreducedKKTVector{T,VT,VI}
    PV = PrimalVector{T,VT,VI}
    FC = JVPCache{VT, VK, PV}
    RC = VJPCache{VT, VK, PV}
    JC = JacobianCache{VT, MT, WM}
    TC = JacobianTransposeCache{VT, MT, WM}
    return MadDiffSolver{T, KKT, Solver, VB, FC, RC, JC, TC}(
        solver, config, kkt, n_p, is_eq,
        nothing, nothing, nothing, nothing,
    )
end

function reset_sensitivity_cache!(sens::MadDiffSolver)
    sens.jvp_cache = nothing
    sens.vjp_cache = nothing
    sens.jac_cache = nothing
    sens.jact_cache = nothing
    sens.kkt = get_sensitivity_kkt(sens.solver, sens.config)
    return sens
end

function jacobian_vector_product!(sens::MadDiffSolver, Δp::AbstractVector)
    return jacobian_vector_product!(JVPResult(sens), sens, Δp)
end

function vector_jacobian_product!(
    sens::MadDiffSolver;
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing, dobj = nothing,
)
    return vector_jacobian_product!(VJPResult(sens), sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
end

function jacobian!(sens::MadDiffSolver)
    jacobian!(JacobianResult(sens), sens)
end
function jacobian_transpose!(sens::MadDiffSolver)
    jacobian_transpose!(JacobianTransposeResult(sens), sens)
end
