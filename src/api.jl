Base.@kwdef mutable struct MadDiffConfig
    kkt_system::Union{Nothing, Type} = nothing
    kkt_options::Union{Nothing, Dict} = nothing
    linear_solver::Union{Nothing, Type} = nothing
    linear_solver_options::Union{Nothing, Dict} = nothing
    reuse_kkt::Bool = true
    skip_kkt_refactorization::Bool = false
end

struct JacobianForwardCache{VT, MT, WM}
    x_nlp::VT
    y_nlp::VT
    grad_x::VT
    grad_p::VT
    hpv_nlp::MT
    jpv_nlp::MT
    dlvar_nlp::MT
    duvar_nlp::MT
    dlcon_nlp::MT
    ducon_nlp::MT
    d2L_dxdp::MT
    dg_dp::MT
    dlvar_dp::MT
    duvar_dp::MT
    dlcon_dp::MT
    ducon_dp::MT
    dz_work::MT
    W::WM
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
    VB, FC, RC, JC
}
    solver::Solver
    config::MadDiffConfig
    kkt::KKT
    n_p::Int
    is_eq::VB
    forward_cache::Union{Nothing, FC}
    reverse_cache::Union{Nothing, RC}
    jacobian_cache::Union{Nothing, JC}
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
    WM = typeof(spzeros(T, 0, 0))
    VK = UnreducedKKTVector{T,VT,VI}
    PV = PrimalVector{T,VT,VI}
    FC = ForwardCache{VT, VK, PV}
    RC = ReverseCache{VT, VK, PV}
    JC = JacobianForwardCache{VT, MT, WM}
    return MadDiffSolver{T, KKT, Solver, VB, FC, RC, JC}(
        solver, config, kkt, n_p, is_eq,
        nothing, nothing, nothing,
    )
end

function reset_sensitivity_cache!(sens::MadDiffSolver)
    sens.forward_cache = nothing
    sens.reverse_cache = nothing
    sens.jacobian_cache = nothing
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

function forward_jacobian!(sens::MadDiffSolver)
    forward_jacobian!(JacobianResult(sens), sens)
end
function reverse_jacobian_transpose!(sens::MadDiffSolver)
    reverse_jacobian_transpose!(JacobianTransposeResult(sens), sens)
end
