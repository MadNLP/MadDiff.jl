# ============================================================================
# Public types and entry points.
# ============================================================================

"""
    MadDiffConfig(; kkt_system = nothing, kkt_options = nothing,
                   linear_solver = nothing, linear_solver_options = nothing,
                   skip_kkt_refactorization = false)

Options for [`MadDiffSolver`](@ref).

If any of `kkt_system`, `kkt_options`, `linear_solver`, or
`linear_solver_options` is non-`nothing`, MadDiff builds its own KKT system
for sensitivity solves rather than re-using the solver's.

`skip_kkt_refactorization = true` reuses the solver's existing factorisation
without refactorising. Only valid when re-using the solver's KKT.
"""
Base.@kwdef mutable struct MadDiffConfig
    kkt_system::Union{Nothing, Type} = nothing
    kkt_options::Union{Nothing, Dict} = nothing
    linear_solver::Union{Nothing, Type} = nothing
    linear_solver_options::Union{Nothing, AbstractDict} = nothing
    skip_kkt_refactorization::Bool = false
end

"""
    MadDiffSolver(solver; config = MadDiffConfig())

Wrap a solved `MadNLP.AbstractMadNLPSolver` for implicit differentiation.
Caches the KKT system (refactored unless `config.skip_kkt_refactorization`)
and lazily allocates JVP/VJP scratch on first use.
"""
mutable struct MadDiffSolver{
    T,
    KKT    <: AbstractKKTSystem{T},
    Solver <: AbstractMadNLPSolver{T},
    AKKT   <: AdjointKKT{T, <:Any, <:AbstractKKTSystem{T}},
    VB, FC, RC,
}
    solver::Solver
    config::MadDiffConfig
    kkt::KKT
    adjoint_kkt::AKKT
    n_p::Int
    is_eq::VB
    jvp_cache::Union{Nothing, FC}
    vjp_cache::Union{Nothing, RC}
end

function MadDiffSolver(
    solver::AbstractMadNLPSolver{T};
    config::MadDiffConfig = MadDiffConfig(),
) where {T}
    assert_solved_and_feasible(solver)

    cb    = solver.cb
    x     = full(solver.x)
    n_con = get_ncon(solver.nlp)
    is_eq = fill!(similar(x, Bool, n_con), false)
    is_eq[cb.ind_eq] .= true

    kkt         = get_sensitivity_kkt(solver, config)
    adjoint_kkt = AdjointKKT(kkt)

    VT = typeof(x)
    VI = typeof(cb.ind_lb)
    VK = UnreducedKKTVector{T, VT, VI}
    PV = PrimalVector{T, VT, VI}
    FC = JVPCache{VT, VK, PV}
    RC = VJPCache{VT, VK, PV}

    return MadDiffSolver{T, typeof(kkt), typeof(solver), typeof(adjoint_kkt),
                         typeof(is_eq), FC, RC}(
        solver, config, kkt, adjoint_kkt,
        get_nparam(solver.nlp), is_eq, nothing, nothing,
    )
end

"""
    reset_sensitivity_cache!(sens::MadDiffSolver)

Drop the JVP/VJP scratch and refresh the sensitivity KKT. Call this after
the underlying `MadNLP` solver has been mutated.
"""
function reset_sensitivity_cache!(sens::MadDiffSolver)
    sens.jvp_cache = nothing
    sens.vjp_cache = nothing
    sens.kkt = get_sensitivity_kkt(sens.solver, sens.config)
    sens.adjoint_kkt = AdjointKKT(sens.kkt)
    return sens
end

"""
    jacobian_vector_product!(sens::MadDiffSolver, Δp) -> JVPResult

Directional sensitivity of the optimal solution along the parameter
perturbation `Δp`. `result.dobj` is not populated; call
[`compute_objective_sensitivity!`](@ref) to fill it in.
"""
jacobian_vector_product!(sens::MadDiffSolver, Δp::AbstractVector) =
    jacobian_vector_product!(JVPResult(sens), sens, Δp)

"""
    compute_objective_sensitivity!(result, sens, Δp)

Populate `result.dobj` for a previously computed JVP result.
"""
function compute_objective_sensitivity! end

"""
    vector_jacobian_product!(sens::MadDiffSolver;
                             dL_dx, dL_dy, dL_dzl, dL_dzu, dobj) -> VJPResult

Backpropagate a scalar-loss cotangent through the optimal solution. At least
one seed must be non-`nothing`; unseeded components skip their adjoint
back-solve. Returns `result.grad_p = ∂L/∂p`.
"""
vector_jacobian_product!(
    sens::MadDiffSolver;
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing,
    dobj = nothing,
) = vector_jacobian_product!(VJPResult(sens), sens;
        dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
