
"""
    MadDiffConfig

Options struct for MadDiff. Except for `skip_kkt_refactorization`, if any options are provided,
    MadDiff will create its own KKT system rather than re-using the solver's.

## Fields
- `kkt_system::Type`: The `MadNLP.AbstractKKTSystem` to use for implicit differentiation. Example: `MadNLP.SparseUnreducedKKTSystem`
- `kkt_options::Dict`: The kwargs to pass to `MadNLP.create_kkt_system`.
- `linear_solver::Type`: The `MadNLP.AbstractLinearSolver` to use for implicit differentation. Example: `MadNLP.MumpsSolver`
- `linear_solver_options::Any`: The `opts` to pass to the constructor of `linear_solver`.
- `skip_kkt_refactorization::Bool`: If set to `true`, MadDiff will not refactorize the KKT system before differentiation. Default is `false`.
"""
Base.@kwdef mutable struct MadDiffConfig
    kkt_system::Union{Nothing, Type} = nothing
    kkt_options::Union{Nothing, Dict} = nothing
    linear_solver::Union{Nothing, Type} = nothing
    linear_solver_options::Any = nothing
    skip_kkt_refactorization::Bool = false
end


"""
    MadDiffSolver(solver::MadNLP.AbstractMadNLPSolver; config::MadDiffConfig = MadDiffConfig())

Create a `MadDiffSolver` from a solved `MadNLP.AbstractMadNLPSolver`.
"""
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
    jvp_cache::Union{Nothing, FC}
    vjp_cache::Union{Nothing, RC}
end

function MadDiffSolver(solver::AbstractMadNLPSolver{T}; config::MadDiffConfig = MadDiffConfig()) where {T}
    assert_solved_and_feasible(solver)

    n_p = get_nparam(solver.nlp)

    cb = solver.cb

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
    VK = UnreducedKKTVector{T,VT,VI}
    PV = PrimalVector{T,VT,VI}
    FC = JVPCache{VT, VK, PV}
    RC = VJPCache{VT, VK, PV}
    return MadDiffSolver{T, KKT, Solver, VB, FC, RC}(
        solver, config, kkt, n_p, is_eq,
        nothing, nothing,
    )
end

"""
    reset_sensitivity_cache!(sens::MadDiffSolver)

Clear the differentiation caches. Must be called upon changes to the underlying `MadNLP.AbstractMadNLPSolver`.
"""
function reset_sensitivity_cache!(sens::MadDiffSolver)
    sens.jvp_cache = nothing
    sens.vjp_cache = nothing
    sens.kkt = get_sensitivity_kkt(sens.solver, sens.config)
    return sens
end

"""
    jacobian_vector_product!(sens::MadDiffSolver, Δp::AbstractVector)

Compute sensitivities of the optimal solution to a parameter perturbation
`Δp` by evaluating the Jacobian–vector product (JVP) of the KKT system
via forward implicit differentiation.

Returns a [`JVPResult`](@ref) with solution sensitivities `dx`, `dy`, `dzl`, `dzu`.
Use [`compute_objective_sensitivity!`](@ref) to populate `result.dobj`.
"""
function jacobian_vector_product!(sens::MadDiffSolver, Δp::AbstractVector)
    return jacobian_vector_product!(JVPResult(sens), sens, Δp)
end

"""
    compute_objective_sensitivity!(result::JVPResult, sens::MadDiffSolver, Δp::AbstractVector)

Populate `result.dobj` for a previously computed JVP result.
"""
function compute_objective_sensitivity! end

"""
    vector_jacobian_product!(sens::MadDiffSolver; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)

Compute the vector–Jacobian product (VJP) needed to backpropagate a scalar loss
through the optimal solution with respect to the parameters, using reverse
implicit differentiation.

Keyword arguments provide the loss sensitivities with respect to the primal/dual
solution components (`dL_dx`, `dL_dy`, `dL_dzl`, `dL_dzu`). A "shortcut" objective contribution
is also accepted under `dobj`. All are optional, but at least one must be provided.

Returns a [`VJPResult`](@ref) containing the parameter gradient `grad_p`.
"""
function vector_jacobian_product!(
    sens::MadDiffSolver;
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing, dobj = nothing,
)
    return vector_jacobian_product!(VJPResult(sens), sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
end
