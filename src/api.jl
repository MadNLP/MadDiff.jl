
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

"""
    reset_sensitivity_cache!(sens::MadDiffSolver)

Clear the differentiation caches. Must be called upon changes to the underlying `MadNLP.AbstractMadNLPSolver`.
"""
function reset_sensitivity_cache!(sens::MadDiffSolver)
    sens.jvp_cache = nothing
    sens.vjp_cache = nothing
    sens.jac_cache = nothing
    sens.jact_cache = nothing
    sens.kkt = get_sensitivity_kkt(sens.solver, sens.config)
    return sens
end

"""
    jacobian_vector_product!(sens::MadDiffSolver, Δp::AbstractVector)

Compute sensitivities of the optimal solution to a parameter perturbation
`Δp` by evaluating the Jacobian–vector product (JVP) of the KKT system
via forward implicit differentiation.

Returns a [`JVPResult`](@ref) with solution sensitivities `dx`, `dy`, `dzl`, `dzu`.
"""
function jacobian_vector_product!(sens::MadDiffSolver, Δp::AbstractVector)
    return jacobian_vector_product!(JVPResult(sens), sens, Δp)
end

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

"""
    jacobian!(sens::MadDiffSolver)

Compute the Jacobian of the optimal solution with respect to the parameters
using forward implicit differentiation.

Returns a [`JacobianResult`](@ref) containing Jacobian blocks.
"""
function jacobian!(sens::MadDiffSolver)
    jacobian!(JacobianResult(sens), sens)
end

"""
    jacobian_transpose!(sens::MadDiffSolver)

Compute the transpose of the Jacobian of the optimal solution with respect to
parameters using reverse implicit differentiation.

Returns a [`JacobianTransposeResult`](@ref) containing Jacobian transpose blocks.
"""
function jacobian_transpose!(sens::MadDiffSolver)
    jacobian_transpose!(JacobianTransposeResult(sens), sens)
end
