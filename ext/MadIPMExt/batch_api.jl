"""
    BatchMadDiffSolver(batch_solver::AbstractBatchMPCSolver)

Create a batch sensitivity solver from a solved `UniformBatchMPCSolver`.
Supports `jacobian_vector_product!` and `vector_jacobian_product!` for
computing sensitivities across all batch instances simultaneously.
"""
mutable struct BatchMadDiffSolver{
    T,
    BatchSolver <: AbstractBatchMPCSolver{T},
    FC, RC,
}
    solver::BatchSolver
    n_p::Int
    is_eq::Vector{Bool}
    jvp_cache::Union{Nothing, FC}
    vjp_cache::Union{Nothing, RC}
end

function MadDiff.reset_sensitivity_cache!(sens::BatchMadDiffSolver)
    sens.jvp_cache = nothing
    sens.vjp_cache = nothing
    return sens
end
