# ============================================================================
# BatchMadDiffSolver — batched analogue of `MadDiffSolver`.
#
# Wraps a solved `UniformBatchMPCSolver`. Exposes the same pair of kernels as
# the scalar solver — `jacobian_vector_product!` (forward) and
# `vector_jacobian_product!` (reverse) — but all inputs and outputs are
# per-instance matrices of shape `(dim, batch_size)`.
# ============================================================================

"""
    BatchMadDiffSolver(batch_solver::UniformBatchMPCSolver)

Construct a batched sensitivity solver around a solved
`UniformBatchMPCSolver`. The constructor refactorises the batch KKT system at
the solution; the resulting object caches that factorisation and lazily
allocates JVP/VJP scratch on first use.
"""
mutable struct BatchMadDiffSolver{T, BS <: UniformBatchMPCSolver{T}, VB, FC, RC}
    solver::BS
    n_p::Int
    is_eq::VB
    jvp_cache::Union{Nothing, FC}
    vjp_cache::Union{Nothing, RC}
end

function MadDiff.reset_sensitivity_cache!(sens::BatchMadDiffSolver)
    sens.jvp_cache = nothing
    sens.vjp_cache = nothing
    return sens
end

MadDiff._solver_proto(sens::BatchMadDiffSolver) = sens.solver.state.workspace.bx
