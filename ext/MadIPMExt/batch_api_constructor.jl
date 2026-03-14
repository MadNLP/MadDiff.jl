function BatchMadDiffSolver(batch_solver::AbstractBatchMPCSolver{T}) where {T}
    bcb = batch_solver.bcb
    m = bcb.ncon
    is_eq = Vector{Bool}(undef, m)
    fill!(is_eq, false)
    is_eq[bcb.ind_eq] .= true

    n_p = batch_solver.nlp.meta.nparam
    bs = batch_solver.batch_size
    MT = typeof(batch_solver.workspace.bx)

    FC = BatchJVPCache{MT}
    RC = BatchVJPCache{MT}

    # FIXME: BatchMadDiffConfig
    refactorize_kkt!(batch_solver.kkt, batch_solver)

    return BatchMadDiffSolver{T, typeof(batch_solver), FC, RC}(
        batch_solver, n_p, is_eq, nothing, nothing,
    )
end
