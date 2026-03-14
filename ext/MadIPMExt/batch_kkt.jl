# converged instances are marked inactive (batch_map=0)..
# FIXME: filter by termination status
function _reset_batch_map!(bkkt::SparseUniformBatchKKTSystem)
    bs = bkkt.batch_size
    for i in 1:bs
        bkkt.batch_map[i] = i
        bkkt.batch_map_rev[i] = i
    end
    bkkt.active_batch_size[] = bs
    return
end

function refactorize_kkt!(kkt::SparseUniformBatchKKTSystem, solver::AbstractBatchMPCSolver)
    _reset_batch_map!(kkt)
    MadIPM.set_aug_diagonal_reg!(kkt, solver)
    MadNLP.eval_jac_wrapper!(solver, kkt)
    MadNLP.eval_lag_hess_wrapper!(solver, kkt)
    MadNLP.build_kkt!(kkt)
    MadNLP.factorize_kkt!(kkt)
    return nothing
end

function MadNLP.solve_kkt!(bkkt::SparseUniformBatchKKTSystem, d::BatchUnreducedKKTVector)
    MadNLP.reduce_rhs!(bkkt, d)
    rhs = bkkt.rhs_buffer
    pd = MadNLP.primal_dual(d)
    copyto!(rhs, pd)
    MadNLP.solve_linear_system!(bkkt, rhs)
    copyto!(pd, rhs)
    MadNLP.finish_aug_solve!(bkkt, d)
    return d
end

function adjoint_solve_kkt!(bkkt::SparseUniformBatchKKTSystem, d::BatchUnreducedKKTVector)
    lb_off = d.n + d.m
    MadIPM._adjoint_finish_bounds_batch!(
        d.values, d.ind_lb, lb_off, bkkt.l_lower, bkkt.l_diag,
                  d.ind_ub, lb_off + d.nlb, bkkt.u_lower, bkkt.u_diag,
    )
    rhs = bkkt.rhs_buffer
    pd = MadNLP.primal_dual(d)
    copyto!(rhs, pd)
    MadNLP.solve_linear_system!(bkkt, rhs)
    copyto!(pd, rhs)
    MadIPM._adjoint_reduce_rhs_batch!(
        d.values, d.ind_lb, lb_off, bkkt.l_diag,
                  d.ind_ub, lb_off + d.nlb, bkkt.u_diag,
    )
    return d
end

function MadDiff.jvp_set_bound_rhs!(::SparseUniformBatchKKTSystem, w::BatchUnreducedKKTVector, dlvar_dp::BatchPrimalVector, duvar_dp::BatchPrimalVector)
    MadNLP.dual_lb(w) .= lower(dlvar_dp)
    MadNLP.dual_ub(w) .= .-upper(duvar_dp)
    return nothing
end

function MadDiff.vjp_fill_pv!(::SparseUniformBatchKKTSystem, pvl::BatchPrimalVector, pvu::BatchPrimalVector, w::BatchUnreducedKKTVector)
    fill!(MadNLP.full(pvl), zero(eltype(MadNLP.full(pvl))))
    fill!(MadNLP.full(pvu), zero(eltype(MadNLP.full(pvu))))
    lower(pvl) .= MadNLP.dual_lb(w)
    upper(pvu) .= .-MadNLP.dual_ub(w)
    return nothing
end

function _solve_with_refine!(sens::BatchMadDiffSolver{T}, w, cache) where {T}
    MadNLP.solve_kkt!(sens.solver.kkt, w)
    return nothing
end

function _adjoint_solve_with_refine!(sens::BatchMadDiffSolver{T}, w, cache) where {T}
    adjoint_solve_kkt!(sens.solver.kkt, w)
    return nothing
end
