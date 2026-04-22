# ============================================================================
# Batch KKT system — adjoint solve + sensitivity RHS / PV helpers.
# ============================================================================

function adjoint_solve_kkt!(
    bkkt::SparseUniformBatchKKTSystem, d::BatchUnreducedKKTVector,
)
    lb_off = d.n + d.m
    # reverse of `_finish_aug_solve_batch!` — we need the forward unwind's
    # inverse here, which uses the same block layout.
    MadIPM._reduce_rhs_batch!(
        d.values, d.ind_lb, lb_off, bkkt.l_diag,
                  d.ind_ub, lb_off + d.nlb, bkkt.u_diag,
    )
    rhs = bkkt.rhs_buffer
    pd  = MadNLP.primal_dual(d)
    copyto!(rhs, pd)
    MadNLP.solve_linear_system!(bkkt, rhs)
    copyto!(pd, rhs)
    MadIPM._finish_aug_solve_batch!(
        d.values, d.ind_lb, lb_off, bkkt.l_lower, bkkt.l_diag,
                  d.ind_ub, lb_off + d.nlb, bkkt.u_lower, bkkt.u_diag,
    )
    return d
end

# Bound-dual RHS for the forward JVP. `SparseUniformBatchKKTSystem` is an
# unreduced system in the batch sense — carry dlvar_dp / duvar_dp through
# directly (no `l_lower` / `u_lower` scaling).
function MadDiff.jvp_bound_rhs!(
    ::SparseUniformBatchKKTSystem, w::BatchUnreducedKKTVector,
    dlvar_dp::BatchPrimalVector, duvar_dp::BatchPrimalVector,
)
    MadNLP.dual_lb(w) .=  lower(dlvar_dp)
    MadNLP.dual_ub(w) .= .-upper(duvar_dp)
    return nothing
end

function MadDiff.vjp_fill_bound_pv!(
    ::SparseUniformBatchKKTSystem,
    pvl::BatchPrimalVector, pvu::BatchPrimalVector, w::BatchUnreducedKKTVector,
)
    fill!(MadNLP.full(pvl), zero(eltype(MadNLP.full(pvl))))
    fill!(MadNLP.full(pvu), zero(eltype(MadNLP.full(pvu))))
    lower(pvl) .=  MadNLP.dual_lb(w)
    upper(pvu) .= .-MadNLP.dual_ub(w)
    return nothing
end

# Override MadDiff's refined solves for the batch path: MadIPM's batch KKT
# system already solves without refinement.
function MadDiff._kkt_solve_with_refine!(
    sens::BatchMadDiffSolver, w::BatchUnreducedKKTVector, _cache, refine!,
)
    refine! === MadNLP.solve_refine! ?
        MadNLP.solve_kkt!(sens.solver.problem.kkt, w) :
        adjoint_solve_kkt!(sens.solver.problem.kkt, w)
    return nothing
end
