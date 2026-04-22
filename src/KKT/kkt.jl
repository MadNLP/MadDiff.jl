# ============================================================================
# KKT preparation and refined (back-)solves driven from a MadDiffSolver.
# ============================================================================

function get_sensitivity_kkt(solver, config::MadDiffConfig)
    reuse = !_needs_new_kkt(config)
    kkt   = reuse ? solver.kkt : build_new_kkt(solver; config)
    (reuse && config.skip_kkt_refactorization) || refactorize_kkt!(kkt, solver)
    return kkt
end

function refactorize_kkt!(kkt, solver::MadNLPSolver)
    set_aug_diagonal!(kkt, solver)
    set_aug_rhs!(solver, kkt, solver.c, solver.mu)
    dual_inf_perturbation!(primal(solver.p), solver.ind_llb, solver.ind_uub,
                           solver.mu, solver.opt.kappa_d)

    shim = kkt === solver.kkt ? solver : _SensitivitySolverShim(solver, kkt)
    inertia_correction!(solver.inertia_corrector, shim) ||
        error("MadDiff: failed to factorize KKT for sensitivities (inertia correction).")
    return nothing
end

function build_new_kkt(solver::AbstractMadNLPSolver; config::MadDiffConfig)
    cb            = solver.cb
    kkt_type      = something(config.kkt_system, SparseUnreducedKKTSystem)
    linear_solver = something(config.linear_solver,
                              _get_wrapper_type(solver.kkt.linear_solver))

    opts = config.kkt_options === nothing ? Dict{Symbol, Any}() :
                                            copy(config.kkt_options)
    config.linear_solver_options === nothing ||
        (opts[:opt_linear_solver] = config.linear_solver_options)

    kkt = create_kkt_system(kkt_type, cb, linear_solver; opts...)
    initialize!(kkt)
    eval_jac_wrapper!(solver, kkt, solver.x)
    eval_lag_hess_wrapper!(solver, kkt, solver.x, solver.y)
    return kkt
end

# ---------- refined solves ----------
#
# Forward JVP runs Richardson IR via MadNLP's `solve_refine!` against the
# forward `mul!`, over the solver's `RichardsonIterator`. Reverse VJP runs
# IR via `adjoint_solve_refine!` against `adjoint_mul!` over an
# `AdjointRichardsonIterator` wrapping this solver's `AdjointKKT` — the
# latter carries the `hess_diag` scratch that makes `adjoint_mul!` correct
# on GPU (see `AdjointKKT` docstring in `adjoint.jl`). Both share the same
# try-improve-retry pattern against the solver's factor.

function _kkt_solve_with_refine!(
    sens::MadDiffSolver, w::AbstractKKTVector, cache, refine!::Function,
)
    solver  = sens.solver
    d, work = cache.kkt_sol, cache.kkt_work

    iterator = if refine! === adjoint_solve_refine!
        AdjointRichardsonIterator(sens.adjoint_kkt;
            opt    = solver.iterator.opt,
            logger = solver.iterator.logger,
            cnt    = solver.cnt,
        )
    elseif sens.kkt === solver.kkt
        solver.iterator
    else
        RichardsonIterator(sens.kkt;
            opt    = solver.iterator.opt,
            logger = solver.iterator.logger,
            cnt    = solver.cnt,
        )
    end

    copyto!(full(d), full(w))
    solver.cnt.linear_solver_time += @elapsed begin
        refine!(d, iterator, w, work) ||
            (improve!(sens.kkt.linear_solver) && refine!(d, iterator, w, work))
    end
    copyto!(full(w), full(d))
    return nothing
end
