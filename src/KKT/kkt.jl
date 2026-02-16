function get_sensitivity_kkt(solver, config::MadDiffConfig)
    if !config.reuse_kkt && isnothing(config.kkt_system)
        error("reuse_kkt=false requires kkt_system to be set.")
    end

    reusing_solver_kkt = config.reuse_kkt && !_needs_new_kkt(config)

    kkt = if reusing_solver_kkt
        solver.kkt
    else
        # config.reuse_kkt && @info "Ignoring reuse_kkt=true since custom KKT config was passed. To silence this message, pass reuse_kkt=false."
        build_new_kkt(
            solver;
            kkt_system = config.kkt_system,
            kkt_options = config.kkt_options,
            linear_solver = config.linear_solver,
            linear_solver_options = config.linear_solver_options,
        )
    end
    if !(reusing_solver_kkt && config.skip_kkt_refactorization)
        refactorize_kkt!(kkt, solver)
    end
    return kkt
end

function refactorize_kkt!(kkt, solver::MadNLPSolver)
    set_aug_diagonal!(kkt, solver)
    set_aug_rhs!(solver, kkt, solver.c, solver.mu)
    dual_inf_perturbation!(primal(solver.p), solver.ind_llb, solver.ind_uub, solver.mu, solver.opt.kappa_d)

    _solver = (kkt === solver.kkt) ? solver : _SensitivitySolverShim(solver, kkt)
    inertia_correction!(solver.inertia_corrector, _solver) ||
        error("Failed to factorize KKT for sensitivities with inertia correction.")
    return nothing
end

function build_new_kkt(
        solver::AbstractMadNLPSolver;
        kkt_system = nothing,
        kkt_options = nothing,
        linear_solver = nothing,
        linear_solver_options = nothing,
    )
    cb = solver.cb
    kkt_orig = solver.kkt

    kkt_type = isnothing(kkt_system) ? SparseUnreducedKKTSystem : kkt_system

    linear_solver_type = isnothing(linear_solver) ?
        _get_wrapper_type(kkt_orig.linear_solver) : linear_solver

    opts = isnothing(kkt_options) ? Dict{Symbol, Any}() : copy(kkt_options)
    !isnothing(linear_solver_options) && (opts[:opt_linear_solver] = linear_solver_options)

    kkt_new = create_kkt_system(kkt_type, cb, linear_solver_type; opts...)
    initialize!(kkt_new)

    eval_jac_wrapper!(solver, kkt_new, solver.x)
    eval_lag_hess_wrapper!(solver, kkt_new, solver.x, solver.y)

    return kkt_new
end

function _solve_with_refine!(sens::MadDiffSolver{T}, w::AbstractKKTVector, cache) where {T}
    d = cache.kkt_sol
    work = cache.kkt_work

    copyto!(full(d), full(w))
    solver = sens.solver
    iterator = if sens.kkt === solver.kkt
        solver.iterator
    else
        RichardsonIterator(
            sens.kkt;
            opt=solver.iterator.opt,
            logger=solver.iterator.logger,
            cnt=solver.cnt,
        )
    end
    solver.cnt.linear_solver_time += @elapsed begin
        if solve_refine!(d, iterator, w, work)
            # ok
        elseif improve!(sens.kkt.linear_solver)
            solve_refine!(d, iterator, w, work)
        end
    end
    copyto!(full(w), full(d))
    return nothing
end

function _adjoint_solve_with_refine!(sens::MadDiffSolver{T}, w::AbstractKKTVector, cache) where {T}
    d = cache.kkt_sol
    work = cache.kkt_work

    copyto!(full(d), full(w))
    solver = sens.solver
    iterator = if sens.kkt === solver.kkt
        solver.iterator
    else
        RichardsonIterator(
            sens.kkt;
            opt=solver.iterator.opt,
            logger=solver.iterator.logger,
            cnt=solver.cnt,
        )
    end
    solver.cnt.linear_solver_time += @elapsed begin
        if adjoint_solve_refine!(d, iterator, w, work)
            # ok
        elseif improve!(sens.kkt.linear_solver)
            adjoint_solve_refine!(d, iterator, w, work)
        end
    end
    copyto!(full(w), full(d))
    return nothing
end

function multi_solve_kkt!(kkt::AbstractKKTSystem, W::AbstractMatrix)
    # TODO: sparse input dense output
    rhs = UnreducedKKTVector(kkt)
    n = length(full(rhs))

    for j in axes(W, 2)
        copyto!(full(rhs), 1, W, (j - 1) * n + 1, n)
        solve_kkt!(kkt, rhs)  # NOTE: no IR in multi_solve
        copyto!(W, (j - 1) * n + 1, full(rhs), 1, n)
    end
    return W
end