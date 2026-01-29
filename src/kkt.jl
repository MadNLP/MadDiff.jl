function _prepare_sensitivity_kkt(solver, config::MadDiffConfig)
    if !config.reuse_kkt && isnothing(config.kkt_system)
        error("reuse_kkt=false requires kkt_system to be set.")
    end

    reusing_solver_kkt = config.reuse_kkt && !_has_custom_config(config)

    kkt = if reusing_solver_kkt
        solver.kkt
    else
        # config.reuse_kkt && @info "Ignoring reuse_kkt=true since custom KKT config was passed. To silence this message, pass reuse_kkt=false."
        _build_kkt_for_sensitivity(
            solver;
            kkt_system = config.kkt_system,
            kkt_options = config.kkt_options,
            linear_solver = config.linear_solver,
            linear_solver_options = config.linear_solver_options,
        )
    end
    if !(reusing_solver_kkt && config.skip_kkt_refactorization)
        _refactorize_kkt_for_sensitivity!(kkt, solver)
    end
    return kkt
end

function _refactorize_kkt_for_sensitivity!(kkt, solver::MadNLPSolver)
    set_aug_diagonal!(kkt, solver)
    set_aug_rhs!(solver, kkt, solver.c, solver.mu)
    dual_inf_perturbation!(primal(solver.p), solver.ind_llb, solver.ind_uub, solver.mu, solver.opt.kappa_d)

    _solver = (kkt === solver.kkt) ? solver : _SensitivitySolverShim(solver, kkt)
    inertia_correction!(solver.inertia_corrector, _solver) ||
        error("Failed to factorize KKT for sensitivities with inertia correction.")
    return nothing
end

function _build_kkt_for_sensitivity(
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
