function _prepare_sensitivity_kkt(solver, config::MadDiffConfig)
    if !config.reuse_kkt && isnothing(config.kkt_system)
        error("reuse_kkt=false requires kkt_system to be set.")
    end

    reusing_solver_kkt = config.reuse_kkt && !_has_custom_config(config)

    if !reusing_solver_kkt && config.regularization === :none
        error("regularization=:none requires reusing the solver's KKT system. " *
              "Either set reuse_kkt=true without custom KKT options, or use regularization=:solver or :inertia.")
    end

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
    _warn_condensed(kkt, config)
    _refactorize_kkt_for_sensitivity!(
        kkt,
        solver,
        config.regularization;
        inertia_shift_step = config.inertia_shift_step,
        inertia_max_corrections = config.inertia_max_corrections,
    )
    return kkt
end

function _warn_condensed(kkt, config::MadDiffConfig)
    typeof(kkt) <: MadNLP.AbstractCondensedKKTSystem && config.should_warn_condensed &&
    @warn "Using condensed KKT may result in poor precision, particularly for dual sensitivities. " *
            "Consider using SparseKKTSystem or SparseUnreducedKKTSystem."
end

function _factorize_kkt!(kkt, solver; initial_reg_w=0.0, initial_reg_c=0.0, step=1e-6, max_corrections=0)
    MadNLP.set_aug_diagonal!(kkt, solver)
    if !iszero(initial_reg_w) || !iszero(initial_reg_c)
        MadNLP.regularize_diagonal!(kkt, initial_reg_w, initial_reg_c)
    end

    for i in 0:max_corrections
        MadNLP.build_kkt!(kkt)
        factorized = true
        try
            MadNLP.factorize!(kkt.linear_solver)
            factorized = _factorization_ok(kkt.linear_solver)
        catch err
            if err isa MadNLP.LinearSolverException
                factorized = false
            else
                rethrow()
            end
        end
        factorized && return nothing
        i == 0 && @warn "Inertia correction needed for sensitivity factorization"
        i == max_corrections && break
        MadNLP.regularize_diagonal!(kkt, step, step)
    end
    error("Failed to factorize KKT after $(max_corrections) inertia correction iterations.")
end

function _refactorize_kkt_for_sensitivity!(
    kkt,
    solver,
    regularization;
    inertia_shift_step = 1.0e-6,
    inertia_max_corrections = 50,
)
    regularization === :none && return nothing

    if regularization === :solver
        _factorize_kkt!(kkt, solver; initial_reg_w=solver.del_w, initial_reg_c=solver.del_c)
    elseif regularization === :inertia
        _factorize_kkt!(kkt, solver; step=inertia_shift_step, max_corrections=inertia_max_corrections)
    else
        error("Unknown regularization option: $(regularization). Use :solver, :none, or :inertia.")
    end
    return nothing
end

function _build_kkt_for_sensitivity(
        solver::MadNLP.AbstractMadNLPSolver;
        kkt_system = nothing,
        kkt_options = nothing,
        linear_solver = nothing,
        linear_solver_options = nothing,
    )
    cb = solver.cb
    kkt_orig = solver.kkt

    kkt_type = isnothing(kkt_system) ? MadNLP.SparseUnreducedKKTSystem : kkt_system

    linear_solver_type = isnothing(linear_solver) ?
        _get_wrapper_type(kkt_orig.linear_solver) : linear_solver

    opts = isnothing(kkt_options) ? Dict{Symbol, Any}() : copy(kkt_options)
    !isnothing(linear_solver_options) && (opts[:opt_linear_solver] = linear_solver_options)

    kkt_new = MadNLP.create_kkt_system(kkt_type, cb, linear_solver_type; opts...)

    MadNLP.eval_jac_wrapper!(solver, kkt_new, solver.x)
    MadNLP.eval_lag_hess_wrapper!(solver, kkt_new, solver.x, solver.y)

    return kkt_new
end

function _extract_sensitivities!(dx_kkt, dλ, dzl, dzu, x, solver)
    n_x = solver.cb.nvar
    dx_kkt .= @view MadNLP.primal(x)[1:n_x]
    dλ .= MadNLP.dual(x)
    dzl .= MadNLP.dual_lb(x)
    dzu .= MadNLP.dual_ub(x)
    return nothing
end
