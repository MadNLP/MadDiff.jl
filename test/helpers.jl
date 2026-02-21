function get_constraint_refs(model)
    cons = ConstraintRef[]
    for (F, S) in list_of_constraint_types(model)
        F == VariableRef && continue  # skip variable bounds and parameters
        append!(cons, all_constraints(model, F, S))
    end
    return cons
end

function get_bound_constraint_refs(model)
    lb_cons = ConstraintRef[]
    ub_cons = ConstraintRef[]
    for (F, S) in list_of_constraint_types(model)
        F != VariableRef && continue  # only variable bounds
        S <: MOI.Parameter && continue  # skip parameters
        if S <: MOI.GreaterThan
            append!(lb_cons, all_constraints(model, F, S))
        elseif S <: MOI.LessThan
            append!(ub_cons, all_constraints(model, F, S))
        end
    end
    return lb_cons, ub_cons
end

function _maddiff_config(maddiff_opts)
    config = MadDiff.MadDiffConfig()
    for (name, value) in pairs(maddiff_opts)
        if name == :kkt_system
            config.kkt_system = value
        elseif name == :kkt_options
            config.kkt_options = value
        elseif name == :linear_solver
            config.linear_solver = value
        elseif name == :linear_solver_options
            config.linear_solver_options = value
        elseif name == :skip_kkt_refactorization
            config.skip_kkt_refactorization = value
        else
            error("Unknown MadDiff test option: $(name)")
        end
    end
    return config
end

function _maddiff_model(optimizer, madnlp_opts, maddiff_opts)
    model = MadDiff.diff_model(
        _wrap_optimizer(optimizer, madnlp_opts);
        config = _maddiff_config(maddiff_opts),
    )
    return model
end

function _run(build_model; diffopt = false, param_idx = 1, dp = 1.0, optimizer = MadNLP.Optimizer, madnlp_opts = (;), maddiff_opts = (;))
    model = if diffopt
        m = Model(() -> DiffOpt.diff_optimizer(_wrap_optimizer(optimizer, madnlp_opts)))
        MOI.set(m, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
        MOI.set(m, DiffOpt.AllowObjectiveAndSolutionInput(), true)
        m
    else
        _maddiff_model(optimizer, madnlp_opts, maddiff_opts)
    end
    if optimizer === MadNLP.Optimizer && get(madnlp_opts, :linear_solver, nothing) === CUDSSSolver
        set_optimizer_attribute(model, "array_type", CuArray)
    end
    set_silent(model)
    vars, params = build_model(model)
    optimize!(model)

    cons = get_constraint_refs(model)
    lb_cons, ub_cons = get_bound_constraint_refs(model)

    param = get_param(params, param_idx)
    DiffOpt.empty_input_sensitivities!(model)
    MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(param), MOI.Parameter(dp))
    DiffOpt.forward_differentiate!(model)

    dx = [MOI.get(model, DiffOpt.ForwardVariablePrimal(), v) for v in vars]
    dy = [MOI.get(model, DiffOpt.ForwardConstraintDual(), c) for c in cons]
    dzl = [MOI.get(model, DiffOpt.ForwardConstraintDual(), c) for c in lb_cons]
    dzu = [MOI.get(model, DiffOpt.ForwardConstraintDual(), c) for c in ub_cons]
    dobj = MOI.get(model, DiffOpt.ForwardObjectiveSensitivity())
    return dx, dy, dzl, dzu, dobj
end

function _wrap_optimizer(optimizer, mad_opts)
    return (;kwargs...) -> optimizer(; kwargs..., mad_opts...)
end

function run_maddiff(build_model; param_idx = 1, dp = 1.0, madnlp_opts = (;), maddiff_opts = (;))
    return _run(build_model; diffopt = false, param_idx, dp, optimizer = MadNLP.Optimizer, madnlp_opts, maddiff_opts)
end
function run_diffopt(build_model; param_idx = 1, dp = 1.0, optimizer = MadNLP.Optimizer, madnlp_opts = (;))
    return _run(build_model; diffopt = true, param_idx, dp, optimizer, madnlp_opts)
end

function _set_reverse_seeds!(model, vars, cons, lb_cons, ub_cons; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
    if !isnothing(dL_dx)
        for (i, v) in enumerate(vars)
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), v, dL_dx[i])
        end
    end
    if !isnothing(dL_dy)
        for (i, c) in enumerate(cons)
            MOI.set(model, DiffOpt.ReverseConstraintDual(), c, dL_dy[i])
        end
    end
    if !isnothing(dL_dzl)
        for (i, c) in enumerate(lb_cons)
            MOI.set(model, DiffOpt.ReverseConstraintDual(), c, dL_dzl[i])
        end
    end
    if !isnothing(dL_dzu)
        for (i, c) in enumerate(ub_cons)
            MOI.set(model, DiffOpt.ReverseConstraintDual(), c, dL_dzu[i])
        end
    end
    if !isnothing(dobj)
        MOI.set(model, DiffOpt.ReverseObjectiveSensitivity(), dobj)
    end
    return nothing
end

function _run_reverse(build_model; diffopt = false, dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing, dobj = nothing, optimizer = MadNLP.Optimizer, madnlp_opts = (;), maddiff_opts = (;))
    model = if !diffopt
        _maddiff_model(optimizer, madnlp_opts, maddiff_opts)
    else
        m = Model(() -> DiffOpt.diff_optimizer(_wrap_optimizer(optimizer, madnlp_opts)))
        MOI.set(m, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
        MOI.set(m, DiffOpt.AllowObjectiveAndSolutionInput(), true)
        m
    end
    if optimizer === MadNLP.Optimizer && get(madnlp_opts, :linear_solver, nothing) === CUDSSSolver
        set_optimizer_attribute(model, "array_type", CuArray)
    end
    set_silent(model)
    vars, params = build_model(model)
    optimize!(model)

    cons = get_constraint_refs(model)
    lb_cons, ub_cons = get_bound_constraint_refs(model)

    DiffOpt.empty_input_sensitivities!(model)

    _set_reverse_seeds!(model, vars, cons, lb_cons, ub_cons; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)

    DiffOpt.reverse_differentiate!(model)

    params_list = params isa AbstractArray ? params : [params]
    return [MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)).value for p in params_list]
end

function run_diffopt_reverse(build_model; dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing, dobj = nothing, optimizer = MadNLP.Optimizer, madnlp_opts = (;))
    return _run_reverse(build_model; diffopt = true, dL_dx, dL_dy, dL_dzl, dL_dzu, dobj, optimizer, madnlp_opts)
end
function run_maddiff_reverse(build_model; dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing, dobj = nothing, madnlp_opts = (;), maddiff_opts = (;))
    return _run_reverse(build_model; diffopt = false, dL_dx, dL_dy, dL_dzl, dL_dzu, dobj, optimizer = MadNLP.Optimizer, madnlp_opts, maddiff_opts)
end

function get_param(params, param_idx)
    return params isa AbstractArray ? params[param_idx] : params
end


function _unwrap_optimizer(optimizer)
    if optimizer isa MOI.Utilities.CachingOptimizer
        return _unwrap_optimizer(optimizer.optimizer)
    elseif optimizer isa MOI.Bridges.LazyBridgeOptimizer
        return _unwrap_optimizer(optimizer.model)
    elseif optimizer isa DiffOpt.POI.Optimizer
        return _unwrap_optimizer(optimizer.optimizer)
    end
    return optimizer
end

_solver_from_model(model) = _unwrap_optimizer(unsafe_backend(model).optimizer).solver

function _check_consistency(sens; atol = 1e-8, rtol = 0.0)
    jf = MadDiff.jacobian!(sens)
    jr = MadDiff.jacobian_transpose!(sens)
    @test isapprox(jr.dx, transpose(jf.dx); atol, rtol)
    @test isapprox(jr.dy, transpose(jf.dy); atol, rtol)
    @test isapprox(jr.dzl, transpose(jf.dzl); atol, rtol)
    @test isapprox(jr.dzu, transpose(jf.dzu); atol, rtol)
    @test isapprox(jr.dobj, jf.dobj; atol, rtol)

    n_x   = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(sens.solver.nlp)
    for j in 1:sens.n_p
        dp = zeros(Float64, sens.n_p); dp[j] = 1.0
        col = MadDiff.jacobian_vector_product!(sens, dp)

        for i in 1:n_x
            dL_dx = zeros(Float64, n_x); dL_dx[i] = 1.0
            @test isapprox(col.dx[i], MadDiff.vector_jacobian_product!(sens; dL_dx).grad_p[j]; atol, rtol)
        end
        for i in 1:n_con
            dL_dy = zeros(Float64, n_con); dL_dy[i] = 1.0
            @test isapprox(col.dy[i], MadDiff.vector_jacobian_product!(sens; dL_dy).grad_p[j]; atol, rtol)
        end
        for i in 1:n_x
            dL_dzl = zeros(Float64, n_x); dL_dzl[i] = 1.0
            @test isapprox(col.dzl[i], MadDiff.vector_jacobian_product!(sens; dL_dzl).grad_p[j]; atol, rtol)
        end
        for i in 1:n_x
            dL_dzu = zeros(Float64, n_x); dL_dzu[i] = 1.0
            @test isapprox(col.dzu[i], MadDiff.vector_jacobian_product!(sens; dL_dzu).grad_p[j]; atol, rtol)
        end
        @test isapprox(col.dobj[], MadDiff.vector_jacobian_product!(sens; dobj = 1.0).grad_p[j]; atol, rtol)
    end
end

function run_maddiff_consistency(build_model; madnlp_opts = (;), maddiff_opts = (;), atol = 1e-8, rtol = 0.0)
    model = _maddiff_model(MadNLP.Optimizer, madnlp_opts, maddiff_opts)
    set_silent(model)
    build_model(model)
    optimize!(model)
    config = _maddiff_config(maddiff_opts)
    _check_consistency(MadDiff.MadDiffSolver(_solver_from_model(model); config); atol, rtol)
end

function get_problem_dims(build_model; optimizer=MadNLP.Optimizer, madnlp_opts=(;), maddiff_opts = (;))
    model = _maddiff_model(optimizer, madnlp_opts, maddiff_opts)
    MOI.set(model, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
    MOI.set(model, DiffOpt.AllowObjectiveAndSolutionInput(), true)
    set_silent(model)
    vars, params = build_model(model)

    cons = get_constraint_refs(model)
    lb_cons, ub_cons = get_bound_constraint_refs(model)

    return length(vars), length(cons), length(lb_cons), length(ub_cons)
end


function _simple_lp()
    c = ones(2)
    Hrows = Int[]
    Hcols = Int[]
    Hvals = Float64[]
    Arows = [1, 1]
    Acols = [1, 2]
    Avals = [1.0; 1.0]
    c0 = 0.0
    lvar = [0.0; 0.0]
    uvar = [Inf; Inf]
    lcon = [1.0]
    ucon = [1.0]
    x0 = ones(2)

    return QuadraticModel(c, Hrows, Hcols, Hvals; Arows, Acols, Avals, lcon, ucon, lvar, uvar, c0, x0, name = "simpleLP")
end

function _build_kkt(KKTSystem, Callback)
    nlp = if KKTSystem == MadIPM.NormalKKTSystem
        _simple_lp()
    elseif KKTSystem == MadNCL.K1sAuglagKKTSystem
        MadNCL.NCLModel(MadNLPTests.HS15Model())
    elseif KKTSystem == MadNCL.K2rAuglagKKTSystem
        MadNCL.NCLModel(MadNLPTests.HS15Model())
    else
        MadNLPTests.HS15Model()
    end
    cb = MadNLP.create_callback(Callback, nlp)

    kkt = MadNLP.create_kkt_system(
        KKTSystem,
        cb,
        KKTSystem <: MadNLP.AbstractDenseKKTSystem ? MadNLP.LapackCPUSolver : MadNLP.MumpsSolver;
    )

    MadNLP.initialize!(kkt)

    x0 = NLPModels.get_x0(cb.nlp)
    y0 = NLPModels.get_y0(cb.nlp)
    jac = MadNLP.get_jacobian(kkt)
    MadNLP._eval_jac_wrapper!(cb, x0, jac)
    MadNLP.compress_jacobian!(kkt)
    hess = MadNLP.get_hessian(kkt)
    MadNLP._eval_lag_hess_wrapper!(cb, x0, y0, hess)
    MadNLP.compress_hessian!(kkt)

    fill!(kkt.l_lower, 1e-3)
    fill!(kkt.u_lower, 1e-3)
    MadNLP._set_aug_diagonal!(kkt)

    MadNLP.build_kkt!(kkt)
    MadNLP.factorize!(kkt.linear_solver)

    return kkt
end

function _run_adjoint_tests(KKTSystem, Callback)
    kkt = _build_kkt(KKTSystem, Callback)
    Random.seed!(42)
    v = MadNLP.UnreducedKKTVector(kkt)
    w = MadNLP.UnreducedKKTVector(kkt)
    randn!(MadNLP.full(v))
    randn!(MadNLP.full(w))

    x = copy(w)
    y = copy(v)
    MadNLP.solve_kkt!(kkt, x)
    MadDiff.adjoint_solve_kkt!(kkt, y)
    @test dot(v, x) ≈ dot(y, w) atol=1e-8

    x = MadNLP.UnreducedKKTVector(kkt)
    y = MadNLP.UnreducedKKTVector(kkt)
    randn!(MadNLP.full(x))
    MadNLP.mul!(w, kkt, x)
    MadDiff.adjoint_mul!(y, kkt, v)
    @test dot(v, w) ≈ dot(y, x) atol=1e-8
    return
end