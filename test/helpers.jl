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

function run_maddiff(build_model; param_idx = 1, dp = 1.0, mad_opts...)
    model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer; mad_opts...))
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
    dλ = [MOI.get(model, DiffOpt.ForwardConstraintDual(), c) for c in cons]
    dzl = [MOI.get(model, DiffOpt.ForwardConstraintDual(), c) for c in lb_cons]
    dzu = [MOI.get(model, DiffOpt.ForwardConstraintDual(), c) for c in ub_cons]
    return dx, dλ, dzl, dzu
end

function run_diffopt(build_model; param_idx = 1, dp = 1.0)
    model = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
    MOI.set(model, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
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
    dλ = [MOI.get(model, DiffOpt.ForwardConstraintDual(), c) for c in cons]
    dzl = [MOI.get(model, DiffOpt.ForwardConstraintDual(), c) for c in lb_cons]
    dzu = [MOI.get(model, DiffOpt.ForwardConstraintDual(), c) for c in ub_cons]
    return dx, dλ, dzl, dzu
end

function run_maddiff_reverse(build_model; dL_dx=nothing, dL_dλ=nothing, dL_dzl=nothing, dL_dzu=nothing, mad_opts...)
    use_ones = isnothing(dL_dx) && isnothing(dL_dλ) && isnothing(dL_dzl) && isnothing(dL_dzu)
    model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer; mad_opts...))
    set_silent(model)
    vars, params = build_model(model)
    optimize!(model)

    cons = get_constraint_refs(model)
    lb_cons, ub_cons = get_bound_constraint_refs(model)

    DiffOpt.empty_input_sensitivities!(model)

    if use_ones || !isnothing(dL_dx)
        vals = use_ones ? ones(length(vars)) : dL_dx
        for (i, v) in enumerate(vars)
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), v, vals[i])
        end
    end
    if use_ones || !isnothing(dL_dλ)
        vals = use_ones ? ones(length(cons)) : dL_dλ
        for (i, c) in enumerate(cons)
            MOI.set(model, DiffOpt.ReverseConstraintDual(), c, vals[i])
        end
    end
    if use_ones || !isnothing(dL_dzl)
        vals = use_ones ? ones(length(lb_cons)) : dL_dzl
        for (i, c) in enumerate(lb_cons)
            MOI.set(model, DiffOpt.ReverseConstraintDual(), c, vals[i])
        end
    end
    if use_ones || !isnothing(dL_dzu)
        vals = use_ones ? ones(length(ub_cons)) : dL_dzu
        for (i, c) in enumerate(ub_cons)
            MOI.set(model, DiffOpt.ReverseConstraintDual(), c, vals[i])
        end
    end

    DiffOpt.reverse_differentiate!(model)

    params_list = params isa AbstractArray ? params : [params]
    return [MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)).value for p in params_list]
end

function run_diffopt_reverse(build_model; dL_dx=nothing, dL_dλ=nothing, dL_dzl=nothing, dL_dzu=nothing)
    use_ones = isnothing(dL_dx) && isnothing(dL_dλ) && isnothing(dL_dzl) && isnothing(dL_dzu)

    model = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
    MOI.set(model, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
    set_silent(model)
    vars, params = build_model(model)
    optimize!(model)

    cons = get_constraint_refs(model)
    lb_cons, ub_cons = get_bound_constraint_refs(model)

    DiffOpt.empty_input_sensitivities!(model)

    if use_ones || !isnothing(dL_dx)
        vals = use_ones ? ones(length(vars)) : dL_dx
        for (i, v) in enumerate(vars)
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), v, vals[i])
        end
    end
    if use_ones || !isnothing(dL_dλ)
        vals = use_ones ? ones(length(cons)) : dL_dλ
        for (i, c) in enumerate(cons)
            MOI.set(model, DiffOpt.ReverseConstraintDual(), c, vals[i])
        end
    end
    if use_ones || !isnothing(dL_dzl)
        vals = use_ones ? ones(length(lb_cons)) : dL_dzl
        for (i, c) in enumerate(lb_cons)
            MOI.set(model, DiffOpt.ReverseConstraintDual(), c, vals[i])
        end
    end
    if use_ones || !isnothing(dL_dzu)
        vals = use_ones ? ones(length(ub_cons)) : dL_dzu
        for (i, c) in enumerate(ub_cons)
            MOI.set(model, DiffOpt.ReverseConstraintDual(), c, vals[i])
        end
    end

    DiffOpt.reverse_differentiate!(model)

    params_list = params isa AbstractArray ? params : [params]
    return [MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)).value for p in params_list]
end

function get_param(params, param_idx)
    return params isa AbstractArray ? params[param_idx] : params
end


function get_problem_dims(build_model)
    model = Model(() -> DiffOpt.diff_optimizer(MadNLP.Optimizer))
    MOI.set(model, DiffOpt.ModelConstructor(), DiffOpt.NonLinearProgram.Model)
    set_silent(model)
    vars, params = build_model(model)
    optimize!(model)

    cons = get_constraint_refs(model)
    lb_cons, ub_cons = get_bound_constraint_refs(model)

    return length(vars), length(cons), length(lb_cons), length(ub_cons)
end

function stack_solution_forward(model, p_vals, params, vars, cons, lb_cons, ub_cons)
    params_list = params isa AbstractArray ? params : [params]
    for (i, p) in enumerate(params_list)
        set_parameter_value(p, p_vals[i])
    end
    optimize!(model)
    x_vals = Float64[value(v) for v in vars]
    λ_vals = Float64[dual(c) for c in cons]
    zl_vals = Float64[dual(c) for c in lb_cons]
    zu_vals = Float64[dual(c) for c in ub_cons]
    return vcat(x_vals, λ_vals, zl_vals, zu_vals)
end

function run_finitediff_forward(build_model; param_idx = 1, dp = 1.0)
    model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
    set_silent(model)
    vars, params = build_model(model)
    optimize!(model)

    cons = get_constraint_refs(model)
    lb_cons, ub_cons = get_bound_constraint_refs(model)

    params_list = params isa AbstractArray ? params : [params]
    n_params = length(params_list)
    p_base = Float64[parameter_value(p) for p in params_list]

    jac = FiniteDiff.finite_difference_jacobian(
        p -> stack_solution_forward(model, p, params, vars, cons, lb_cons, ub_cons),
        p_base
    )

    Δp = zeros(n_params)
    Δp[param_idx] = dp

    Δs = jac * Δp

    n_x = length(vars)
    n_con = length(cons)
    n_lb = length(lb_cons)
    n_ub = length(ub_cons)

    dx = Δs[1:n_x]
    dλ = Δs[n_x+1:n_x+n_con]
    dzl = Δs[n_x+n_con+1:n_x+n_con+n_lb]
    dzu = Δs[n_x+n_con+n_lb+1:end]

    return dx, dλ, dzl, dzu
end

function run_finitediff_reverse(build_model, dL_dx_vals, dL_dλ_vals, dL_dzl_vals, dL_dzu_vals)
    model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
    set_silent(model)
    vars, params = build_model(model)
    optimize!(model)

    cons = get_constraint_refs(model)
    lb_cons, ub_cons = get_bound_constraint_refs(model)

    params_list = params isa AbstractArray ? params : [params]
    n_params = length(params_list)
    p_base = Float64[parameter_value(p) for p in params_list]

    jac = FiniteDiff.finite_difference_jacobian(
        p -> stack_solution_forward(model, p, params, vars, cons, lb_cons, ub_cons),
        p_base
    )

    dL_ds = Float64[dL_dx_vals; dL_dλ_vals; dL_dzl_vals; dL_dzu_vals]

    grad_p = jac' * dL_ds

    return grad_p
end
