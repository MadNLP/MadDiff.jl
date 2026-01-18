const VI = MOI.VariableIndex
const LB_CI{T} = MOI.ConstraintIndex{VI, MOI.GreaterThan{T}}
const UB_CI{T} = MOI.ConstraintIndex{VI, MOI.LessThan{T}}
const INT_CI{T} = MOI.ConstraintIndex{VI, MOI.Interval{T}}
const EQ_CI{T} = MOI.ConstraintIndex{VI, MOI.EqualTo{T}}

mutable struct SensitivityContext{T}
    primal_vars::Vector{VI}
    params::Vector{VI}
    primal_idx::Dict{VI, Int}
    param_idx::Dict{VI, Int}
    vi_to_lb_ci::Dict{VI, LB_CI{T}}
    vi_to_ub_ci::Dict{VI, UB_CI{T}}
    vi_to_interval_ci::Dict{VI, INT_CI{T}}
    vi_to_eq_ci::Dict{VI, EQ_CI{T}}
    n_x::Int
    n_p::Int
    n_qp::Int
    n_nlp::Int
    nlp_evaluator::Union{Nothing, MOI.Nonlinear.Evaluator}
    x_combined::Vector{T}
    p_vals::Vector{T}
    d2L_dxdp::Vector{T}  # JVP output: ∂²L/∂x∂p * Δp
    dg_dp::Vector{T}      # JVP output: ∂g/∂p * Δp
    Δp::Vector{T}
    grad_p::Vector{T}     # VJP output: ∂L/∂p
    v_extended::Vector{T}
    jvp_result::Vector{T}
    hvp_result::Vector{T}
    param_nlp_refs::Vector{MOI.Nonlinear.ParameterIndex}
end

function SensitivityContext(primal_vars, params, n_qp, n_nlp, vi_to_lb_ci, vi_to_ub_ci, vi_to_interval_ci, vi_to_eq_ci; T = Float64, param_nlp_refs = MOI.Nonlinear.ParameterIndex[])
    n_x = length(primal_vars)
    n_p = length(params)
    n_con = n_qp + n_nlp
    primal_idx = Dict(v => i for (i, v) in enumerate(primal_vars))
    param_idx = Dict(p => i for (i, p) in enumerate(params))
    return SensitivityContext{T}(
        primal_vars, params, primal_idx, param_idx, vi_to_lb_ci, vi_to_ub_ci, vi_to_interval_ci, vi_to_eq_ci,
        n_x, n_p, n_qp, n_nlp, nothing,
        zeros(T, n_x + n_p),
        zeros(T, n_p),
        zeros(T, n_x),
        zeros(T, n_con),
        zeros(T, n_p),
        zeros(T, n_p),
        zeros(T, n_x + n_p),
        zeros(T, n_nlp),
        zeros(T, n_x + n_p),
        param_nlp_refs,
    )
end

get_primal_idx(ctx::SensitivityContext, vi::MOI.VariableIndex) = get(ctx.primal_idx, vi, 0)
get_param_idx(ctx::SensitivityContext, vi::MOI.VariableIndex) = get(ctx.param_idx, vi, 0)

function SensitivityContext(model; T = Float64)
    primal_vars = sort(MOI.get(model.variables, MOI.ListOfVariableIndices()); by = x -> x.value)
    params = sort(collect(keys(model.parameters)); by = x -> x.value)
    n_qp = length(model.qp_data.constraints)
    n_nlp = length(model.nlp_data.constraint_bounds)

    vi_to_lb_ci = Dict{VI, LB_CI{T}}()
    vi_to_ub_ci = Dict{VI, UB_CI{T}}()
    vi_to_interval_ci = Dict{VI, INT_CI{T}}()
    vi_to_eq_ci = Dict{VI, EQ_CI{T}}()

    for ci in MOI.get(model, MOI.ListOfConstraintIndices{VI, MOI.GreaterThan{T}}())
        vi = MOI.get(model, MOI.ConstraintFunction(), ci)
        vi_to_lb_ci[vi] = ci
    end
    for ci in MOI.get(model, MOI.ListOfConstraintIndices{VI, MOI.LessThan{T}}())
        vi = MOI.get(model, MOI.ConstraintFunction(), ci)
        vi_to_ub_ci[vi] = ci
    end
    for ci in MOI.get(model, MOI.ListOfConstraintIndices{VI, MOI.Interval{T}}())
        vi = MOI.get(model, MOI.ConstraintFunction(), ci)
        vi_to_interval_ci[vi] = ci
    end
    for ci in MOI.get(model, MOI.ListOfConstraintIndices{VI, MOI.EqualTo{T}}())
        vi = MOI.get(model, MOI.ConstraintFunction(), ci)
        vi_to_eq_ci[vi] = ci
    end

    param_nlp_refs = [model.parameters[p] for p in params]

    return SensitivityContext(primal_vars, params, n_qp, n_nlp, vi_to_lb_ci, vi_to_ub_ci, vi_to_interval_ci, vi_to_eq_ci; T, param_nlp_refs)
end

function _remap_params_to_vars(expr, n_x)
    new_expr = MOI.Nonlinear.Expression()
    resize!(new_expr.nodes, length(expr.nodes))
    for (i, node) in enumerate(expr.nodes)
        if node.type == MOI.Nonlinear.NODE_PARAMETER
            # remap parameter index p to variable index n_x + p
            new_expr.nodes[i] = MOI.Nonlinear.Node(MOI.Nonlinear.NODE_MOI_VARIABLE, n_x + node.index, node.parent)
        else
            new_expr.nodes[i] = node
        end
    end
    append!(new_expr.values, expr.values)
    return new_expr
end

function create_param_as_vars_model(nlp_model, n_x)
    new_model = MOI.Nonlinear.Model()
    new_model.operators = nlp_model.operators

    for expr in nlp_model.expressions
        new_expr = _remap_params_to_vars(expr, n_x)
        push!(new_model.expressions, new_expr)
    end

    if !isnothing(nlp_model.objective)
        new_model.objective = _remap_params_to_vars(nlp_model.objective, n_x)
    end

    for (ci, constraint) in nlp_model.constraints
        new_expr = _remap_params_to_vars(constraint.expression, n_x)
        new_model.constraints[ci] = MOI.Nonlinear.Constraint(new_expr, constraint.set)
    end

    return new_model
end

function create_sensitivity_evaluator(nlp_model, n_x, n_p)
    sens_model = create_param_as_vars_model(nlp_model, n_x)
    combined_vars = [MOI.VariableIndex(i) for i in 1:(n_x + n_p)]
    evaluator = MOI.Nonlinear.Evaluator(sens_model, MOI.Nonlinear.SparseReverseMode(), combined_vars)
    MOI.initialize(evaluator, [:JacVec, :HessVec])
    return evaluator
end