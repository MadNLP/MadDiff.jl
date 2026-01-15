function MOI.set(
        model::Optimizer,
        ::MadDiff.ReverseVariablePrimal,
        vi::MOI.VariableIndex,
        value::Real,
    )
    model.reverse.primal_inputs[vi] = value
    return _clear_outputs!(model)  # keep KKT factorization
end

function MOI.set(
        model::Optimizer,
        ::MadDiff.ReverseConstraintDual,
        ci::MOI.ConstraintIndex,
        value::Real,
    )
    model.reverse.dual_inputs[ci] = value
    return _clear_outputs!(model)  # keep KKT factorization
end

function MadDiff.reverse_differentiate!(model::Optimizer)
    model.diff_time = @elapsed _reverse_differentiate_impl!(model)
    return nothing
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, dL_dλ, dL_dzl, dL_dzu, vi_to_lb_idx, vi_to_ub_idx
) where {S <: MOI.GreaterThan}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    i = get(vi_to_lb_idx, vi, 0)
    !iszero(i) && (dL_dzl[i] = val)
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, dL_dλ, dL_dzl, dL_dzu, vi_to_lb_idx, vi_to_ub_idx
) where {S <: MOI.LessThan}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    i = get(vi_to_ub_idx, vi, 0)
    !iszero(i) && (dL_dzu[i] = -val)
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, dL_dλ, dL_dzl, dL_dzu, vi_to_lb_idx, vi_to_ub_idx
) where {S <: MOI.Interval}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    i_lb = get(vi_to_lb_idx, vi, 0)
    i_ub = get(vi_to_ub_idx, vi, 0)
    !iszero(i_lb) && (dL_dzl[i_lb] = val)
    !iszero(i_ub) && (dL_dzu[i_ub] = -val)
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, dL_dλ, dL_dzl, dL_dzu, vi_to_lb_idx, vi_to_ub_idx
) where {S <: MOI.EqualTo}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    i_lb = get(vi_to_lb_idx, vi, 0)
    i_ub = get(vi_to_ub_idx, vi, 0)
    !iszero(i_lb) && (dL_dzl[i_lb] = val)
    !iszero(i_ub) && (dL_dzu[i_ub] = -val)
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{F, S}, val, inner, dL_dλ, dL_dzl, dL_dzu, vi_to_lb_idx, vi_to_ub_idx
) where {F, S}
    row = _constraint_row(inner, ci)
    dL_dλ[row] = val
end

function _make_param_pullback_closure(model, ctx::SensitivityContext{T}) where {T}
    return function(out, adj_x, adj_λ, adj_zl, adj_zu, sens)
        x = model.inner.result.solution
        n_con = length(adj_λ)
        y = _get_y_cache!(model, n_con)
        MadNLP.unpack_y!(y, model.inner.solver.cb, model.inner.solver.y)

        grad_p = _compute_param_pullback!(model.inner, x, y, adj_x, adj_λ, ctx)
        out .= grad_p
        return out
    end
end

function _reverse_differentiate_impl!(model::Optimizer{OT, T}) where {OT, T}
    inner = model.inner
    solver = inner.solver

    isnothing(solver) && error("Optimizer must be solved first")
    MadDiff.assert_solved_and_feasible(solver)
    isempty(inner.parameters) && error("No parameters in model")

    ctx = _get_sensitivity_context!(model)
    n_con = NLPModels.get_ncon(solver.nlp)
    sens = _get_sensitivity_solver!(model)

    dL_dx = _get_dL_dx!(model, ctx.n_x)
    for (vi, val) in model.reverse.primal_inputs
        dL_dx[ctx.primal_idx[vi]] = val
    end

    dL_dλ = _get_dL_dλ!(model, n_con)

    dL_dzl = _get_dL_dzl!(model, sens.dims.n_lb)
    dL_dzu = _get_dL_dzu!(model, sens.dims.n_ub)
    vi_to_lb_idx = model.vi_to_lb_idx
    vi_to_ub_idx = model.vi_to_ub_idx

    for (ci, val) in model.reverse.dual_inputs
        _process_reverse_dual_input!(ci, val, inner, dL_dλ, dL_dzl, dL_dzu, vi_to_lb_idx, vi_to_ub_idx)
    end

    # moi and madnlp have opposite conventions
    dL_dλ .*= -one(T)
    result = MadDiff.reverse_differentiate!(sens; dL_dx, dL_dλ, dL_dzl, dL_dzu)

    ∂L_∂p = result.grad_p
    for (ci, vi) in model.param_ci_to_vi
        model.reverse.param_outputs[ci] = ∂L_∂p[ctx.param_idx[vi]]
    end
    return
end

function MOI.get(
        model::Optimizer,
        ::MadDiff.ReverseConstraintSet,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}},
    ) where {T}
    return MOI.Parameter(model.reverse.param_outputs[ci])
end
