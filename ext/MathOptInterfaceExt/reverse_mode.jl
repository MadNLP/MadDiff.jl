function MOI.set(
        model::Optimizer,
        ::MadDiff.ReverseVariablePrimal,
        vi::MOI.VariableIndex,
        value::Real,
    )
    model.reverse.primal_seeds[vi] = value
    return _clear_outputs!(model)  # keep KKT factorization
end

function MOI.set(
        model::Optimizer,
        ::MadDiff.ReverseConstraintDual,
        ci::MOI.ConstraintIndex,
        value::Real,
    )
    model.reverse.dual_seeds[ci] = value
    return _clear_outputs!(model)  # keep KKT factorization
end

function MadDiff.reverse_differentiate!(model::Optimizer)
    model.diff_time = @elapsed _reverse_differentiate_impl!(model)
    return nothing
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, dL_dy, dL_dzl, dL_dzu
) where {S <: MOI.GreaterThan}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = vi.value
    dL_dzl[idx] = val
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, dL_dy, dL_dzl, dL_dzu
) where {S <: MOI.LessThan}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = vi.value
    dL_dzu[idx] = -val
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, dL_dy, dL_dzl, dL_dzu
) where {S <: MOI.Interval}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = vi.value
    dL_dzl[idx] = val
    dL_dzu[idx] = -val
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, dL_dy, dL_dzl, dL_dzu
) where {S <: MOI.EqualTo}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = vi.value
    dL_dzl[idx] = val
    dL_dzu[idx] = -val
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{F, S}, val, inner, dL_dy, dL_dzl, dL_dzu
) where {F, S}
    row = _constraint_row(inner, ci)
    dL_dy[row] = val
end

function _reverse_differentiate_impl!(model::Optimizer{OT, T}) where {OT, T}
    inner = model.inner
    solver = inner.solver

    isnothing(solver) && error("Optimizer must be solved first")
    MadDiff.assert_solved_and_feasible(solver)
    isempty(inner.parameters) && error("No parameters in model")

    sens = _get_sensitivity_solver!(model)

    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(solver.nlp)

    dL_dx = _get_dL_dx!(model, n_x)
    for (vi, val) in model.reverse.primal_seeds
        idx = vi.value
        dL_dx[idx] = val
    end

    dL_dy = _get_dL_dy!(model, n_con)
    dL_dzl = _get_dL_dzl!(model, n_x)
    dL_dzu = _get_dL_dzu!(model, n_x)

    for (ci, val) in model.reverse.dual_seeds
        _process_reverse_dual_input!(ci, val, inner, dL_dy, dL_dzl, dL_dzu)
    end

    dL_dy .*= -solver.cb.obj_sign

    VT = typeof(solver.y)
    if VT <: Vector
        result = MadDiff.reverse_differentiate!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu)
        grad_p_cpu = result.grad_p
    else
        dL_dx_gpu = dL_dx isa VT ? dL_dx : VT(dL_dx)
        dL_dy_gpu = dL_dy isa VT ? dL_dy : VT(dL_dy)
        dL_dzl_gpu = dL_dzl isa VT ? dL_dzl : VT(dL_dzl)
        dL_dzu_gpu = dL_dzu isa VT ? dL_dzu : VT(dL_dzu)
        result = MadDiff.reverse_differentiate!(sens; dL_dx=dL_dx_gpu, dL_dy=dL_dy_gpu, dL_dzl=dL_dzl_gpu, dL_dzu=dL_dzu_gpu)
        grad_p_cpu = result.grad_p isa Vector ? result.grad_p : Array(result.grad_p)
    end

    for (ci, vi) in model.param_ci_to_vi
        idx = inner.param_vi_to_idx[vi]
        model.reverse.param_outputs[ci] = grad_p_cpu[idx]
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

function MadDiff.get_reverse_parameter(
        model::Optimizer,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}},
    ) where {T}
    return model.reverse.param_outputs[ci]
end
