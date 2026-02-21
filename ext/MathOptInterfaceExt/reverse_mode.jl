function MOI.set(
        wrapper::DiffOptWrapper,
        ::MadDiff.ReverseVariablePrimal,
        vi::MOI.VariableIndex,
        value::Real,
    )
    wrapper.reverse.primal_seeds[vi] = value
    return _clear_outputs!(wrapper)  # keep KKT factorization
end

function MOI.set(
        wrapper::DiffOptWrapper,
        ::MadDiff.ReverseConstraintDual,
        ci::MOI.ConstraintIndex,
        value::Real,
    )
    wrapper.reverse.dual_seeds[ci] = value
    return _clear_outputs!(wrapper)  # keep KKT factorization
end

function MOI.set(
        wrapper::DiffOptWrapper{OT, T},
        ::MadDiff.ReverseObjectiveSensitivity,
        value::Real,
    ) where {OT, T}
    wrapper.reverse.dobj = value
    return _clear_outputs!(wrapper)  # keep KKT factorization
end

function MadDiff.reverse_differentiate!(wrapper::DiffOptWrapper)
    wrapper.diff_time = @elapsed _reverse_differentiate_impl!(wrapper)
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

function _reverse_differentiate_impl!(wrapper::DiffOptWrapper{OT, T}) where {OT, T}
    inner = wrapper.inner
    solver = inner.solver

    isnothing(solver) && error("Optimizer must be solved first")
    MadDiff.assert_solved_and_feasible(solver)
    isempty(inner.parameters) && error("No parameters in model")

    sens = _get_sensitivity_solver!(wrapper)

    n_x = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(solver.nlp)

    dL_dx = _get_dL_dx!(wrapper, n_x)
    for (vi, val) in wrapper.reverse.primal_seeds
        idx = vi.value
        dL_dx[idx] = val
    end

    dL_dy = _get_dL_dy!(wrapper, n_con)
    dL_dzl = _get_dL_dzl!(wrapper, n_x)
    dL_dzu = _get_dL_dzu!(wrapper, n_x)

    for (ci, val) in wrapper.reverse.dual_seeds
        _process_reverse_dual_input!(ci, val, inner, dL_dy, dL_dzl, dL_dzu)
    end

    dL_dy .*= -solver.cb.obj_sign
    dobj = wrapper.reverse.dobj

    VT = typeof(solver.y)
    if VT <: Vector
        result = MadDiff.vector_jacobian_product!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
        grad_p_cpu = result.grad_p
    else
        # TODO: pre-allocate
        dL_dx_gpu = VT(dL_dx)
        dL_dy_gpu = VT(dL_dy)
        dL_dzl_gpu = VT(dL_dzl)
        dL_dzu_gpu = VT(dL_dzu)
        result = MadDiff.vector_jacobian_product!(sens; dL_dx=dL_dx_gpu, dL_dy=dL_dy_gpu, dL_dzl=dL_dzl_gpu, dL_dzu=dL_dzu_gpu, dobj)
        grad_p_cpu = Array(result.grad_p)
    end

    for (ci, vi) in wrapper.param_ci_to_vi
        idx = inner.param_vi_to_idx[vi]
        wrapper.reverse.param_outputs[ci] = grad_p_cpu[idx]
    end
    return
end

function MOI.get(
        wrapper::DiffOptWrapper,
        ::MadDiff.ReverseConstraintSet,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}},
    ) where {T}
    return MOI.Parameter(wrapper.reverse.param_outputs[ci])
end

function MadDiff.get_reverse_parameter(
        wrapper::DiffOptWrapper,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}},
    ) where {T}
    return wrapper.reverse.param_outputs[ci]
end
