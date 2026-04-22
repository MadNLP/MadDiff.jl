# ============================================================================
# Reverse-mode (VJP) entry points for `DiffOptWrapper`.
# ============================================================================

# Each seed setter marks outputs dirty; `reverse_differentiate!` clears once
# before running. This keeps a loop of many `MOI.set` calls O(1) instead of
# O(N · #outputs).
function MOI.set(wrapper::DiffOptWrapper, ::MadDiff.ReverseVariablePrimal,
                 vi::MOI.VariableIndex, value::Real)
    wrapper.reverse.primal_seeds[vi] = value
    _mark_outputs_dirty!(wrapper)
    return nothing
end

function MOI.set(wrapper::DiffOptWrapper, ::MadDiff.ReverseConstraintDual,
                 ci::MOI.ConstraintIndex, value::Real)
    wrapper.reverse.dual_seeds[ci] = value
    _mark_outputs_dirty!(wrapper)
    return nothing
end

function MOI.set(wrapper::DiffOptWrapper{OT, T},
                 ::MadDiff.ReverseObjectiveSensitivity, value::Real) where {OT, T}
    wrapper.reverse.dobj = value
    _mark_outputs_dirty!(wrapper)
    return nothing
end

function MadDiff.reverse_differentiate!(wrapper::DiffOptWrapper)
    _outputs_dirty(wrapper) && _clear_outputs!(wrapper)
    wrapper.diff_time = @elapsed _reverse_differentiate_impl!(wrapper)
    return nothing
end

function _reverse_differentiate_impl!(wrapper::DiffOptWrapper{OT, T}) where {OT, T}
    inner  = wrapper.inner
    solver = inner.solver
    _check_ready(inner, solver)

    sens  = _get_sensitivity_solver!(wrapper)
    n_x   = NLPModels.get_nvar(sens.solver.nlp)
    n_con = NLPModels.get_ncon(solver.nlp)

    dL_dx  = _scratch!(wrapper.work.dL_dx,  n_x)
    dL_dy  = _scratch!(wrapper.work.dL_dy,  n_con)
    dL_dzl = _scratch!(wrapper.work.dL_dzl, n_x)
    dL_dzu = _scratch!(wrapper.work.dL_dzu, n_x)

    for (vi, val) in wrapper.reverse.primal_seeds
        dL_dx[vi.value] = val
    end
    n_qp = length(inner.qp_data.constraints)
    for (ci, val) in wrapper.reverse.dual_seeds
        _place_reverse_seed!(ci, val, inner, n_qp, dL_dy, dL_dzl, dL_dzu)
    end
    dL_dy .*= -solver.cb.obj_sign

    dev(v) = _to_device(v, solver.y)
    result = MadDiff.vector_jacobian_product!(
        sens;
        dL_dx  = dev(dL_dx),  dL_dy  = dev(dL_dy),
        dL_dzl = dev(dL_dzl), dL_dzu = dev(dL_dzu),
        dobj   = wrapper.reverse.dobj,
    )
    grad_p = _to_host(result.grad_p)

    for (ci, vi) in wrapper.param_ci_to_vi
        wrapper.reverse.param_outputs[ci] = grad_p[inner.param_vi_to_idx[vi]]
    end
    return nothing
end

# ---------- dual seed placement ----------

# Variable-bound constraints push seeds into the bound-dual slots; every other
# constraint kind routes to `dL_dy` via `_constraint_row`.
function _place_reverse_seed!(ci::MOI.ConstraintIndex{MOI.VariableIndex, S},
                               val, inner, _n_qp, _dL_dy, dL_dzl, dL_dzu) where {S}
    idx = MOI.get(inner, MOI.ConstraintFunction(), ci).value
    if S <: MOI.GreaterThan
        dL_dzl[idx] =  val
    elseif S <: MOI.LessThan
        dL_dzu[idx] = -val
    elseif S <: Union{MOI.Interval, MOI.EqualTo}
        dL_dzl[idx] =  val
        dL_dzu[idx] = -val
    else
        error("MadDiff: unsupported variable-bound set $(S) for reverse seed.")
    end
    return nothing
end

_place_reverse_seed!(ci::MOI.ConstraintIndex, val, _inner, n_qp, dL_dy, _dzl, _dzu) =
    (dL_dy[_constraint_row(n_qp, ci)] = val; nothing)

# ---------- getters ----------

MOI.get(wrapper::DiffOptWrapper, ::MadDiff.ReverseConstraintSet,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}}) where {T} =
    MOI.Parameter(wrapper.reverse.param_outputs[ci])

MadDiff.get_reverse_parameter(wrapper::DiffOptWrapper,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}}) where {T} =
    wrapper.reverse.param_outputs[ci]
