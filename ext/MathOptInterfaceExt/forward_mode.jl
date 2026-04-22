# ============================================================================
# Forward-mode (JVP) entry points for `DiffOptWrapper`.
# ============================================================================

function MOI.set(
    wrapper::DiffOptWrapper,
    ::MadDiff.ForwardConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}},
    set::MOI.Parameter{T},
) where {T}
    wrapper.forward.param_perturbations[ci] = set.value
    _mark_outputs_dirty!(wrapper)
    return nothing
end

function MadDiff.forward_differentiate!(wrapper::DiffOptWrapper)
    _outputs_dirty(wrapper) && _clear_outputs!(wrapper)
    wrapper.diff_time = @elapsed _forward_differentiate_impl!(wrapper)
    return nothing
end

function _forward_differentiate_impl!(wrapper::DiffOptWrapper{OT, T}) where {OT, T}
    inner  = wrapper.inner
    solver = inner.solver
    _check_ready(inner, solver)

    Δp = _scratch!(wrapper.work.dp, inner.param_n_p)
    for (ci, dp) in wrapper.forward.param_perturbations
        Δp[inner.param_vi_to_idx[wrapper.param_ci_to_vi[ci]]] = dp
    end

    sens      = _get_sensitivity_solver!(wrapper)
    Δp_dev    = _to_device(Δp, solver.y)
    result    = MadDiff.jacobian_vector_product!(sens, Δp_dev)
    dx_host   = _to_host(result.dx)
    dy_host   = _to_host(result.dy)

    for (i, vi) in enumerate(inner.param_var_order)
        wrapper.forward.primal_sensitivities[vi] = dx_host[i]
    end

    dy = _scratch!(wrapper.work.dy, NLPModels.get_ncon(solver.nlp))
    dy .= .-solver.cb.obj_sign .* dy_host
    _store_dual_sensitivities!(wrapper.forward.dual_sensitivities, inner, dy)
    _store_bound_dual_sensitivities!(wrapper.forward.dual_sensitivities,
                                     inner, result)

    wrapper.forward.objective_sensitivity = nothing
    wrapper.forward.jvp_result            = result
    wrapper.forward.param_direction       = Δp_dev
    return nothing
end

# ---------- result marshalling ----------

function _store_dual_sensitivities!(dst, inner, dy)
    n_qp = length(inner.qp_data.constraints)
    for (F, S) in MOI.get(inner, MOI.ListOfConstraintTypesPresent())
        (F === MOI.VariableIndex || S <: MOI.Parameter) && continue
        for ci in MOI.get(inner, MOI.ListOfConstraintIndices{F, S}())
            dst[ci] = dy[_constraint_row(n_qp, ci)]
        end
    end
    if inner.nlp_model !== nothing
        for (nlp_idx, con) in inner.nlp_model.constraints
            S  = typeof(con.set)
            ci = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction, S}(nlp_idx.value)
            dst[ci] = dy[n_qp + nlp_idx.value]
        end
    end
    return nothing
end

function _store_bound_dual_sensitivities!(dst, inner, result)
    dzl = _to_host(result.dzl)
    dzu = _to_host(result.dzu)
    # Per bound-type pass; each `MOI.ListOfConstraintIndices` query iterates
    # one set kind. We don't try to fuse into a single pass because MOI
    # doesn't expose a typed "all variable-bound constraints" iterator;
    # gathering them ourselves would mean reimplementing the optimizer's
    # index layout. Four small queries are cheaper than one reflective scan.
    @inline _apply!(S, combine) = for ci in MOI.get(inner,
            MOI.ListOfConstraintIndices{MOI.VariableIndex, S}())
        idx      = MOI.get(inner, MOI.ConstraintFunction(), ci).value
        dst[ci]  = combine(dzl[idx], dzu[idx])
    end
    _apply!(MOI.GreaterThan{Float64}, (l, _u) ->  l)
    _apply!(MOI.LessThan{Float64},    (_l, u) -> -u)
    _apply!(MOI.Interval{Float64},    (l, u)  ->  l - u)
    _apply!(MOI.EqualTo{Float64},     (l, u)  ->  l - u)
    return nothing
end

_constraint_row(n_qp::Int, ci::MOI.ConstraintIndex{F, S}) where {F, S} =
    F === MOI.ScalarNonlinearFunction ? n_qp + ci.value : ci.value

# ---------- attribute getters ----------

MOI.get(wrapper::DiffOptWrapper, ::MadDiff.ForwardVariablePrimal, vi::MOI.VariableIndex) =
    wrapper.forward.primal_sensitivities[vi]

MOI.get(wrapper::DiffOptWrapper, ::MadDiff.ForwardConstraintDual, ci::MOI.ConstraintIndex) =
    wrapper.forward.dual_sensitivities[ci]

function MOI.get(wrapper::DiffOptWrapper{OT, T},
                 ::MadDiff.ForwardObjectiveSensitivity) where {OT, T}
    fwd = wrapper.forward
    fwd.objective_sensitivity === nothing || return fwd.objective_sensitivity
    fwd.jvp_result === nothing && return zero(T)

    MadDiff.compute_objective_sensitivity!(
        fwd.jvp_result, _get_sensitivity_solver!(wrapper), fwd.param_direction,
    )
    fwd.objective_sensitivity = fwd.jvp_result.dobj[]
    return fwd.objective_sensitivity
end

# ---------- helpers ----------

function _check_ready(inner, solver)
    solver === nothing && error("MadDiff: optimizer must be solved first.")
    MadDiff.assert_solved_and_feasible(solver)
    isempty(inner.parameters) && error("MadDiff: model has no parameters.")
    return nothing
end

# Move `Δp` to the device hosting `ref`; leave `Vector`s untouched.
_to_device(Δp::AbstractVector, ref::Vector) = Δp
_to_device(Δp::AbstractVector, ref)         = typeof(ref)(Δp)

_to_host(x::Vector) = x
_to_host(x)         = Array(x)
