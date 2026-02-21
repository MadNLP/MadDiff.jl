function MOI.set(
        wrapper::DiffOptWrapper,
        ::MadDiff.ForwardConstraintSet,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}},
        set::MOI.Parameter{T},
    ) where {T}
    wrapper.forward.param_perturbations[ci] = set.value
    return _clear_outputs!(wrapper)  # keep KKT factorization
end

function MadDiff.forward_differentiate!(wrapper::DiffOptWrapper)
    wrapper.diff_time = @elapsed _forward_differentiate_impl!(wrapper)
    return nothing
end

function _forward_differentiate_impl!(wrapper::DiffOptWrapper{OT, T}) where {OT, T}
    inner = wrapper.inner
    solver = inner.solver

    isnothing(solver) && error("Optimizer must be solved first")
    MadDiff.assert_solved_and_feasible(solver)
    isempty(inner.parameters) && error("No parameters in model")

    n_p = inner.param_n_p
    Δp = zeros(T, n_p)
    for (ci, dp) in wrapper.forward.param_perturbations
        vi = wrapper.param_ci_to_vi[ci]
        idx = inner.param_vi_to_idx[vi]
        Δp[idx] = dp
    end

    sens = _get_sensitivity_solver!(wrapper)

    VT = typeof(solver.y)
    if VT <: Vector
        result = MadDiff.jacobian_vector_product!(sens, Δp)
        dx_cpu = result.dx
        dy_cpu = result.dy
    else
        # TODO: pre-allocate
        Δp_gpu = VT(Δp)
        result = MadDiff.jacobian_vector_product!(sens, Δp_gpu)
        dx_cpu = Array(result.dx)
        dy_cpu = Array(result.dy)
    end

    primal_vars = inner.param_var_order
    for (i, vi) in enumerate(primal_vars)
        wrapper.forward.primal_sensitivities[vi] = dx_cpu[i]
    end

    n_con = NLPModels.get_ncon(solver.nlp)
    obj_sign = solver.cb.obj_sign
    dy = _get_dy_cache!(wrapper, n_con)
    dy .= (.-obj_sign) .* dy_cpu

    _store_dual_sensitivities!(wrapper.forward.dual_sensitivities, inner, dy)
    _store_bound_dual_sensitivities!(wrapper, sens, result, inner)
    wrapper.forward.objective_sensitivity = result.dobj[]
    return
end

function _constraint_row(inner, ci::MOI.ConstraintIndex{F, S}) where {F, S}
    if F == MOI.ScalarNonlinearFunction
        return length(inner.qp_data.constraints) + ci.value
    else
        return ci.value
    end
end

function _store_dual_sensitivities!(dual_sensitivities, inner, dy)
    for (F, S) in MOI.get(inner, MOI.ListOfConstraintTypesPresent())
        F == MOI.VariableIndex && continue
        S <: MOI.Parameter && continue
        for ci in MOI.get(inner, MOI.ListOfConstraintIndices{F, S}())
            row = _constraint_row(inner, ci)
            dual_sensitivities[ci] = dy[row]
        end
    end
    if inner.nlp_model !== nothing
        n_qp = length(inner.qp_data.constraints)
        for (nlp_idx, con) in inner.nlp_model.constraints
            S = typeof(con.set)
            ci = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction, S}(nlp_idx.value)
            row = n_qp + nlp_idx.value
            dual_sensitivities[ci] = dy[row]
        end
    end
    return
end

function _store_bound_dual_sensitivities!(wrapper, sens, result, inner)
    dsens = wrapper.forward.dual_sensitivities

    dzl = result.dzl isa Vector ? result.dzl : Array(result.dzl)
    dzu = result.dzu isa Vector ? result.dzu : Array(result.dzu)

    for ci in MOI.get(inner, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.GreaterThan{Float64}}())
        vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
        idx = vi.value
        dsens[ci] = dzl[idx]
    end
    for ci in MOI.get(inner, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.LessThan{Float64}}())
        vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
        idx = vi.value
        dsens[ci] = -dzu[idx]
    end
    for ci in MOI.get(inner, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Interval{Float64}}())
        vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
        idx = vi.value
        dsens[ci] = dzl[idx] - dzu[idx]
    end
    for ci in MOI.get(inner, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.EqualTo{Float64}}())
        vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
        idx = vi.value
        dsens[ci] = dzl[idx] - dzu[idx]
    end

    return
end

function MOI.get(wrapper::DiffOptWrapper, ::MadDiff.ForwardVariablePrimal, vi::MOI.VariableIndex)
    return wrapper.forward.primal_sensitivities[vi]
end

function MOI.get(wrapper::DiffOptWrapper, ::MadDiff.ForwardConstraintDual, ci::MOI.ConstraintIndex)
    return wrapper.forward.dual_sensitivities[ci]
end

function MOI.get(wrapper::DiffOptWrapper, ::MadDiff.ForwardObjectiveSensitivity)
    return wrapper.forward.objective_sensitivity
end
