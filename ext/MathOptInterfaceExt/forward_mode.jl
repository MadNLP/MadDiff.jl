function MOI.set(
        model::Optimizer,
        ::MadDiff.ForwardConstraintSet,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}},
        set::MOI.Parameter{T},
    ) where {T}
    model.forward.param_perturbations[ci] = set.value
    return _clear_outputs!(model)  # keep KKT factorization
end

function MadDiff.forward_differentiate!(model::Optimizer)
    model.diff_time = @elapsed _forward_differentiate_impl!(model)
    return nothing
end

function _forward_differentiate_impl!(model::Optimizer{OT, T}) where {OT, T}
    inner = model.inner
    solver = inner.solver

    isnothing(solver) && error("Optimizer must be solved first")
    MadDiff.assert_solved_and_feasible(solver)
    isempty(inner.parameters) && error("No parameters in model")

    n_p = inner.param_n_p
    Δp = zeros(T, n_p)
    for (ci, dp) in model.forward.param_perturbations
        vi = model.param_ci_to_vi[ci]
        idx = inner.param_vi_to_idx[vi]
        Δp[idx] = dp
    end

    sens = _get_sensitivity_solver!(model)

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
        model.forward.primal_sensitivities[vi] = dx_cpu[i]
    end

    n_con = NLPModels.get_ncon(solver.nlp)
    obj_sign = solver.cb.obj_sign
    dy = _get_dy_cache!(model, n_con)
    dy .= (.-obj_sign) .* dy_cpu

    _store_dual_sensitivities!(model.forward.dual_sensitivities, model.forward.vector_dual_sensitivities, inner, dy)
    _store_bound_dual_sensitivities!(model, sens, result, inner)
    model.forward.objective_sensitivity = result.dobj[]
    return
end

function _constraint_row(inner, ci::MOI.ConstraintIndex{F, S}) where {F, S}
    if F == MOI.ScalarNonlinearFunction
        return length(inner.qp_data.constraints) + ci.value
    else
        return ci.value
    end
end

function _vno_rows(
    inner,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.VectorNonlinearOracle{Float64}},
)
    offset = length(inner.qp_data)
    for i in 1:(ci.value - 1)
        _, s = inner.vector_nonlinear_oracle_constraints[i]
        offset += s.set.output_dimension
    end
    _, s = inner.vector_nonlinear_oracle_constraints[ci.value]
    return offset .+ (1:s.set.output_dimension)
end

function _store_dual_sensitivities!(dual_sensitivities, vector_dual_sensitivities, inner, dy)
    for (F, S) in MOI.get(inner, MOI.ListOfConstraintTypesPresent())
        F == MOI.VariableIndex && continue
        S <: MOI.Parameter && continue
        if F == MOI.VectorOfVariables && S == MOI.VectorNonlinearOracle{Float64}
            for ci in MOI.get(inner, MOI.ListOfConstraintIndices{F, S}())
                rows = _vno_rows(inner, ci)
                vector_dual_sensitivities[ci] = dy[rows]
            end
            continue
        end
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

function _store_bound_dual_sensitivities!(model, sens, result, inner)
    dsens = model.forward.dual_sensitivities

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

function MOI.get(model::Optimizer, ::MadDiff.ForwardVariablePrimal, vi::MOI.VariableIndex)
    return model.forward.primal_sensitivities[vi]
end

function MOI.get(model::Optimizer, ::MadDiff.ForwardConstraintDual, ci::MOI.ConstraintIndex)
    return model.forward.dual_sensitivities[ci]
end

function MOI.get(
    model::Optimizer,
    ::MadDiff.ForwardConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.VectorNonlinearOracle{Float64}},
)
    return model.forward.vector_dual_sensitivities[ci]
end

function MOI.get(model::Optimizer, ::MadDiff.ForwardObjectiveSensitivity)
    return model.forward.objective_sensitivity
end
