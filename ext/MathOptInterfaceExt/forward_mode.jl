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

    ctx = _get_sensitivity_context!(model)
    n_con = NLPModels.get_ncon(solver.nlp)

    Δp = ctx.Δp
    fill!(Δp, zero(T))
    for (ci, dp) in model.forward.param_perturbations
        vi = model.param_ci_to_vi[ci]
        Δp[ctx.param_idx[vi]] = dp
    end

    x = inner.result.solution
    obj_sign = solver.cb.obj_sign
    VT = typeof(solver.y)
    if VT <: Vector
        y = _get_y_cache!(model, n_con)
        MadNLP.unpack_y!(y, solver.cb, solver.y)
        y .*= obj_sign
    else
        y_gpu = similar(solver.y, n_con)
        MadNLP.unpack_y!(y_gpu, solver.cb, solver.y)
        y_gpu .*= obj_sign
        y = Array(y_gpu)
    end
    d2L_dxdp, dg_dp = _compute_param_jvp!(inner, x, y, Δp, ctx)

    sens = _get_sensitivity_solver!(model)

    VT = typeof(solver.y)
    if VT <: Vector
        result = MadDiff.forward_differentiate!(sens; d2L_dxdp, dg_dp)
        dx_cpu = result.dx
        dy_cpu = result.dy
    else
        d2L_dxdp_gpu = d2L_dxdp isa VT ? d2L_dxdp : VT(d2L_dxdp)
        dg_dp_gpu = dg_dp isa VT ? dg_dp : VT(dg_dp)
        result = MadDiff.forward_differentiate!(sens; d2L_dxdp=d2L_dxdp_gpu, dg_dp=dg_dp_gpu)
        dx_cpu = result.dx isa Vector ? result.dx : Array(result.dx)
        dy_cpu = result.dy isa Vector ? result.dy : Array(result.dy)
    end

    for (i, vi) in enumerate(ctx.primal_vars)
        model.forward.primal_sensitivities[vi] = dx_cpu[i]
    end

    dy = _get_dy_cache!(model, n_con)
    dy .= (.-obj_sign) .* dy_cpu

    _store_dual_sensitivities!(model.forward.dual_sensitivities, inner, dy)
    _store_bound_dual_sensitivities!(model, sens, result, ctx)
    return
end

function _constraint_row(inner, ci::MOI.ConstraintIndex{F, S}) where {F, S}
    if F == MOI.ScalarNonlinearFunction
        return length(inner.qp_data.constraints) + ci.value
    else
        return ci.value  # TODO: check aff + quad + nlp
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

function _store_bound_dual_sensitivities!(model, sens, result, ctx)
    dsens = model.forward.dual_sensitivities

    dzl = result.dzl isa Vector ? result.dzl : Array(result.dzl)
    dzu = result.dzu isa Vector ? result.dzu : Array(result.dzu)

    for (vi, ci) in ctx.vi_to_lb_ci
        idx = ctx.primal_idx[vi]
        dsens[ci] = dzl[idx]
    end
    for (vi, ci) in ctx.vi_to_ub_ci
        idx = ctx.primal_idx[vi]
        dsens[ci] = -dzu[idx]
    end

    for (vi, ci) in ctx.vi_to_interval_ci
        idx = ctx.primal_idx[vi]
        dsens[ci] = dzl[idx] - dzu[idx]
    end

    for (vi, ci) in ctx.vi_to_eq_ci
        idx = ctx.primal_idx[vi]
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
