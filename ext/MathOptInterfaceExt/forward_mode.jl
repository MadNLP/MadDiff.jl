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
    VT = typeof(solver.y)
    if VT <: Vector
        y = _get_y_cache!(model, n_con)
        MadNLP.unpack_y!(y, solver.cb, solver.y)
    else
        y_gpu = similar(solver.y, n_con)
        MadNLP.unpack_y!(y_gpu, solver.cb, solver.y)
        y = Array(y_gpu)
    end
    Dxp_L_Δp, Dp_g_Δp = compute_param_jvp!(inner, x, y, Δp, ctx)

    sens = _get_sensitivity_solver!(model)

    VT = typeof(solver.y)
    if VT <: Vector
        result = MadDiff.forward_differentiate!(sens; Dxp_L=Dxp_L_Δp, Dp_g=Dp_g_Δp)
        dx_cpu = result.dx
        dλ_cpu = result.dλ
    else
        Dxp_L_gpu = VT(Dxp_L_Δp)
        Dp_g_gpu = VT(Dp_g_Δp)
        result = MadDiff.forward_differentiate!(sens; Dxp_L=Dxp_L_gpu, Dp_g=Dp_g_gpu)
        dx_cpu = Array(result.dx)
        dλ_cpu = Array(result.dλ)
    end

    for (i, vi) in enumerate(ctx.primal_vars)
        model.forward.primal_sensitivities[vi] = dx_cpu[i]
    end

    dλ = _get_dλ_cache!(model, n_con)
    # MOI and MadNLP have opposite sign conventions for constraint duals
    dλ .= .-dλ_cpu

    _store_dual_sensitivities!(model.forward.dual_sensitivities, inner, dλ)
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

function _store_dual_sensitivities!(dual_sensitivities, inner, dλ)
    for (F, S) in MOI.get(inner, MOI.ListOfConstraintTypesPresent())
        F == MOI.VariableIndex && continue
        S <: MOI.Parameter && continue
        for ci in MOI.get(inner, MOI.ListOfConstraintIndices{F, S}())
            row = _constraint_row(inner, ci)
            dual_sensitivities[ci] = dλ[row]
        end
    end
    if inner.nlp_model !== nothing
        n_qp = length(inner.qp_data.constraints)
        for (nlp_idx, con) in inner.nlp_model.constraints
            S = typeof(con.set)
            ci = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction, S}(nlp_idx.value)
            row = n_qp + nlp_idx.value
            dual_sensitivities[ci] = dλ[row]
        end
    end
    return
end

function _store_bound_dual_sensitivities!(model, sens, result, ctx)
    dsens = model.forward.dual_sensitivities
    vi_to_lb_idx = model.vi_to_lb_idx
    vi_to_ub_idx = model.vi_to_ub_idx

    dzl = result.dzl isa Vector ? result.dzl : Array(result.dzl)
    dzu = result.dzu isa Vector ? result.dzu : Array(result.dzu)

    for (vi, ci) in ctx.vi_to_lb_ci
        i = get(vi_to_lb_idx, vi, 0)
        !iszero(i) && (dsens[ci] = dzl[i])
    end
    for (vi, ci) in ctx.vi_to_ub_ci
        i = get(vi_to_ub_idx, vi, 0)
        !iszero(i) && (dsens[ci] = -dzu[i])
    end

    for (vi, ci) in ctx.vi_to_interval_ci
        dzl_idx = get(vi_to_lb_idx, vi, 0)
        dzu_idx = get(vi_to_ub_idx, vi, 0)
        if !iszero(dzl_idx) && !iszero(dzu_idx)
            dsens[ci] = dzl[dzl_idx] - dzu[dzu_idx]
        elseif !iszero(dzl_idx)
            dsens[ci] = dzl[dzl_idx]
        elseif !iszero(dzu_idx)
            dsens[ci] = -dzu[dzu_idx]
        end
    end

    for (vi, ci) in ctx.vi_to_eq_ci
        dzl_idx = get(vi_to_lb_idx, vi, 0)
        dzu_idx = get(vi_to_ub_idx, vi, 0)
        if !iszero(dzl_idx) && !iszero(dzu_idx)
            dsens[ci] = dzl[dzl_idx] - dzu[dzu_idx]
        elseif !iszero(dzl_idx)
            dsens[ci] = dzl[dzl_idx]
        elseif !iszero(dzu_idx)
            dsens[ci] = -dzu[dzu_idx]
        end
    end

    return
end

function MOI.get(model::Optimizer, ::MadDiff.ForwardVariablePrimal, vi::MOI.VariableIndex)
    return model.forward.primal_sensitivities[vi]
end

function MOI.get(model::Optimizer, ::MadDiff.ForwardConstraintDual, ci::MOI.ConstraintIndex)
    return model.forward.dual_sensitivities[ci]
end
