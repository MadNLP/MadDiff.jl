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
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, ctx, dL_dλ, dL_dzl, dL_dzu
) where {S <: MOI.GreaterThan}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = get(ctx.primal_idx, vi, 0)
    !iszero(idx) && (dL_dzl[idx] = val)
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, ctx, dL_dλ, dL_dzl, dL_dzu
) where {S <: MOI.LessThan}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = get(ctx.primal_idx, vi, 0)
    !iszero(idx) && (dL_dzu[idx] = -val)
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, ctx, dL_dλ, dL_dzl, dL_dzu
) where {S <: MOI.Interval}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = get(ctx.primal_idx, vi, 0)
    if !iszero(idx)
        dL_dzl[idx] = val
        dL_dzu[idx] = -val
    end
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, ctx, dL_dλ, dL_dzl, dL_dzu
) where {S <: MOI.EqualTo}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = get(ctx.primal_idx, vi, 0)
    if !iszero(idx)
        dL_dzl[idx] = val
        dL_dzu[idx] = -val
    end
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{F, S}, val, inner, ctx, dL_dλ, dL_dzl, dL_dzu
) where {F, S}
    row = _constraint_row(inner, ci)
    dL_dλ[row] = val
end

function _make_param_pullback_closure()
    return function(out, dx, dλ, dzl, dzu, sens)
        model = sens.ext
        x = model.inner.result.solution
        n_con = length(dλ)
        solver = model.inner.solver
        obj_sign = solver.cb.obj_sign
        VT = typeof(solver.y)
        if VT <: Vector
            y = _get_y_cache!(model, n_con)
            MadNLP.unpack_y!(y, solver.cb, solver.y)
            y .*= obj_sign
            dx_cpu = dx
            dλ_cpu = dλ .* obj_sign
        else
            y_gpu = similar(solver.y, n_con)
            MadNLP.unpack_y!(y_gpu, solver.cb, solver.y)
            y_gpu .*= obj_sign
            y = Array(y_gpu)
            dx_cpu = dx isa Vector ? dx : Array(dx)
            dλ_cpu = (dλ isa Vector ? dλ : Array(dλ)) .* obj_sign
        end

        grad_p = _compute_param_pullback!(model.inner, x, y, dx_cpu, dλ_cpu, model.sensitivity_context)
        copyto!(out, grad_p)
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

    n_x = NLPModels.get_nvar(sens.solver.nlp)
    dL_dx = _get_dL_dx!(model, n_x)
    for (vi, val) in model.reverse.primal_seeds
        dL_dx[ctx.primal_idx[vi]] = val
    end

    dL_dλ = _get_dL_dλ!(model, n_con)
    dL_dzl = _get_dL_dzl!(model, n_x)
    dL_dzu = _get_dL_dzu!(model, n_x)

    for (ci, val) in model.reverse.dual_seeds
        _process_reverse_dual_input!(ci, val, inner, ctx, dL_dλ, dL_dzl, dL_dzu)
    end

    dL_dλ .*= -solver.cb.obj_sign

    VT = typeof(solver.y)
    if VT <: Vector
        result = MadDiff.reverse_differentiate!(sens; dL_dx, dL_dλ, dL_dzl, dL_dzu)
        grad_p_cpu = result.grad_p
    else
        dL_dx_gpu = dL_dx isa VT ? dL_dx : VT(dL_dx)
        dL_dλ_gpu = dL_dλ isa VT ? dL_dλ : VT(dL_dλ)
        dL_dzl_gpu = dL_dzl isa VT ? dL_dzl : VT(dL_dzl)
        dL_dzu_gpu = dL_dzu isa VT ? dL_dzu : VT(dL_dzu)
        result = MadDiff.reverse_differentiate!(sens; dL_dx=dL_dx_gpu, dL_dλ=dL_dλ_gpu, dL_dzl=dL_dzl_gpu, dL_dzu=dL_dzu_gpu)
        grad_p_cpu = result.grad_p isa Vector ? result.grad_p : Array(result.grad_p)
    end

    for (ci, vi) in model.param_ci_to_vi
        model.reverse.param_outputs[ci] = grad_p_cpu[ctx.param_idx[vi]]
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
