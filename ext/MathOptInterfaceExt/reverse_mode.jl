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
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, ctx, seed_λ, seed_zl, seed_zu
) where {S <: MOI.GreaterThan}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = get(ctx.primal_idx, vi, 0)
    !iszero(idx) && (seed_zl[idx] = val)
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, ctx, seed_λ, seed_zl, seed_zu
) where {S <: MOI.LessThan}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = get(ctx.primal_idx, vi, 0)
    !iszero(idx) && (seed_zu[idx] = -val)
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, ctx, seed_λ, seed_zl, seed_zu
) where {S <: MOI.Interval}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = get(ctx.primal_idx, vi, 0)
    if !iszero(idx)
        seed_zl[idx] = val
        seed_zu[idx] = -val
    end
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S}, val, inner, ctx, seed_λ, seed_zl, seed_zu
) where {S <: MOI.EqualTo}
    vi = MOI.get(inner, MOI.ConstraintFunction(), ci)
    idx = get(ctx.primal_idx, vi, 0)
    if !iszero(idx)
        seed_zl[idx] = val
        seed_zu[idx] = -val
    end
end

function _process_reverse_dual_input!(
    ci::MOI.ConstraintIndex{F, S}, val, inner, ctx, seed_λ, seed_zl, seed_zu
) where {F, S}
    row = _constraint_row(inner, ci)
    seed_λ[row] = val
end

function _make_param_pullback_closure(model, ctx::SensitivityContext{T}) where {T}
    return function(out, dx, dλ, dzl, dzu, sens)
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

        grad_p = _compute_param_pullback!(model.inner, x, y, dx_cpu, dλ_cpu, ctx)
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

    n_x = sens.dims.n_x
    seed_x = _get_seed_x!(model, n_x)
    for (vi, val) in model.reverse.primal_seeds
        seed_x[ctx.primal_idx[vi]] = val
    end

    seed_λ = _get_seed_λ!(model, n_con)
    seed_zl = _get_seed_zl!(model, n_x)
    seed_zu = _get_seed_zu!(model, n_x)

    for (ci, val) in model.reverse.dual_seeds
        _process_reverse_dual_input!(ci, val, inner, ctx, seed_λ, seed_zl, seed_zu)
    end

    seed_λ .*= -solver.cb.obj_sign

    VT = typeof(solver.y)
    if VT <: Vector
        result = MadDiff.reverse_differentiate!(sens; seed_x, seed_λ, seed_zl, seed_zu)
        grad_p_cpu = result.grad_p
    else
        seed_x_gpu = seed_x isa VT ? seed_x : VT(seed_x)
        seed_λ_gpu = seed_λ isa VT ? seed_λ : VT(seed_λ)
        seed_zl_gpu = seed_zl isa VT ? seed_zl : VT(seed_zl)
        seed_zu_gpu = seed_zu isa VT ? seed_zu : VT(seed_zu)
        result = MadDiff.reverse_differentiate!(sens; seed_x=seed_x_gpu, seed_λ=seed_λ_gpu, seed_zl=seed_zl_gpu, seed_zu=seed_zu_gpu)
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
