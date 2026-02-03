function _compute_qp_vjp_cons!(grad_p, f::MOI.ScalarAffineFunction, ctx, x, dy_row, dx, scale)
    for term in f.terms
        i = get_param_idx(ctx, term.variable)
        !iszero(i) && (grad_p[i] += term.coefficient * dy_row)
    end
    return
end
function _compute_qp_vjp_cons!(grad_p, f::MOI.ScalarQuadraticFunction, ctx, x, dy_row, dx, scale)
    p_vals = ctx.p_vals
    for term in f.affine_terms
        i = get_param_idx(ctx, term.variable)
        !iszero(i) && (grad_p[i] += term.coefficient * dy_row)
    end
    for term in f.quadratic_terms
        vi, vj = term.variable_1, term.variable_2
        pi, pj = get_param_idx(ctx, vi), get_param_idx(ctx, vj)
        xi, xj = get_primal_idx(ctx, vi), get_primal_idx(ctx, vj)
        coef = term.coefficient
        !iszero(pi) && !iszero(xj) && (grad_p[pi] += coef * x[xj] * dy_row)
        !iszero(pj) && !iszero(xi) && (grad_p[pj] += coef * x[xi] * dy_row)
        !iszero(pi) && !iszero(xj) && (grad_p[pi] += scale * coef * dx[xj])
        !iszero(pj) && !iszero(xi) && (grad_p[pj] += scale * coef * dx[xi])
        if !iszero(pi) && !iszero(pj)
            if pi == pj
                grad_p[pi] += coef * p_vals[pi] * dy_row
            else
                grad_p[pi] += coef * p_vals[pj] * dy_row
                grad_p[pj] += coef * p_vals[pi] * dy_row
            end
        end
    end
    return
end

_compute_qp_vjp_obj!(grad_p, f::MOI.ScalarAffineFunction, ctx, dx, scale) = nothing
function _compute_qp_vjp_obj!(grad_p, f::MOI.ScalarQuadraticFunction, ctx::SensitivityContext, dx, scale)
    for term in f.quadratic_terms
        vi, vj = term.variable_1, term.variable_2
        pi, pj = get_param_idx(ctx, vi), get_param_idx(ctx, vj)
        xi, xj = get_primal_idx(ctx, vi), get_primal_idx(ctx, vj)
        coef = term.coefficient
        !iszero(pi) && !iszero(xj) && (grad_p[pi] += scale * coef * dx[xj])
        !iszero(pj) && !iszero(xi) && (grad_p[pj] += scale * coef * dx[xi])
    end
    return
end

function _compute_nlp_vjp_objcons!(evaluator, ctx, grad_p, x_combined::AbstractVector{T}, y, dx, dy, n_x, σ) where {T}
    n_p = length(grad_p)
    n_nlp = length(dy)

    if n_nlp > 0
        hvp_result = ctx.hvp_result
        MOI.eval_constraint_jacobian_transpose_product(evaluator, hvp_result, x_combined, dy)
        @views grad_p .+= hvp_result[n_x+1:n_x+n_p]
    end

    v_extended = ctx.v_extended
    fill!(v_extended, zero(T))
    @views v_extended[1:n_x] .= dx

    hvp_result = ctx.hvp_result
    MOI.eval_hessian_lagrangian_product(evaluator, hvp_result, x_combined, v_extended, σ, y)
    @views grad_p .+= hvp_result[n_x+1:n_x+n_p]

    return
end

function _compute_param_pullback!(model, x::AbstractVector{T}, y, dx, dy, ctx::SensitivityContext, obj_scale) where {T}
    grad_p = ctx.grad_p
    fill!(grad_p, zero(T))

    σ = model.sense == MOI.MIN_SENSE ? one(T) : -one(T)

    p_vals = ctx.p_vals
    for (i, ref) in enumerate(ctx.param_nlp_refs)
        p_vals[i] = model.nlp_model[ref]
    end

    x_combined = ctx.x_combined
    for (i, vi) in enumerate(ctx.primal_vars)
        x_combined[i] = x[vi.value]
    end
    x_dense = @view x_combined[1:ctx.n_x]

    _compute_qp_vjp_obj!(grad_p, model.qp_data.objective, ctx, dx, σ * obj_scale)
    for (row, constraint) in enumerate(model.qp_data.constraints)
        _compute_qp_vjp_cons!(grad_p, constraint, ctx, x_dense, dy[row], dx, y[row])
    end

    has_nlp = (ctx.n_nlp > 0 || (!isnothing(model.nlp_model) && !isnothing(model.nlp_model.objective)))
    if has_nlp && !isnothing(model.nlp_model)
        if isnothing(ctx.nlp_evaluator)
            ctx.nlp_evaluator = create_sensitivity_evaluator(model.nlp_model, ctx.n_x, ctx.n_p)
        end

        for (i, ref) in enumerate(ctx.param_nlp_refs)
            x_combined[ctx.n_x + i] = model.nlp_model[ref]
        end

        μ = @view y[(ctx.n_qp + 1):(ctx.n_qp + ctx.n_nlp)]
        dμ = @view dy[(ctx.n_qp + 1):(ctx.n_qp + ctx.n_nlp)]
        _compute_nlp_vjp_objcons!(ctx.nlp_evaluator, ctx, grad_p, x_combined, μ, dx, dμ, ctx.n_x, σ * obj_scale)
    end

    grad_p .*= -one(T)  # outer negation in vjp

    return grad_p
end
