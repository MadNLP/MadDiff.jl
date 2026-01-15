function compute_qp_vjp_cons!(∂L_∂p, f::MOI.ScalarAffineFunction, ctx, x, adj_λ_row, adj_x, scale)
    for term in f.terms
        i = get_param_idx(ctx, term.variable)
        !iszero(i) && (∂L_∂p[i] += term.coefficient * adj_λ_row)
    end
    return
end
function compute_qp_vjp_cons!(∂L_∂p, f::MOI.ScalarQuadraticFunction, ctx, x, adj_λ_row, adj_x, scale)
    p_vals = ctx.p_vals
    for term in f.affine_terms
        i = get_param_idx(ctx, term.variable)
        !iszero(i) && (∂L_∂p[i] += term.coefficient * adj_λ_row)
    end
    for term in f.quadratic_terms
        vi, vj = term.variable_1, term.variable_2
        pi, pj = get_param_idx(ctx, vi), get_param_idx(ctx, vj)
        xi, xj = get_primal_idx(ctx, vi), get_primal_idx(ctx, vj)
        coef = term.coefficient
        !iszero(pi) && !iszero(xj) && (∂L_∂p[pi] += coef * x[xj] * adj_λ_row)
        !iszero(pj) && !iszero(xi) && (∂L_∂p[pj] += coef * x[xi] * adj_λ_row)
        !iszero(pi) && !iszero(xj) && (∂L_∂p[pi] += scale * coef * adj_x[xj])
        !iszero(pj) && !iszero(xi) && (∂L_∂p[pj] += scale * coef * adj_x[xi])
        if !iszero(pi) && !iszero(pj)
            if pi == pj
                ∂L_∂p[pi] += coef * p_vals[pi] * adj_λ_row
            else
                ∂L_∂p[pi] += coef * p_vals[pj] * adj_λ_row
                ∂L_∂p[pj] += coef * p_vals[pi] * adj_λ_row
            end
        end
    end
    return
end

compute_qp_vjp_obj!(∂L_∂p, f::MOI.ScalarAffineFunction, ctx, adj_x, scale) = nothing
function compute_qp_vjp_obj!(∂L_∂p, f::MOI.ScalarQuadraticFunction, ctx::SensitivityContext, adj_x, scale)
    for term in f.quadratic_terms
        vi, vj = term.variable_1, term.variable_2
        pi, pj = get_param_idx(ctx, vi), get_param_idx(ctx, vj)
        xi, xj = get_primal_idx(ctx, vi), get_primal_idx(ctx, vj)
        coef = term.coefficient
        !iszero(pi) && !iszero(xj) && (∂L_∂p[pi] += scale * coef * adj_x[xj])
        !iszero(pj) && !iszero(xi) && (∂L_∂p[pj] += scale * coef * adj_x[xi])
    end
    return
end

function compute_nlp_vjp_objcons!(evaluator, ctx, ∂L_∂p, x_combined::AbstractVector{T}, y, adj_x, adj_λ, n_x, σ) where {T}
    n_p = length(∂L_∂p)
    n_nlp = length(adj_λ)

    if n_nlp > 0
        hvp_result = ctx.hvp_result
        MOI.eval_constraint_jacobian_transpose_product(evaluator, hvp_result, x_combined, adj_λ)
        @views ∂L_∂p .+= hvp_result[n_x+1:n_x+n_p]
    end

    v_extended = ctx.v_extended
    fill!(v_extended, zero(T))
    @views v_extended[1:n_x] .= adj_x

    hvp_result = ctx.hvp_result
    MOI.eval_hessian_lagrangian_product(evaluator, hvp_result, x_combined, v_extended, σ, y)
    @views ∂L_∂p .+= hvp_result[n_x+1:n_x+n_p]

    return
end

function _compute_param_pullback!(model, x::AbstractVector{T}, y, adj_x, adj_λ, ctx::SensitivityContext) where {T}
    ∂L_∂p = ctx.∂L_∂p
    fill!(∂L_∂p, zero(T))

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

    compute_qp_vjp_obj!(∂L_∂p, model.qp_data.objective, ctx, adj_x, σ)
    for (row, constraint) in enumerate(model.qp_data.constraints)
        compute_qp_vjp_cons!(∂L_∂p, constraint, ctx, x_dense, adj_λ[row], adj_x, y[row])
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
        adj_μ = @view adj_λ[(ctx.n_qp + 1):(ctx.n_qp + ctx.n_nlp)]
        compute_nlp_vjp_objcons!(ctx.nlp_evaluator, ctx, ∂L_∂p, x_combined, μ, adj_x, adj_μ, ctx.n_x, σ)
    end

    ∂L_∂p .*= -one(T)  # flip to madnlp convention

    return ∂L_∂p
end
