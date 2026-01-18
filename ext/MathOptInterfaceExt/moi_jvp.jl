function _compute_qp_jvp_cons!(d2L_dxdp, dg_dp, f::MOI.ScalarAffineFunction, ctx, x, Δp, row, scale)
    for term in f.terms
        i = get_param_idx(ctx, term.variable)
        !iszero(i) && (dg_dp[row] += term.coefficient * Δp[i])
    end
    return
end
function _compute_qp_jvp_cons!(d2L_dxdp, dg_dp, f::MOI.ScalarQuadraticFunction, ctx, x, Δp, row, scale)
    p_vals = ctx.p_vals
    for term in f.affine_terms
        i = get_param_idx(ctx, term.variable)
        !iszero(i) && (dg_dp[row] += term.coefficient * Δp[i])
    end
    for term in f.quadratic_terms
        vi, vj = term.variable_1, term.variable_2
        pi, pj = get_param_idx(ctx, vi), get_param_idx(ctx, vj)
        xi, xj = get_primal_idx(ctx, vi), get_primal_idx(ctx, vj)
        coef = term.coefficient
        !iszero(pi) && !iszero(xj) && (dg_dp[row] += coef * x[xj] * Δp[pi])
        !iszero(pj) && !iszero(xi) && (dg_dp[row] += coef * x[xi] * Δp[pj])
        !iszero(pi) && !iszero(xj) && (d2L_dxdp[xj] += scale * coef * Δp[pi])
        !iszero(pj) && !iszero(xi) && (d2L_dxdp[xi] += scale * coef * Δp[pj])
        if !iszero(pi) && !iszero(pj)
            if pi == pj
                dg_dp[row] += coef * p_vals[pi] * Δp[pi]
            else
                dg_dp[row] += coef * p_vals[pj] * Δp[pi]
                dg_dp[row] += coef * p_vals[pi] * Δp[pj]
            end
        end
    end
    return
end

_compute_qp_jvp_obj!(d2L_dxdp, f::MOI.ScalarAffineFunction, ctx, Δp, scale) = nothing
function _compute_qp_jvp_obj!(d2L_dxdp, f::MOI.ScalarQuadraticFunction, ctx, Δp, scale)
    for term in f.quadratic_terms
        vi, vj = term.variable_1, term.variable_2
        pi, pj = get_param_idx(ctx, vi), get_param_idx(ctx, vj)
        xi, xj = get_primal_idx(ctx, vi), get_primal_idx(ctx, vj)
        coef = term.coefficient
        !iszero(pi) && !iszero(xj) && (d2L_dxdp[xj] += scale * coef * Δp[pi])
        !iszero(pj) && !iszero(xi) && (d2L_dxdp[xi] += scale * coef * Δp[pj])
    end
    return
end

function _compute_nlp_jvp_objcons!(evaluator, ctx, d2L_dxdp::AbstractVector{T}, dg_dp, x_combined, y, Δp, n_x, σ) where {T}
    n_p = length(Δp)
    n_nlp = length(dg_dp)

    v_extended = ctx.v_extended
    fill!(v_extended, zero(T))
    @views v_extended[n_x+1:n_x+n_p] .= Δp

    if n_nlp > 0
        jvp_result = ctx.jvp_result
        MOI.eval_constraint_jacobian_product(evaluator, jvp_result, x_combined, v_extended)
        dg_dp .+= jvp_result
    end

    hvp_result = ctx.hvp_result
    MOI.eval_hessian_lagrangian_product(evaluator, hvp_result, x_combined, v_extended, σ, y)
    @views d2L_dxdp .+= hvp_result[1:n_x]

    return
end

function _compute_param_jvp!(model, x::AbstractVector{T}, y, Δp, ctx) where {T}
    d2L_dxdp = ctx.d2L_dxdp
    dg_dp = ctx.dg_dp
    fill!(d2L_dxdp, zero(T))
    fill!(dg_dp, zero(T))

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

    _compute_qp_jvp_obj!(d2L_dxdp, model.qp_data.objective, ctx, Δp, σ)
    for (row, constraint) in enumerate(model.qp_data.constraints)
        _compute_qp_jvp_cons!(d2L_dxdp, dg_dp, constraint, ctx, x_dense, Δp, row, y[row])
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
        _compute_nlp_jvp_objcons!(ctx.nlp_evaluator, ctx, d2L_dxdp, view(dg_dp, (ctx.n_qp + 1):(ctx.n_qp + ctx.n_nlp)), x_combined, μ, Δp, ctx.n_x, σ)
    end

    return d2L_dxdp, dg_dp
end
