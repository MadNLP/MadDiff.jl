function adjoint_solve_kkt_system! end

function adjoint_mul! end

function _adjoint_kktmul!(
    w::AbstractKKTVector,
    x::AbstractKKTVector,
    reg,
    du_diag,
    l_lower,
    u_lower,
    l_diag,
    u_diag,
    alpha,
    beta,
)
    primal(w) .+= alpha .* reg .* primal(x)
    dual(w) .+= alpha .* du_diag .* dual(x)
    w.xp_lr .+= alpha .* (l_lower .* dual_lb(x))
    w.xp_ur .+= alpha .* (u_lower .* dual_ub(x))
    dual_lb(w) .= beta .* dual_lb(w) .+ alpha .* (.-x.xp_lr .- l_diag .* dual_lb(x))
    dual_ub(w) .= beta .* dual_ub(w) .+ alpha .* ( x.xp_ur .+ u_diag .* dual_ub(x))
    return
end

function _adjoint_scaled_kktmul!(
    w::AbstractKKTVector,
    x::AbstractKKTVector,
    reg,
    du_diag,
    l_lower,
    u_lower,
    l_diag,
    u_diag,
    alpha,
    beta,
)
    primal(w) .+= alpha .* reg .* primal(x)
    dual(w) .+= alpha .* du_diag .* dual(x)
    w.xp_lr .+= alpha .* (l_lower .* dual_lb(x))
    w.xp_ur .+= alpha .* (u_lower .* dual_ub(x))
    dual_lb(w) .= beta .* dual_lb(w) .+ alpha .* (.-x.xp_lr .+ l_diag .* dual_lb(x))
    dual_ub(w) .= beta .* dual_ub(w) .+ alpha .* ( x.xp_ur .- u_diag .* dual_ub(x))
    return
end

function adjoint_solve_refine_wrapper!(d, solver, p, w)
    result = false

    solver.cnt.linear_solver_time += @elapsed begin
        if adjoint_solve_refine!(d, solver.iterator, p, w)
            result = true
        else
            if improve!(solver.kkt.linear_solver)
                if adjoint_solve_refine!(d, solver.iterator, p, w)
                    result = true
                end
            end
        end
    end

    return result
end

function adjoint_solve_refine!(
    x::VT,
    iterator::R,
    b::VT,
    w::VT,
    ) where {T, VT, R <: RichardsonIterator{T}}
    @debug(iterator.logger, "Adjoint iterative solver initiated")

    norm_b = norm(full(b), Inf)
    residual_ratio = zero(T)

    fill!(full(x), zero(T))

    if norm_b != zero(T)
        copyto!(full(w), full(b))
        iterator.cnt.ir = 0

        while true
            adjoint_solve_kkt_system!(iterator.kkt, w)
            axpy!(1., full(w), full(x))
            copyto!(full(w), full(b))

            adjoint_mul!(w, iterator.kkt, x, -one(T), one(T))

            norm_w = norm(full(w), Inf)
            norm_x = norm(full(x), Inf)
            residual_ratio = norm_w / (min(norm_x, 1e6 * norm_b) + norm_b)

            if mod(iterator.cnt.ir, 10)==0
                @debug(iterator.logger,"iterator.cnt.ir ||res||")
            end
            @debug(iterator.logger, @sprintf("%4i %6.2e", iterator.cnt.ir, residual_ratio))
            iterator.cnt.ir += 1

            if (iterator.cnt.ir >= iterator.opt.richardson_max_iter) || (residual_ratio < iterator.opt.richardson_tol)
                break
            end
        end
    end

    @debug(
        iterator.logger,
        @sprintf(
            "Adjoint iterative solver terminated with %4i refinement steps and residual = %6.2e",
            iterator.cnt.ir, residual_ratio
        ),
    )

    return residual_ratio < iterator.opt.richardson_acceptable_tol
end