function adjoint_solve! end

function adjoint_mul! end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::SparseKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT, MT, QN}
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::SparseUnreducedKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT, MT, QN}
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::ScaledSparseKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT, MT, QN}
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    _adjoint_scaled_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::SparseCondensedKKTSystem{T,VT,MT,QN},
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT, MT, QN}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+m)
    xz = view(full(x), n+m+1:n+2*m)

    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2*m)

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)
    mul!(wx, kkt.jt_csc,  xz, alpha, one(T))
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    wz .-= alpha .* xs
    ws .= beta .* ws .- alpha .* xz

    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function adjoint_mul!(
    w::AbstractKKTVector{T},
    kkt::AbstractDenseKKTSystem,
    x::AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T}
    (m, n) = size(kkt.jac)
    wx = @view(primal(w)[1:n])
    ws = @view(primal(w)[n+1:end])
    wy = dual(w)
    wz = @view(dual(w)[kkt.ind_ineq])

    xx = @view(primal(x)[1:n])
    xs = @view(primal(x)[n+1:end])
    xy = dual(x)
    xz = @view(dual(x)[kkt.ind_ineq])

    _symv!('L', alpha, kkt.hess, xx, beta, wx)
    if m > 0
        mul!(wx, kkt.jac', xy, alpha, one(T))
        mul!(wy, kkt.jac,  xx, alpha, beta)
    end
    ws .= beta .* ws .- alpha .* xz
    wz .-= alpha .* xs
    _adjoint_kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

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
            adjoint_solve!(iterator.kkt, w)
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

function _adjoint_unreduced_post!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    dual_lb(w) .*= .-kkt.l_lower_aug
    dual_ub(w) .*= kkt.u_lower_aug
    return
end

function _adjoint_unreduced_pre!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    f(x,y) = iszero(y) ? x : x/y
    wzl = dual_lb(w)
    wzu = dual_ub(w)
    wzl .= f.(wzl, kkt.l_lower_aug)
    wzu .= f.(wzu, kkt.u_lower_aug)
    return
end

# Adjoint solve for the unreduced KKT system (SparseUnreducedKKTSystem).
#
# Forward solve:
#   rₗ = Dₗ⁻¹ rₗ, rᵤ = Dᵤ⁻¹ rᵤ   (with Dₗ = diag(l_lower_aug), Dᵤ = diag(u_lower_aug))
#   K x = r
#   xₗ = -Dₗ xₗ,  xᵤ = Dᵤ xᵤ
#
# Adjoint (reverse) updates:
#   gₗ = -Dₗ gₗ,  gᵤ = Dᵤ gᵤ
#   K y = g
#   gₗ = Dₗ⁻¹ gₗ, gᵤ = Dᵤ⁻¹ gᵤ
function adjoint_solve!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    _adjoint_unreduced_post!(kkt, w)
    solve!(kkt.linear_solver, full(w))
    _adjoint_unreduced_pre!(kkt, w)
    return w
end

function _adjoint_scaled_post!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    w.xp_lr .-= (kkt.l_lower ./ kkt.l_diag) .* dlb
    w.xp_ur .+= (kkt.u_lower ./ kkt.u_diag) .* dub
    dlb .= dlb ./ kkt.l_diag
    dub .= .-dub ./ kkt.u_diag
    return
end

function _adjoint_scaled_pre!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    r3 = kkt.buffer1
    r4 = kkt.buffer2
    r3 .= w.xp
    r4 .= w.xp
    w.xp .*= kkt.scaling_factor
    dual_lb(w) .+= r3[kkt.ind_lb] ./ sqrt.(kkt.l_diag)
    dual_ub(w) .+= r4[kkt.ind_ub] ./ sqrt.(kkt.u_diag)
    return
end

# Adjoint solve for the scaled reduced KKT system (ScaledSparseKKTSystem).
#
# Forward solve:
#   r̂ₚ = S rₚ + Dₗ⁻½ rₗ + Dᵤ⁻½ rᵤ
#   K_scaled z = [r̂ₚ; r_d]
#   xₚ = S zₚ
#   rₗ = Dₗ⁻¹(rₗ - L xₗ), rᵤ = Dᵤ⁻¹(-rᵤ + U xᵤ)
# with S = diag(scaling_factor), Dₗ = diag(l_diag), Dᵤ = diag(u_diag),
# L = diag(l_lower), U = diag(u_lower).
#
# Adjoint (reverse) updates:
#   xₗ += -L Dₗ⁻¹ gₗ, xᵤ += U Dᵤ⁻¹ gᵤ
#   gₗ = Dₗ⁻¹ gₗ,     gᵤ = -Dᵤ⁻¹ gᵤ
#   zₚ += S gₚ
#   gₗ += Dₗ⁻½ zₚ,    gᵤ += Dᵤ⁻½ zₚ
function adjoint_solve!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    _adjoint_scaled_post!(kkt, w)
    w.xp .*= kkt.scaling_factor
    solve!(kkt.linear_solver, primal_dual(w))
    _adjoint_scaled_pre!(kkt, w)
    return w
end

function _adjoint_finish_bounds!(kkt::SparseKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    w.xp_lr .+= (kkt.l_lower ./ kkt.l_diag) .* dlb
    dlb .= .-dlb ./ kkt.l_diag
    w.xp_ur .-= (kkt.u_lower ./ kkt.u_diag) .* dub
    dub .= dub ./ kkt.u_diag
    return
end

function _adjoint_reduce_rhs!(kkt::SparseKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    dlb .-= w.xp_lr ./ kkt.l_diag
    dub .-= w.xp_ur ./ kkt.u_diag
    return
end

function _adjoint_finish_bounds!(kkt::SparseCondensedKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    w.xp_lr .+= (kkt.l_lower ./ kkt.l_diag) .* dlb
    dlb .= .-dlb ./ kkt.l_diag
    w.xp_ur .-= (kkt.u_lower ./ kkt.u_diag) .* dub
    dub .= dub ./ kkt.u_diag
    return
end

function _adjoint_reduce_rhs!(kkt::SparseCondensedKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    dlb .-= w.xp_lr ./ kkt.l_diag
    dub .-= w.xp_ur ./ kkt.u_diag
    return
end

function _adjoint_finish_bounds!(kkt::DenseKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    w.xp_lr .+= (kkt.l_lower ./ kkt.l_diag) .* dlb
    dlb .= .-dlb ./ kkt.l_diag
    w.xp_ur .-= (kkt.u_lower ./ kkt.u_diag) .* dub
    dub .= dub ./ kkt.u_diag
    return
end

function _adjoint_reduce_rhs!(kkt::DenseKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    dlb .-= w.xp_lr ./ kkt.l_diag
    dub .-= w.xp_ur ./ kkt.u_diag
    return
end

function _adjoint_finish_bounds!(kkt::DenseCondensedKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    w.xp_lr .+= (kkt.l_lower ./ kkt.l_diag) .* dlb
    dlb .= .-dlb ./ kkt.l_diag
    w.xp_ur .-= (kkt.u_lower ./ kkt.u_diag) .* dub
    dub .= dub ./ kkt.u_diag
    return
end

function _adjoint_reduce_rhs!(kkt::DenseCondensedKKTSystem, w::AbstractKKTVector)
    dlb = dual_lb(w)
    dub = dual_ub(w)
    dlb .-= w.xp_lr ./ kkt.l_diag
    dub .-= w.xp_ur ./ kkt.u_diag
    return
end

# Adjoint solve for the reduced KKT system (SparseKKTSystem).
#
# Forward solve (with bound elimination):
#   r̂ₚ = rₚ - Dₗ⁻¹ rₗ - Dᵤ⁻¹ rᵤ
#   K_red x = r̂
#   yₗ = Dₗ⁻¹(-rₗ + L xₗ),  yᵤ = Dᵤ⁻¹(rᵤ - U xᵤ)
# with Dₗ = diag(l_diag), Dᵤ = diag(u_diag), L = diag(l_lower), U = diag(u_lower).
#
# Adjoint (reverse) updates:
#   xₗ += L Dₗ⁻¹ gₗ,  xᵤ += -U Dᵤ⁻¹ gᵤ
#   gₗ = -Dₗ⁻¹ gₗ,   gᵤ =  Dᵤ⁻¹ gᵤ
#   K_red y = [gₚ; g_d]
#   gₗ += -Dₗ⁻¹ yₗ,  gᵤ += -Dᵤ⁻¹ yᵤ
function adjoint_solve!(kkt::SparseKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    solve!(kkt.linear_solver, primal_dual(w))
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_solve!(
    kkt::SparseKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}

    qn = kkt.quasi_newton
    n, p = size(qn)
    # Load buffers
    xr = qn._w2
    Tk = qn.Tk
    w_ = primal_dual(w)
    nn = length(w_)

    _adjoint_finish_bounds!(kkt, w)

    fill!(Tk, zero(T))

    # Resize arrays with correct dimension
    if size(qn.E) != (nn, 2*p)
        qn.E = zeros(T, nn, 2*p)
        qn.H = zeros(T, nn, 2*p)
    else
        fill!(qn.E, zero(T))
        fill!(qn.H, zero(T))
    end

    # Solve LBFGS system with Sherman-Morrison-Woodbury formula
    # (C + E P Eᵀ)⁻¹ = C⁻¹ - C⁻¹ E (P + Eᵀ C⁻¹ E) Eᵀ C⁻¹
    #
    # P = [ -Iₚ  0  ] (size 2p × 2p) and E = [ U  V ] (size (n+m) × 2p)
    #     [  0   Iₚ ]                        [ 0  0 ]

    # Solve linear system without low-rank part
    solve!(kkt.linear_solver, w_)  # w_ stores the solution of Cx = b

    # Add low-rank correction
    if p > 0
        @inbounds for i in 1:n, j in 1:p
            qn.E[i, j] = qn.U[i, j]
            qn.E[i, j+p] = qn.V[i, j]
        end
        copyto!(qn.H, qn.E)

        multi_solve!(kkt.linear_solver, qn.H)  # H = C⁻¹ E

        for i = 1:p
            Tk[i,i] = -one(T)                  # Tₖ = P
            Tk[i+p,i+p] = one(T)
        end
        mul!(Tk, qn.E', qn.H, one(T), one(T))  # Tₖ = (P + Eᵀ C⁻¹ E)

        F, ipiv, info = LAPACK.sytrf!('L', Tk) # Tₖ⁻¹

        mul!(xr, qn.E', w_)                    # xᵣ = Eᵀ C⁻¹ b
        LAPACK.sytrs!('L', F, ipiv, xr)        # xᵣ = (P + Eᵀ C⁻¹ E)⁻¹ Eᵀ C⁻¹ b
        mul!(w_, qn.H, xr, -one(T), one(T))    # x = x - C⁻¹ E xᵣ
    end

    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function _adjoint_condensed_solve!(kkt::SparseCondensedKKTSystem{T}, w::AbstractKKTVector) where T
    (n,m) = size(kkt.jt_csc)

    # Decompose buffers
    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2*m)
    Σs = view(kkt.pr_diag, n+1:n+m)

    buf = kkt.buffer
    buf2 = kkt.buffer2

    # Reverse ws = (ws + wz) ./ Σs
    wz .+= ws ./ Σs
    ws ./= Σs

    # Reverse wz = -buf + diag_buffer .* buf2
    buf .= .-wz
    buf2 .= kkt.diag_buffer .* wz

    # Reverse buf2 = jt_csc' * wx
    mul!(wx, kkt.jt_csc, buf2, one(T), one(T))

    # Reverse wx = A⁻¹(wx)
    solve!(kkt.linear_solver, wx)

    # Reverse wx = wx + jt_csc * buf
    mul!(buf, kkt.jt_csc', wx, one(T), one(T))

    # Reverse buf = diag_buffer .* (wz + ws ./ Σs)
    buf .= kkt.diag_buffer .* buf
    wz .= buf
    ws .+= buf ./ Σs
    return
end

# Adjoint solve for the condensed KKT system (SparseCondensedKKTSystem).
#
# Forward solve:
#   r̂ₚ = rₚ + Jᵀ (D (r_z + Σₛ⁻¹ r_s))
#   K_cond x = r̂ₚ
#   r_z = -D (r_z + Σₛ⁻¹ r_s) + D J x
#   r_s = Σₛ⁻¹ (r_s + r_z)
#
# Adjoint (reverse) updates:
#   g_z += Σₛ⁻¹ g_s, g_s = Σₛ⁻¹ g_s
#   g_buf = -g_z, g_buf2 = D g_z
#   g_x += J g_buf2, g_x = K_cond⁻¹ g_x
#   g_buf += Jᵀ g_x
#   g_z += D g_buf, g_s += Σₛ⁻¹ D g_buf
function adjoint_solve!(kkt::SparseCondensedKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_condensed_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_solve!(kkt::DenseKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    solve!(kkt.linear_solver, primal_dual(w))
    _adjoint_reduce_rhs!(kkt, w)
    return w
end

function adjoint_solve!(
    kkt::SparseUnreducedKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
    error("Quasi-Newton approximation of the Hessian is not supported by the KKT formulation SparseUnreducedKKTSystem. Please use SparseKKTSystem instead.")
end

function adjoint_solve!(
    kkt::SparseCondensedKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
    error("Quasi-Newton approximation of the Hessian is not supported by the KKT formulation SparseCondensedKKTSystem. Please use SparseKKTSystem instead.")
end

function adjoint_solve!(
    kkt::ScaledSparseKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
    error("Quasi-Newton approximation of the Hessian is not supported by the KKT formulation ScaledSparseKKTSystem. Please use SparseKKTSystem instead.")
end

function _adjoint_dense_condensed_solve!(kkt::DenseCondensedKKTSystem{T}, w::AbstractKKTVector) where T
    n = num_variables(kkt)
    n_eq, ns = kkt.n_eq, kkt.n_ineq

    # Decompose rhs
    wx = view(full(w), 1:n)
    ws = view(full(w), n+1:n+ns)
    wy = view(full(w), kkt.ind_eq_shifted)
    wz = view(full(w), kkt.ind_ineq_shifted)

    x = kkt.pd_buffer
    xx = view(x, 1:n)
    xy = view(x, n+1:n+n_eq)

    Σs = get_slack_regularization(kkt)
    buf = kkt.buffer
    buf_ineq = view(buf, kkt.ind_ineq)

    fill!(x, zero(T))

    # g_z += Σs⁻¹ g_s, g_s = Σs⁻¹ g_s
    wz .+= ws ./ Σs
    ws ./= Σs

    # Save g_z and build g_x + Jᵀ (D g_z)
    fill!(buf, zero(T))
    buf_ineq .= wz
    wz .*= kkt.diag_buffer

    xy .= wy
    wy .= 0
    mul!(wx, kkt.jac', dual(w), one(T), one(T))

    # Solve K_condᵀ g = [g_x; g_y]
    xx .= wx
    solve!(kkt.linear_solver, x)

    # g_r_x, g_r_y
    wx .= xx
    wy .+= xy

    # g_B = -g_z + J g_r_x
    buf_ineq .*= .-one(T)
    mul!(buf, kkt.jac, xx, one(T), one(T))

    # g_r_z, g_r_s
    buf_ineq .*= kkt.diag_buffer
    wz .= buf_ineq
    ws .+= buf_ineq ./ Σs
    return
end

function adjoint_solve!(kkt::DenseCondensedKKTSystem, w::AbstractKKTVector)
    _adjoint_finish_bounds!(kkt, w)
    _adjoint_dense_condensed_solve!(kkt, w)
    _adjoint_reduce_rhs!(kkt, w)
    return w
end