function unpack_dx!(x_full::AbstractMatrix, cb::AbstractCallback, x::AbstractMatrix)
    x_full .= x[1:cb.nvar, :]
end

function pack_hess!(x::AbstractMatrix, cb::SparseCallback{T, VT, VI, NLP, FH}, x_full::AbstractMatrix) where {T, VT, VI, NLP, FH<:MakeParameter}
    pack_dx!(x, cb, x_full)
    x .*= cb.obj_scale[]
end

function pack_z!(z::AbstractMatrix, cb::AbstractCallback, z_full::AbstractMatrix)
    z .= z_full ./ cb.obj_scale[]
end

function pack_slack!(s::AbstractMatrix, cb::AbstractCallback, s_full::AbstractMatrix)
    s .= (s_full .* cb.con_scale)[cb.ind_ineq, :]
end

function unpack_dx!(x_full::AbstractMatrix, cb::SparseCallback{T, VT, VI, NLP, FH}, x::AbstractMatrix) where {T, VT, VI, NLP, FH<:MakeParameter}
    fill!(x_full, zero(eltype(x_full)))
    x_full[cb.fixed_handler.free, :] .= x[1:cb.nvar, :]
end

function pack_dx!(x::AbstractMatrix, cb::SparseCallback{T, VT, VI, NLP, FH}, x_full::AbstractMatrix) where {T, VT, VI, NLP, FH<:MakeParameter}
    x .= @view x_full[cb.fixed_handler.free, :]
end

function pack_z!(z::AbstractMatrix, cb::SparseCallback{T, VT, VI, NLP, FH}, z_full::AbstractMatrix) where {T, VT, VI, NLP, FH<:MakeParameter}
    free = cb.fixed_handler.free
    z .= @view(z_full[free, :]) ./ cb.obj_scale[]
end

function unpack_dzl!(dz::AbstractMatrix, cb, rhs::AbstractMatrix, pv::AbstractMatrix)
    fill!(pv, zero(eltype(pv)))
    pv[cb.ind_lb, :] .= rhs
    unpack_dx!(dz, cb, @view pv[1:cb.nvar, :])
    dz ./= cb.obj_scale[]
end

function unpack_dzu!(dz::AbstractMatrix, cb, rhs::AbstractMatrix, pv::AbstractMatrix)
    fill!(pv, zero(eltype(pv)))
    pv[cb.ind_ub, :] .= rhs
    unpack_dx!(dz, cb, @view pv[1:cb.nvar, :])
    dz ./= cb.obj_scale[]
end

function pack_dzl!(dz::AbstractMatrix, cb::AbstractCallback, dz_full::AbstractMatrix, pv::AbstractMatrix)
    fill!(pv, zero(eltype(pv)))
    pack_z!(view(pv, 1:cb.nvar, :), cb, view(dz_full, 1:cb.nvar, :))
    dz .= pv[cb.ind_lb, :]
end

function pack_dzl!(dz::AbstractMatrix, cb::SparseCallback{T, VT, VI, NLP, FH}, dz_full::AbstractMatrix, pv::AbstractMatrix) where {T, VT, VI, NLP, FH<:MakeParameter}
    fill!(pv, zero(eltype(pv)))
    pack_z!(view(pv, 1:cb.nvar, :), cb, dz_full)
    dz .= pv[cb.ind_lb, :]
end

function pack_dzu!(dz::AbstractMatrix, cb::AbstractCallback, dz_full::AbstractMatrix, pv::AbstractMatrix)
    fill!(pv, zero(eltype(pv)))
    pack_z!(view(pv, 1:cb.nvar, :), cb, view(dz_full, 1:cb.nvar, :))
    dz .= pv[cb.ind_ub, :]
end

function pack_dzu!(dz::AbstractMatrix, cb::SparseCallback{T, VT, VI, NLP, FH}, dz_full::AbstractMatrix, pv::AbstractMatrix) where {T, VT, VI, NLP, FH<:MakeParameter}
    fill!(pv, zero(eltype(pv)))
    pack_z!(view(pv, 1:cb.nvar, :), cb, dz_full)
    dz .= pv[cb.ind_ub, :]
end

function set_fixed_sensitivity!(dx::AbstractMatrix, cb::SparseCallback{T, VT, VI, NLP, FH}, dlvar_dp, duvar_dp) where {T, VT, VI, NLP, FH<:MakeParameter}
    fixed_idx = cb.fixed_handler.fixed
    if isnothing(dlvar_dp) && isnothing(duvar_dp)
        return nothing
    elseif isnothing(duvar_dp)
        dx[fixed_idx, :] .= dlvar_dp[fixed_idx, :]
    elseif isnothing(dlvar_dp)
        dx[fixed_idx, :] .= duvar_dp[fixed_idx, :]
    else
        dx[fixed_idx, :] .= (dlvar_dp[fixed_idx, :] .+ duvar_dp[fixed_idx, :]) ./ 2
    end
    return nothing
end

function pack_hess_param!(dest::AbstractVector, vals::AbstractVector, obj_scale)
    dest .= -obj_scale .* vals
    return nothing
end

function pack_jac_param!(dest::AbstractVector, vals::AbstractVector, con_scale::AbstractVector)
    dest .= -con_scale .* vals
    return nothing
end

function pack_lcon_eq_param!(dest::AbstractVector, vals::AbstractVector, con_scale::AbstractVector, is_eq::AbstractVector)
    dest .= (con_scale .* is_eq .* vals) ./ 2
    return nothing
end

function pack_ucon_eq_param!(dest::AbstractVector, vals::AbstractVector, con_scale::AbstractVector, is_eq::AbstractVector)
    dest .= (con_scale .* is_eq .* vals) ./ 2
    return nothing
end

function pack_lvar_param!(dest::AbstractVector, vals::AbstractVector, ::AbstractUnreducedKKTSystem, _)
    dest .= vals
    return nothing
end
function pack_lvar_param!(dest::AbstractVector, vals::AbstractVector, kkt, pos)
    dest .= vals .* kkt.l_lower[pos]
    return nothing
end

function pack_uvar_param!(dest::AbstractVector, vals::AbstractVector, ::AbstractUnreducedKKTSystem, _)
    dest .= .-vals
    return nothing
end
function pack_uvar_param!(dest::AbstractVector, vals::AbstractVector, kkt, pos)
    dest .= .-vals .* kkt.u_lower[pos]
    return nothing
end

function pack_lcon_slack_param!(dest::AbstractVector, vals::AbstractVector, sc::AbstractVector, ::AbstractUnreducedKKTSystem, _)
    dest .= sc .* vals
    return nothing
end
function pack_lcon_slack_param!(dest::AbstractVector, vals::AbstractVector, sc::AbstractVector, kkt, pos)
    dest .= (sc .* vals) .* kkt.l_lower[pos]
    return nothing
end

function pack_ucon_slack_param!(dest::AbstractVector, vals::AbstractVector, sc::AbstractVector, ::AbstractUnreducedKKTSystem, _)
    dest .= .-(sc .* vals)
    return nothing
end
function pack_ucon_slack_param!(dest::AbstractVector, vals::AbstractVector, sc::AbstractVector, kkt, pos)
    dest .= .-(sc .* vals) .* kkt.u_lower[pos]
    return nothing
end

function unpack_slack!(out::AbstractMatrix, cb, dz::AbstractMatrix, is_eq, dy::AbstractMatrix)
    out .= (is_eq .* dy ./ 2) .* cb.con_scale
    out[cb.ind_ineq, :] .+= view(dz, cb.nvar + 1:size(dz, 1), :) .* cb.con_scale[cb.ind_ineq]
    return nothing
end

function _jacobian_row_ranges(kkt)
    n_primal = length(kkt.pr_diag)
    n_dual = length(kkt.du_diag)
    n_lb = length(kkt.l_diag)
    n_ub = length(kkt.u_diag)
    primal_rows = 1:n_primal
    dual_rows = n_primal + 1:n_primal + n_dual
    lb_rows = n_primal + n_dual + 1:n_primal + n_dual + n_lb
    ub_rows = n_primal + n_dual + n_lb + 1:n_primal + n_dual + n_lb + n_ub
    return primal_rows, dual_rows, lb_rows, ub_rows
end

function _set_fixed_sensitivity_from_sparse!(::AbstractMatrix, ::AbstractCallback, ::Any)
    return nothing
end

function _set_fixed_sensitivity_from_sparse!(
    dx::AbstractMatrix,
    cb::SparseCallback{T, VT, VI, NLP, FH},
    jcache,
) where {T, VT, VI, NLP, FH<:MakeParameter}
    fixed = jcache.fixed_idx
    isempty(fixed) && return nothing
    has_l = !isempty(jcache.fixed_l.src)
    has_u = !isempty(jcache.fixed_u.src)
    (!has_l && !has_u) && return nothing

    if has_l
        fl = jcache.fixed_l
        fl.raw.V .= jcache.lvar_raw.V[fl.src]
        transfer!(fl.mat, fl.raw, fl.map)
        fill!(fl.work, zero(T))
        mul!(fl.work, fl.mat, jcache.eye_p)
    end

    if has_u
        fu = jcache.fixed_u
        fu.raw.V .= jcache.uvar_raw.V[fu.src]
        transfer!(fu.mat, fu.raw, fu.map)
        fill!(fu.work, zero(T))
        mul!(fu.work, fu.mat, jcache.eye_p)
    end

    if has_l && has_u
        @views dx[fixed, :] .= (jcache.fixed_l.work .+ jcache.fixed_u.work) ./ 2
    elseif has_l
        @views dx[fixed, :] .= jcache.fixed_l.work
    else
        @views dx[fixed, :] .= jcache.fixed_u.work
    end
    return nothing
end