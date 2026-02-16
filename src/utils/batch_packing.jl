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

function unpack_slack!(out::AbstractMatrix, cb, dz::AbstractMatrix, is_eq, dy::AbstractMatrix)
    out .= is_eq .* dy ./ 2
    out[cb.ind_ineq, :] .+= view(dz, cb.nvar + 1:size(dz, 1), :) .* cb.con_scale[cb.ind_ineq]
    out .*= cb.con_scale
    return nothing
end
