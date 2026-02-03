function unpack_dx!(x_full, cb::AbstractCallback, x)
    x_full .= x[1:cb.nvar]
end
function pack_cons!(c, cb::AbstractCallback, c_full)
    c .= c_full .* cb.con_scale
end
function pack_dx!(x, cb::AbstractCallback, x_full)
    x .= x_full
end
function pack_hess(x, cb::AbstractCallback, x_full)
    pack_dx!(x, cb, x_full)
    x .*= cb.obj_scale[]
end
function pack_z!(z, cb::AbstractCallback, z_full)
    z .= z_full ./ cb.obj_scale[]
end
function pack_dy!(y, cb::AbstractCallback, y_full)
    y .= (y_full .* (cb.obj_sign / cb.obj_scale[])) .* cb.con_scale
end
function pack_slack!(s, cb::AbstractCallback, s_full)
    s .= (s_full .* cb.con_scale)[cb.ind_ineq]
end
function unpack_dx!(x_full, cb::SparseCallback{T, VT, VI, NLP, FH}, x) where {T, VT, VI, NLP, FH<:MakeParameter}
    fill!(x_full, zero(eltype(x_full)))
    x_full[cb.fixed_handler.free] .= x[1:cb.nvar]
end
function pack_dx!(x, cb::SparseCallback{T, VT, VI, NLP, FH}, x_full) where {T, VT, VI, NLP, FH<:MakeParameter}
    x .= @view x_full[cb.fixed_handler.free]
end
function pack_hess(x, cb::SparseCallback{T, VT, VI, NLP, FH}, x_full) where {T, VT, VI, NLP, FH<:MakeParameter}
    pack_dx!(x, cb, x_full)
    x .*= cb.obj_scale[]
end
function pack_z!(z, cb::SparseCallback{T, VT, VI, NLP, FH}, z_full) where {T, VT, VI, NLP, FH<:MakeParameter}
    free = cb.fixed_handler.free
    z .= @view(z_full[free]) ./ cb.obj_scale[]
end
function unpack_dzl!(dz, cb, rhs, pv)
    fill!(full(pv), zero(eltype(full(pv))))
    pv.values_lr .= rhs
    unpack_z!(dz, cb, variable(pv))
end
function unpack_dzu!(dz, cb, rhs, pv)
    fill!(full(pv), zero(eltype(full(pv))))
    pv.values_ur .= rhs
    unpack_z!(dz, cb, variable(pv))
end
function pack_dzl!(dz, cb, dz_full, pv)
    pack_z!(variable(pv), cb, dz_full)
    dz .= pv.values_lr
end
function pack_dzu!(dz, cb, dz_full, pv)
    pack_z!(variable(pv), cb, dz_full)
    dz .= pv.values_ur
end

set_fixed_sensitivity!(dx, cb::AbstractCallback, dlvar_dp, duvar_dp) = nothing
function set_fixed_sensitivity!(dx, cb::SparseCallback{T, VT, VI, NLP, FH}, dlvar_dp, duvar_dp) where {T, VT, VI, NLP, FH<:MakeParameter}
    fixed_idx = cb.fixed_handler.fixed
    if isnothing(dlvar_dp) && isnothing(duvar_dp)
        return nothing
    elseif isnothing(duvar_dp)
        dx[fixed_idx] .= dlvar_dp[fixed_idx]
    elseif isnothing(dlvar_dp)
        dx[fixed_idx] .= duvar_dp[fixed_idx]
    else
        dx[fixed_idx] .= (dlvar_dp[fixed_idx] .+ duvar_dp[fixed_idx]) ./ 2
    end
    return nothing
end