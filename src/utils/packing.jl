# ============================================================================
# Scale/sign/index bookkeeping between NLP-facing tensors and the KKT layout.
#
# The two methods per operation split on whether the callback fixes variables
# via `MakeParameter` (which packs/unpacks a `free` subset) or not.
# ============================================================================

const SparseCB_MP{T,VT,VI,NLP} = SparseCallback{T, VT, VI, NLP, <:MakeParameter}

# ---------- x/z: primal variables and bound duals ----------

unpack_dx!(x_full, cb::AbstractCallback, x) = (x_full .= @view(x[1:cb.nvar]); x_full)

pack_dx!(x, cb::AbstractCallback, x_full) = (x .= x_full; x)

function unpack_dx!(x_full::AbstractVector, cb::SparseCB_MP, x::AbstractVector)
    fill!(x_full, zero(eltype(x_full)))
    x_full[cb.fixed_handler.free] .= @view(x[1:cb.nvar])
    return x_full
end

pack_dx!(x::AbstractVector, cb::SparseCB_MP, x_full::AbstractVector) =
    (x .= @view x_full[cb.fixed_handler.free]; x)

# ---------- hessian, bound duals, constraint scaling ----------

pack_hess!(x, cb::AbstractCallback, x_full) =
    (pack_dx!(x, cb, x_full); x .*= cb.obj_scale[]; x)

pack_z!(z, cb::AbstractCallback, z_full) = (z .= z_full ./ cb.obj_scale[]; z)

function pack_z!(z::AbstractVector, cb::SparseCB_MP, z_full::AbstractVector)
    z .= @view(z_full[cb.fixed_handler.free]) ./ cb.obj_scale[]
    return z
end

pack_cons!(c, cb::AbstractCallback, c_full) = (c .= c_full .* cb.con_scale; c)

pack_dy!(y, cb::AbstractCallback, y_full) =
    (y .= (y_full .* (cb.obj_sign / cb.obj_scale[])) .* cb.con_scale; y)

# Avoid materializing the full `(s_full .* cb.con_scale)` product before
# slicing — iterate only the `ind_ineq` rows via views so the broadcast is
# O(length(ind_ineq)) instead of O(length(s_full)) and allocation-free.
pack_slack!(s, cb::AbstractCallback, s_full) =
    (s .= @view(s_full[cb.ind_ineq]) .* @view(cb.con_scale[cb.ind_ineq]); s)

# ---------- bound duals via PrimalVector scratch ----------

function unpack_dzl!(dz, cb, rhs, pv)
    fill!(full(pv), zero(eltype(full(pv))))
    pv.values_lr .= rhs
    unpack_z!(dz, cb, variable(pv))
    return dz
end

function unpack_dzu!(dz, cb, rhs, pv)
    fill!(full(pv), zero(eltype(full(pv))))
    pv.values_ur .= rhs
    unpack_z!(dz, cb, variable(pv))
    return dz
end

function pack_dzl!(dz, cb, dz_full, pv)
    pack_z!(variable(pv), cb, dz_full)
    dz .= pv.values_lr
    return dz
end

function pack_dzu!(dz, cb, dz_full, pv)
    pack_z!(variable(pv), cb, dz_full)
    dz .= pv.values_ur
    return dz
end

# ---------- fixed-variable sensitivity handoff ----------

set_fixed_sensitivity!(_dx, ::AbstractCallback, _dlvar, _duvar) = nothing

function set_fixed_sensitivity!(dx::AbstractVector, cb::SparseCB_MP, dlvar, duvar)
    fixed = cb.fixed_handler.fixed
    if dlvar === nothing && duvar === nothing
        return nothing
    elseif duvar === nothing
        dx[fixed] .= view(dlvar, fixed)
    elseif dlvar === nothing
        dx[fixed] .= view(duvar, fixed)
    else
        dxv = view(dx, fixed); dlv = view(dlvar, fixed); duv = view(duvar, fixed)
        @. dxv = (dlv + duv) / 2
    end
    return nothing
end

# ---------- slack recombination ----------

function unpack_slack!(out, cb, dz, is_eq, dy)
    @. out = is_eq * dy / 2 * cb.con_scale
    out_i = @view out[cb.ind_ineq]
    cs_i  = @view cb.con_scale[cb.ind_ineq]
    @. out_i += slack(dz) * cs_i
    return out
end
