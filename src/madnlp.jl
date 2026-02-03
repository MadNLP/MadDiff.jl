function unpack_x_fixed_zero!(x_full, cb::AbstractCallback, x)
    x_full .= x[1:cb.nvar]
end
function pack_cons!(c, cb::AbstractCallback, c_full)
    c .= c_full .* cb.con_scale
end
function pack_x!(x, cb::AbstractCallback, x_full)
    x .= x_full
end
function pack_x_obj!(x, cb::AbstractCallback, x_full)
    pack_x!(x, cb, x_full)
    x .*= cb.obj_scale[]
end
function pack_z!(z, cb::AbstractCallback, z_full)
    z .= z_full ./ cb.obj_scale[]
end
function pack_y!(y, cb::AbstractCallback, y_full)
    y .= (y_full .* (cb.obj_sign / cb.obj_scale[])) .* cb.con_scale
end
function pack_slack!(s, cb::AbstractCallback, s_full)
    s .= (s_full .* cb.con_scale)[cb.ind_ineq]
end
function unpack_x_fixed_zero!(x_full, cb::SparseCallback{T, VT, VI, NLP, FH}, x) where {T, VT, VI, NLP, FH<:MakeParameter}
    fill!(x_full, zero(eltype(x_full)))
    x_full[cb.fixed_handler.free] .= x[1:cb.nvar]
end
function pack_x!(x, cb::SparseCallback{T, VT, VI, NLP, FH}, x_full) where {T, VT, VI, NLP, FH<:MakeParameter}
    x .= @view x_full[cb.fixed_handler.free]
end
function pack_x_obj!(x, cb::SparseCallback{T, VT, VI, NLP, FH}, x_full) where {T, VT, VI, NLP, FH<:MakeParameter}
    pack_x!(x, cb, x_full)
    x .*= cb.obj_scale[]
end
function pack_z!(z, cb::SparseCallback{T, VT, VI, NLP, FH}, z_full) where {T, VT, VI, NLP, FH<:MakeParameter}
    free = cb.fixed_handler.free
    z .= @view(z_full[free]) ./ cb.obj_scale[]
end
