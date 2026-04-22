# ============================================================================
# Batch pack/unpack — matrix overloads of the scalar packing helpers.
#
# The scalar `pack_*` / `unpack_*` take vectors and a `SparseCallback`; the
# batch variants take `(dim, bs)` matrices and a `UniformBatchCallback`. The
# scale factors (`obj_scale`, `con_scale`, `obj_sign`) are per-instance row
# vectors so broadcasts align naturally.
# ============================================================================

function MadDiff.pack_hess!(x::AbstractMatrix, bcb::UniformBatchCallback,
                             x_full::AbstractMatrix)
    MadDiff.pack_dx!(x, bcb, x_full)
    x .*= bcb.obj_scale
    return x
end

MadDiff.pack_cons!(c::AbstractMatrix, bcb::UniformBatchCallback, c_full::AbstractMatrix) =
    (c .= c_full .* bcb.con_scale; c)

MadDiff.pack_z!(z::AbstractMatrix, bcb::UniformBatchCallback, z_full::AbstractMatrix) =
    (z .= z_full ./ bcb.obj_scale; z)

MadDiff.pack_dy!(y::AbstractMatrix, bcb::UniformBatchCallback, y_full::AbstractMatrix) =
    (y .= (y_full .* (bcb.obj_sign ./ bcb.obj_scale)) .* bcb.con_scale; y)

function MadDiff.unpack_dzl!(dz::AbstractMatrix, bcb::UniformBatchCallback,
                              rhs::AbstractMatrix, pv::BatchPrimalVector)
    fill!(MadNLP.full(pv), zero(eltype(MadNLP.full(pv))))
    lower(pv) .= rhs
    MadDiff.unpack_dx!(dz, bcb, MadNLP.variable(pv))
    dz ./= bcb.obj_scale
    return dz
end

function MadDiff.unpack_dzu!(dz::AbstractMatrix, bcb::UniformBatchCallback,
                              rhs::AbstractMatrix, pv::BatchPrimalVector)
    fill!(MadNLP.full(pv), zero(eltype(MadNLP.full(pv))))
    upper(pv) .= rhs
    MadDiff.unpack_dx!(dz, bcb, MadNLP.variable(pv))
    dz ./= bcb.obj_scale
    return dz
end

function MadDiff.unpack_slack!(out::AbstractMatrix, bcb::UniformBatchCallback,
                                dz::BatchPrimalVector, is_eq, dy::AbstractMatrix)
    out .= (is_eq .* dy ./ 2) .* bcb.con_scale
    isempty(bcb.ind_ineq) ||
        (view(out, bcb.ind_ineq, :) .+= MadNLP.slack(dz) .*
         view(bcb.con_scale, bcb.ind_ineq, :))
    return out
end
