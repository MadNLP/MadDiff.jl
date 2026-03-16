function MadDiff.pack_hess!(x::AbstractMatrix, bcb::UniformBatchCallback, x_full::AbstractMatrix)
    MadDiff.pack_dx!(x, bcb, x_full)
    x .*= bcb.obj_scale
    return nothing
end

function MadDiff.pack_cons!(c::AbstractMatrix, bcb::UniformBatchCallback, c_full::AbstractMatrix)
    c .= c_full .* bcb.con_scale
    return nothing
end

function MadDiff.pack_z!(z::AbstractMatrix, bcb::UniformBatchCallback, z_full::AbstractMatrix)
    z .= z_full ./ bcb.obj_scale
    return nothing
end

function MadDiff.pack_dy!(y::AbstractMatrix, bcb::UniformBatchCallback, y_full::AbstractMatrix)
    y .= (y_full .* (bcb.obj_sign ./ bcb.obj_scale)) .* bcb.con_scale
    return nothing
end

function MadDiff.unpack_dzl!(dz::AbstractMatrix, bcb::UniformBatchCallback, rhs::AbstractMatrix, pv::BatchPrimalVector)
    fill!(MadNLP.full(pv), zero(eltype(MadNLP.full(pv))))
    lower(pv) .= rhs
    MadDiff.unpack_dx!(dz, bcb, MadNLP.variable(pv))
    dz ./= bcb.obj_scale
    return nothing
end

function MadDiff.unpack_dzu!(dz::AbstractMatrix, bcb::UniformBatchCallback, rhs::AbstractMatrix, pv::BatchPrimalVector)
    fill!(MadNLP.full(pv), zero(eltype(MadNLP.full(pv))))
    upper(pv) .= rhs
    MadDiff.unpack_dx!(dz, bcb, MadNLP.variable(pv))
    dz ./= bcb.obj_scale
    return nothing
end

function MadDiff.unpack_slack!(out::AbstractMatrix, bcb::UniformBatchCallback, dz::BatchPrimalVector, is_eq, dy::AbstractMatrix)
    out .= (is_eq .* dy ./ 2) .* bcb.con_scale
    ns = length(bcb.ind_ineq)
    if ns > 0
        out[bcb.ind_ineq, :] .+= MadNLP.slack(dz) .* bcb.con_scale[bcb.ind_ineq, :]
    end
    return nothing
end
