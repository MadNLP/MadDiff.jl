function jacobian_transpose!(result::JacobianTransposeResult, sens::MadDiffSolver{T}) where {T}
    cb = sens.solver.cb
    n_x = get_nvar(sens.solver.nlp)
    n_con = get_ncon(sens.solver.nlp)

    work = VJPResult(sens)

    dL_dx = zeros_like(cb, T, n_x)
    dL_dy = zeros_like(cb, T, n_con)
    dL_dzl = zeros_like(cb, T, n_x)
    dL_dzu = zeros_like(cb, T, n_x)

    for i in 1:n_x
        onehot!(dL_dx, i)
        vector_jacobian_product!(work, sens; dL_dx)
        view(result.dx, :, i) .= work.grad_p
    end

    for i in 1:n_con
        onehot!(dL_dy, i)
        vector_jacobian_product!(work, sens; dL_dy)
        view(result.dy, :, i) .= work.grad_p
    end

    for i in 1:n_x
        onehot!(dL_dzl, i)
        vector_jacobian_product!(work, sens; dL_dzl)
        view(result.dzl, :, i) .= work.grad_p
    end

    for i in 1:n_x
        onehot!(dL_dzu, i)
        vector_jacobian_product!(work, sens; dL_dzu)
        view(result.dzu, :, i) .= work.grad_p
    end

    vector_jacobian_product!(work, sens; dobj = one(T))
    result.dobj .= work.grad_p
    return result
end

function onehot!(x::AbstractArray{T}, i) where {T}
    fill!(x, zero(T))
    x[i] = one(T)
end