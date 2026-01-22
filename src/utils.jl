struct SharedCache{VT, VK, PV}
    rhs::VK
    dzl_kkt::VT
    dzu_kkt::VT
    dλ::VT
    dx_full::VT
    zl_buffer::PV
    zu_buffer::PV
    dx_kkt::VT
end

function _get_shared_cache!(sens)
    if isnothing(sens.shared_cache)
        cb = sens.solver.cb
        dims = sens.dims
        x_array = MadNLP.full(sens.solver.x)
        T = eltype(x_array)
        VT = typeof(x_array)
        sens.shared_cache = SharedCache(
            MadNLP.UnreducedKKTVector(sens.kkt),
            _zeros_like(x_array, T, length(cb.ind_lb)),
            _zeros_like(x_array, T, length(cb.ind_ub)),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_x),
            _create_primal_buffer(VT, cb),
            _create_primal_buffer(VT, cb),
            _zeros_like(x_array, T, cb.nvar),
        )
    end
    return sens.shared_cache
end

function _pack_primal!(out, v, cb::MadNLP.AbstractCallback)
    copyto!(out, v)
    return out
end
function _pack_primal!(out, v, cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter}
    out .= @view v[cb.fixed_handler.free]
    return out
end

function _pack_slack!(out, v, cb)
    @views out .= v[cb.ind_ineq]
    return out
end

_kkt_to_full_idx(cb::MadNLP.AbstractCallback, kkt_idx) = kkt_idx
_kkt_to_full_idx(cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}, kkt_idx) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter} = cb.fixed_handler.free[kkt_idx]

_zeros_like(x_array, ::Type{T}, n::Int) where {T} = fill!(similar(x_array, T, n), zero(T))

_falses_like(x_array, n::Int) = fill!(similar(x_array, Bool, n), false)

_to_bool_array(x_array, v::BitVector) = convert(Vector{Bool}, v)
_to_bool_array(x_array, v) = v

_get_fixed_idx(::MadNLP.AbstractCallback, ref_array) = similar(ref_array, Int, 0)
function _get_fixed_idx(cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}, ref_array) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter}
    fixed = cb.fixed_handler.fixed
    result = similar(ref_array, eltype(fixed), length(fixed))
    copyto!(result, fixed)
    return result
end
function _unpack_primal!(v_full, cb::MadNLP.AbstractCallback, v_kkt)
    v_full .= v_kkt
end
function _unpack_primal!(v_full, cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}, v_kkt) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter}
    fill!(v_full, zero(eltype(v_full)))
    v_full[cb.fixed_handler.free] .= v_kkt
end

function _unpack_z!(z_full, cb, z_kkt, pv::MadNLP.PrimalVector, pv_values)
    fill!(MadNLP.full(pv), zero(eltype(MadNLP.full(pv))))
    pv_values .= z_kkt
    MadNLP.unpack_z!(z_full, cb, MadNLP.variable(pv))
end

function _scatter_to_primal_vector!(pv::MadNLP.PrimalVector, v_full, cb)
    fill!(MadNLP.full(pv), zero(eltype(MadNLP.full(pv))))
    _pack_primal!(MadNLP.variable(pv), v_full, cb)
end

_pack_seed_x!(cache, ::Nothing, cb) = nothing
_pack_seed_x!(cache, seed_x, cb) = _pack_primal!(cache, seed_x, cb)

_pack_seed_λ!(cache, ::Nothing, cb) = nothing
function _pack_seed_λ!(cache, seed_λ, cb)
    MadNLP.unpack_y!(cache, cb, seed_λ)
    return cache
end

_pack_seed_z!(cache, ::Nothing, cb, pv::MadNLP.PrimalVector, pv_values) = fill!(cache, zero(eltype(cache)))
function _pack_seed_z!(cache, seed_z, cb, pv::MadNLP.PrimalVector, pv_values)
    _scatter_to_primal_vector!(pv, seed_z, cb)
    cache .= pv_values ./ cb.obj_scale[]
    return nothing
end

function _create_primal_buffer(::Type{VT}, cb::MadNLP.AbstractCallback) where VT
    MadNLP.PrimalVector(VT, cb.nvar, length(cb.ind_ineq), cb.ind_lb, cb.ind_ub)
end

_pullback_add!(out, ::Nothing, v) = nothing
function _pullback_add!(out, M, v)
    @lencheck size(M, 1) v
    out .+= M' * v
end

_pullback_sub!(out, ::Nothing, v) = nothing
function _pullback_sub!(out, M, v)
    @lencheck size(M, 1) v
    out .-= M' * v
end

_copy_and_scale_lag!(out, ::Nothing, cb, cache) = nothing
function _copy_and_scale_lag!(out, d2L_dxdp, cb, cache)
    copyto!(cache.d2L_dxdp, d2L_dxdp)
    _pack_primal!(out, cache.d2L_dxdp, cb)
    out .*= cb.obj_scale[]
    return out
end
_copy_and_scale_con!(dest, ::Nothing, cb) = nothing
function _copy_and_scale_con!(dest, src, cb)
    copyto!(dest, src)
    dest .*= cb.con_scale
    return dest
end

_get_bound_scale(kkt::MadNLP.AbstractReducedKKTSystem) = (kkt.l_lower, kkt.u_lower)
_get_bound_scale(kkt::MadNLP.AbstractCondensedKKTSystem) = (kkt.l_lower, kkt.u_lower)
_get_bound_scale(::MadNLP.SparseUnreducedKKTSystem) = (nothing, nothing)

function _build_bound_pert!(pv::MadNLP.PrimalVector, dvar_dp, dcon_dp, cb)
    T = eltype(MadNLP.full(pv))
    fill!(MadNLP.full(pv), zero(T))
    _set_variable_pert!(pv, dvar_dp, cb)
    _set_slack_pert!(pv, dcon_dp, cb)
    return pv
end

_set_variable_pert!(pv, ::Nothing, cb) = nothing
_set_variable_pert!(pv, dvar_dp, cb) = _pack_primal!(MadNLP.variable(pv), dvar_dp, cb)

_set_slack_pert!(pv, ::Nothing, cb) = nothing
_set_slack_pert!(pv, dcon_dp, cb) = _pack_slack!(MadNLP.slack(pv), dcon_dp, cb)

function _set_fixed_sensitivity!(dx, dl_dp, du_dp, dims)
    isempty(dims.fixed_idx) && return nothing
    _unpack_fixed!(dx, dl_dp, du_dp, dims)
    return nothing
end

_unpack_fixed!(dx, ::Nothing, ::Nothing, dims) = nothing
_unpack_fixed!(dx, dl_dp, ::Nothing, dims) = (dx[dims.fixed_idx] .= dl_dp[dims.fixed_idx];)
_unpack_fixed!(dx, ::Nothing, du_dp, dims) = (dx[dims.fixed_idx] .= du_dp[dims.fixed_idx];)
function _unpack_fixed!(dx, dl_dp, du_dp, dims)
    dx[dims.fixed_idx] .= (dl_dp[dims.fixed_idx] .+ du_dp[dims.fixed_idx]) ./ 2
    return nothing
end
