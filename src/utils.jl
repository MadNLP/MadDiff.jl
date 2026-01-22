const SUCCESSFUL_STATUSES = (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)
function assert_solved_and_feasible(solver::MadNLP.AbstractMadNLPSolver)
    solver.status ∉ SUCCESSFUL_STATUSES &&
        error("Solver did not converge successfully: $(solver.status)")
    return nothing
end

struct SensitivityDims{VI, VB}
    n_x::Int
    n_con::Int
    n_p::Int
    fixed_idx::VI
    is_eq::VB
end

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

function _create_primal_buffer(::Type{VT}, cb::MadNLP.AbstractCallback) where VT
    MadNLP.PrimalVector(VT, cb.nvar, length(cb.ind_ineq), cb.ind_lb, cb.ind_ub)
end

struct ForwardCache{VT}
    d2L_dxdp::VT
    dg_dp::VT
    dlcon_dp::VT
    ducon_dp::VT
end

function _get_forward_cache!(sens)
    if isnothing(sens.forward_cache)
        dims = sens.dims
        x_array = MadNLP.full(sens.solver.x)
        T = eltype(x_array)
        sens.forward_cache = ForwardCache(
            _zeros_like(x_array, T, dims.n_x),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, dims.n_con),
        )
    end
    return sens.forward_cache
end

struct ReverseCache{VT}
    dL_dx::VT
    dL_dλ::VT
    dL_dzl::VT
    dL_dzu::VT
    eq_scale::VT
    grad_p::VT
    dλ_scaled::VT
end

function _get_reverse_cache!(sens)
    if isnothing(sens.reverse_cache)
        cb = sens.solver.cb
        dims = sens.dims
        x_array = MadNLP.full(sens.solver.x)
        T = eltype(x_array)

        sens.reverse_cache = ReverseCache(
            _zeros_like(x_array, T, cb.nvar),
            _zeros_like(x_array, T, dims.n_con),
            _zeros_like(x_array, T, length(cb.ind_lb)),
            _zeros_like(x_array, T, length(cb.ind_ub)),
            ifelse.(dims.is_eq, T(1 // 2), one(T)),
            _zeros_like(x_array, T, dims.n_p),
            _zeros_like(x_array, T, dims.n_con),
        )
    end
    return sens.reverse_cache
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

_zeros_like(x_array, ::Type{T}, n::Int) where {T} = fill!(similar(x_array, T, n), zero(T))

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
    return nothing
end
function _unpack_primal!(v_full, cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}, v_kkt) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter}
    fill!(v_full, zero(eltype(v_full)))
    v_full[cb.fixed_handler.free] .= v_kkt
    return nothing
end

function _unpack_z!(z_full, cb, z_kkt, pv::MadNLP.PrimalVector, pv_values)
    fill!(MadNLP.full(pv), zero(eltype(MadNLP.full(pv))))
    pv_values .= z_kkt
    MadNLP.unpack_z!(z_full, cb, MadNLP.variable(pv))
    return nothing
end

_pack_dL_dx!(cache, ::Nothing, cb) = nothing
_pack_dL_dx!(cache, dL_dx, cb) = _pack_primal!(cache, dL_dx, cb)

_pack_dL_dλ!(cache, ::Nothing, cb) = nothing
function _pack_dL_dλ!(cache, dL_dλ, cb)
    MadNLP.unpack_y!(cache, cb, dL_dλ)
    return cache
end

_pack_dL_dz!(cache, ::Nothing, cb, pv::MadNLP.PrimalVector, pv_values) = fill!(cache, zero(eltype(cache)))
function _pack_dL_dz!(cache, dL_dz, cb, pv::MadNLP.PrimalVector, pv_values)
    fill!(MadNLP.full(pv), zero(eltype(MadNLP.full(pv))))
    _pack_primal!(MadNLP.variable(pv), dL_dz, cb)
    cache .= pv_values ./ cb.obj_scale[]
    return nothing
end

_pullback_add!(out, ::Nothing, v) = nothing
function _pullback_add!(out, M, v)
    @lencheck size(M, 1) v
    out .+= M' * v
    return nothing
end

_pullback_sub!(out, ::Nothing, v) = nothing
function _pullback_sub!(out, M, v)
    @lencheck size(M, 1) v
    out .-= M' * v
    return nothing
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

_set_variable_pert!(pv, ::Nothing, cb) = nothing
_set_variable_pert!(pv, dvar_dp, cb) = _pack_primal!(MadNLP.variable(pv), dvar_dp, cb)

_set_slack_pert!(pv, ::Nothing, cb) = nothing
_set_slack_pert!(pv, dcon_dp, cb) = _pack_slack!(MadNLP.slack(pv), dcon_dp, cb)

# TODO: ask madnlp/madnlpgpu for this
_factorization_ok(ls::MadNLP.LDLSolver) = MadNLP.LDLFactorizations.factorized(ls.inner)
_factorization_ok(ls::MadNLP.CHOLMODSolver) = MadNLP.issuccess(ls.inner)
_factorization_ok(ls::MadNLP.UmfpackSolver) = MadNLP.UMFPACK.issuccess(ls.inner)
_factorization_ok(ls::MadNLP.LapackCPUSolver) = ls.info[] == 0
_factorization_ok(ls::MadNLP.MumpsSolver) = !ls.is_singular
_factorization_ok(::Any) = true

function _has_custom_config(config::MadDiffConfig)
    return !isnothing(config.kkt_system) ||
        !isnothing(config.kkt_options) ||
        !isnothing(config.linear_solver) ||
        !isnothing(config.linear_solver_options)
end

_get_wrapper_type(x) = Base.typename(typeof(x)).wrapper