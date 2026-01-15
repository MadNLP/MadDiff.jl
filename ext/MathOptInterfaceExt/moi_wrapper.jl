MOI.is_empty(m::Optimizer) = MOI.is_empty(m.inner)
MOI.supports(m::Optimizer, x::MOI.AnyAttribute, args...) = MOI.supports(m.inner, x, args...)
MOI.get(m::Optimizer, x::MOI.AnyAttribute, args...) = MOI.get(m.inner, x, args...)
MOI.get(m::Optimizer, x::MOI.AnyAttribute, y::Vector) = MOI.get(m.inner, x, y)
MOI.set(m::Optimizer, x::MOI.AnyAttribute, args...) = MOI.set(m.inner, x, args...)
MOI.is_valid(m::Optimizer, i) = MOI.is_valid(m.inner, i)
MOI.add_variable(m::Optimizer) = MOI.add_variable(m.inner)
MOI.add_variables(m::Optimizer, n::Int) = MOI.add_variables(m.inner, n)
MOI.add_constraint(m::Optimizer, f::MOI.AbstractFunction, s::MOI.AbstractSet) = MOI.add_constraint(m.inner, f, s)
MOI.add_constrained_variable(m::Optimizer, set::MOI.AbstractScalarSet) = MOI.add_constrained_variable(m.inner, set)
MOI.add_constrained_variables(m::Optimizer, set::MOI.AbstractVectorSet) = MOI.add_constrained_variables(m.inner, set)
MOI.supports_incremental_interface(m::Optimizer) = MOI.supports_incremental_interface(m.inner)
MOI.supports_add_constrained_variable(m::Optimizer, S::Type{<:MOI.AbstractScalarSet}) = MOI.supports_add_constrained_variable(m.inner, S)
MOI.supports_constraint(m::Optimizer, F::Type{<:MOI.AbstractFunction}, S::Type{<:MOI.AbstractSet}) = MOI.supports_constraint(m.inner, F, S)
MOI.supports_add_constrained_variables(m::Optimizer, S::Type{<:MOI.AbstractVectorSet}) = MOI.supports_add_constrained_variables(m.inner, S)
MOI.supports_add_constrained_variables(m::Optimizer, S::Type{MOI.Reals}) = MOI.supports_add_constrained_variables(m.inner, S)

function MOI.add_constrained_variable(m::Optimizer, set::MOI.Parameter{T}) where {T}
    vi, ci = MOI.add_constrained_variable(m.inner, set)
    m.param_ci_to_vi[ci] = vi
    return vi, ci
end
MOI.delete(m::Optimizer, vi::MOI.VariableIndex) = MOI.delete(m.inner, vi)
function MOI.delete(m::Optimizer, ci::MOI.ConstraintIndex)
    delete!(m.param_ci_to_vi, ci)  # no-op if not a parameter constraint
    return MOI.delete(m.inner, ci)
end

MOI.get(model::Optimizer, ::MadDiff.DifferentiateTimeSec) = model.diff_time
function MOI.get(m::Optimizer, attr::MOI.RawOptimizerAttribute)
    name = attr.name
    if name == MadDiff.MADDIFF_KKTSYSTEM
        return m.sensitivity_config.kkt_system
    elseif name == MadDiff.MADDIFF_KKTSYSTEM_OPTIONS
        return m.sensitivity_config.kkt_options
    elseif name == MadDiff.MADDIFF_LINEARSOLVER
        return m.sensitivity_config.linear_solver
    elseif name == MadDiff.MADDIFF_LINEARSOLVER_OPTIONS
        return m.sensitivity_config.linear_solver_options
    elseif name == MadDiff.MADDIFF_REGULARIZATION
        return m.sensitivity_config.regularization
    else
        return MOI.get(m.inner, attr)
    end
end

function MOI.set(m::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    name = attr.name
    if name == MadDiff.MADDIFF_KKTSYSTEM
        m.sensitivity_config.kkt_system = value
    elseif name == MadDiff.MADDIFF_KKTSYSTEM_OPTIONS
        m.sensitivity_config.kkt_options = value
    elseif name == MadDiff.MADDIFF_LINEARSOLVER
        m.sensitivity_config.linear_solver = value
    elseif name == MadDiff.MADDIFF_LINEARSOLVER_OPTIONS
        m.sensitivity_config.linear_solver_options = value
    elseif name == MadDiff.MADDIFF_REGULARIZATION
        m.sensitivity_config.regularization = value
    else
        MOI.set(m.inner, attr, value)
        return
    end
    _invalidate_sensitivity!(m)
    return
end

function MOI.supports(m::Optimizer, attr::MOI.RawOptimizerAttribute)
    name = attr.name
    return name in (
        MadDiff.MADDIFF_KKTSYSTEM,
        MadDiff.MADDIFF_KKTSYSTEM_OPTIONS,
        MadDiff.MADDIFF_LINEARSOLVER,
        MadDiff.MADDIFF_LINEARSOLVER_OPTIONS,
        MadDiff.MADDIFF_REGULARIZATION,
    ) || MOI.supports(m.inner, attr)
end

function MOI.empty!(m::Optimizer)
    MOI.empty!(m.inner)
    empty!(m.param_ci_to_vi)
    empty!(m.vi_to_lb_idx)
    empty!(m.vi_to_ub_idx)
    empty!(m.forward.param_perturbations)
    empty!(m.reverse.primal_inputs)
    empty!(m.reverse.dual_inputs)
    return _invalidate_sensitivity!(m)
end

function MOI.optimize!(m::Optimizer)
    _invalidate_sensitivity!(m)
    return MOI.optimize!(m.inner)
end

function MOI.copy_to(m::Optimizer, src::MOI.ModelLike)
    _invalidate_sensitivity!(m)
    return MOI.copy_to(m.inner, src)
end

function MOI.get(m::Optimizer, ::MOI.SolverName)
    return "MadDiff[" * MOI.get(m.inner, MOI.SolverName()) * "]"
end

function MadDiff.empty_input_sensitivities!(model::Optimizer)
    empty!(model.forward.param_perturbations)
    empty!(model.reverse.primal_inputs)
    empty!(model.reverse.dual_inputs)
    return _clear_outputs!(model)
end

function _clear_outputs!(m::Optimizer{OT, T}) where {OT, T}
    empty!(m.forward.primal_sensitivities)
    empty!(m.forward.dual_sensitivities)
    empty!(m.reverse.param_outputs)
    return m.diff_time = zero(T)
end

function _invalidate_sensitivity!(m::Optimizer{OT, T}) where {OT, T}
    m.sensitivity_solver = nothing
    m.sensitivity_context = nothing
    return _clear_outputs!(m)
end

function _get_sensitivity_solver!(model::Optimizer)
    if isnothing(model.sensitivity_solver)
        ctx = _get_sensitivity_context!(model)
        model.sensitivity_solver = MadDiff.MadDiffSolver(
            model.inner.solver;
            config = model.sensitivity_config,
            param_pullback = _make_param_pullback_closure(model, ctx),
            n_params = ctx.n_p,
        )
        _populate_bound_idx_mappings!(model)
    end
    return model.sensitivity_solver
end

_kkt_to_moi_idx(cb::MadNLP.AbstractCallback, kkt_idx) = kkt_idx
_kkt_to_moi_idx(cb::MadNLP.SparseCallback{T,VT,VI,NLP,FH}, kkt_idx) where {T,VT,VI,NLP,FH<:MadNLP.MakeParameter} = cb.fixed_handler.free[kkt_idx]

function _populate_bound_idx_mappings!(model::Optimizer)
    sens = model.sensitivity_solver
    ctx = _get_sensitivity_context!(model)
    cb = sens.solver.cb
    empty!(model.vi_to_lb_idx)
    empty!(model.vi_to_ub_idx)
    dims = sens.dims
    for (i, kkt_idx) in enumerate(dims.ind_lb)
        kkt_idx > dims.n_x_kkt && continue  # skip slack bounds
        moi_idx = _kkt_to_moi_idx(cb, kkt_idx)
        vi = ctx.primal_vars[moi_idx]
        model.vi_to_lb_idx[vi] = i
    end
    for (i, kkt_idx) in enumerate(dims.ind_ub)
        kkt_idx > dims.n_x_kkt && continue  # skip slack bounds
        moi_idx = _kkt_to_moi_idx(cb, kkt_idx)
        vi = ctx.primal_vars[moi_idx]
        model.vi_to_ub_idx[vi] = i
    end
    return
end

function _get_sensitivity_context!(model::Optimizer{OT, T}) where {OT, T}
    if isnothing(model.sensitivity_context)
        model.sensitivity_context = SensitivityContext(model.inner; T)
    end
    return model.sensitivity_context
end

function _resize_and_zero!(cache::Vector{T}, n::Int) where {T}
    length(cache) != n && resize!(cache, n)
    fill!(cache, zero(T))
    return cache
end

_get_y_cache!(model::Optimizer, n::Int) = _resize_and_zero!(model.work.y_cache, n)
_get_dλ_cache!(model::Optimizer, n::Int) = _resize_and_zero!(model.work.dλ_cache, n)
_get_dL_dx!(model::Optimizer, n::Int) = _resize_and_zero!(model.work.dL_dx, n)
_get_dL_dλ!(model::Optimizer, n::Int) = _resize_and_zero!(model.work.dL_dλ, n)
_get_dL_dzl!(model::Optimizer, n::Int) = _resize_and_zero!(model.work.dL_dzl, n)
_get_dL_dzu!(model::Optimizer, n::Int) = _resize_and_zero!(model.work.dL_dzu, n)