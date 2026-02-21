MOI.get(wrapper::DiffOptWrapper, ::MadDiff.DifferentiateTimeSec) = wrapper.diff_time

function MadDiff.empty_input_sensitivities!(wrapper::DiffOptWrapper)
    empty!(wrapper.forward.param_perturbations)
    empty!(wrapper.reverse.primal_seeds)
    empty!(wrapper.reverse.dual_seeds)
    wrapper.reverse.dobj = nothing
    return _clear_outputs!(wrapper)
end

function _clear_outputs!(wrapper::DiffOptWrapper{OT, T}) where {OT, T}
    empty!(wrapper.forward.primal_sensitivities)
    empty!(wrapper.forward.dual_sensitivities)
    wrapper.forward.objective_sensitivity = zero(T)
    empty!(wrapper.reverse.param_outputs)
    return wrapper.diff_time = zero(T)
end

function _invalidate_factorization!(wrapper::DiffOptWrapper{OT, T}) where {OT, T}
    wrapper.sensitivity_solver = nothing
    return _clear_outputs!(m)
end

function _invalidate_sensitivity!(wrapper::DiffOptWrapper{OT, T}) where {OT, T}
    return _invalidate_factorization!(wrapper)
end

function _get_sensitivity_solver!(wrapper::DiffOptWrapper)
    if isnothing(wrapper.sensitivity_solver)
        wrapper.sensitivity_solver = MadDiff.MadDiffSolver(wrapper.inner.solver; config = wrapper.sensitivity_config)
    end
    return wrapper.sensitivity_solver
end

function _resize_and_zero!(cache::Vector{T}, n::Int) where {T}
    length(cache) != n && resize!(cache, n)
    fill!(cache, zero(T))
    return cache
end

_get_dy_cache!(wrapper::DiffOptWrapper, n::Int) = _resize_and_zero!(wrapper.work.dy_cache, n)
_get_dL_dx!(wrapper::DiffOptWrapper, n::Int) = _resize_and_zero!(wrapper.work.dL_dx, n)
_get_dL_dy!(wrapper::DiffOptWrapper, n::Int) = _resize_and_zero!(wrapper.work.dL_dy, n)
_get_dL_dzl!(wrapper::DiffOptWrapper, n::Int) = _resize_and_zero!(wrapper.work.dL_dzl, n)
_get_dL_dzu!(wrapper::DiffOptWrapper, n::Int) = _resize_and_zero!(wrapper.work.dL_dzu, n)