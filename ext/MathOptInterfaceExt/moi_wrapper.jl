# ============================================================================
# Cache/lifecycle helpers for `DiffOptWrapper`.
#
# `empty_input_sensitivities!` clears user-set seeds and drops cached outputs
# but keeps the KKT factorization. `_clear_outputs!` alone drops outputs when
# a seed change invalidates them; it is lazy-triggered by the `outputs_dirty`
# flag flipped by the individual MOI setters.
# ============================================================================

MOI.get(wrapper::DiffOptWrapper, ::MadDiff.DifferentiateTimeSec) = wrapper.diff_time

function MadDiff.empty_input_sensitivities!(wrapper::DiffOptWrapper)
    empty!(wrapper.forward.param_perturbations)
    empty!(wrapper.reverse.primal_seeds)
    empty!(wrapper.reverse.dual_seeds)
    wrapper.reverse.dobj = nothing
    _clear_outputs!(wrapper)
    return wrapper
end

function _clear_outputs!(wrapper::DiffOptWrapper{OT, T}) where {OT, T}
    empty!(wrapper.forward.primal_sensitivities)
    empty!(wrapper.forward.dual_sensitivities)
    wrapper.forward.objective_sensitivity = nothing
    wrapper.forward.jvp_result = nothing
    wrapper.forward.param_direction = nothing
    empty!(wrapper.reverse.param_outputs)
    wrapper.diff_time = zero(T)
    wrapper.outputs_dirty = false
    return wrapper
end

function _get_sensitivity_solver!(wrapper::DiffOptWrapper)
    wrapper.sensitivity_solver === nothing || return wrapper.sensitivity_solver
    wrapper.sensitivity_solver = MadDiff.MadDiffSolver(
        wrapper.inner.solver; config = wrapper.sensitivity_config,
    )
    return wrapper.sensitivity_solver
end

function _scratch!(buf::AbstractVector{T}, n::Int) where {T}
    length(buf) == n || resize!(buf, n)
    fill!(buf, zero(T))
    return buf
end
