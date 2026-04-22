module MathOptInterfaceExt

# ============================================================================
# MadDiff × MathOptInterface.
#
# `DiffOptWrapper` adapts a solved `MadNLP.Optimizer` to the MadDiff MOI
# attribute surface (`Forward*`, `Reverse*`). The DiffOpt extension sits on
# top and forwards DiffOpt's matching attributes to ours.
# ============================================================================

import MadDiff
import MadNLP
import MathOptInterface as MOI

const NLPModels = MadNLP.NLPModels

# ---------- per-mode state ----------

mutable struct ForwardModeData{T}
    param_perturbations::Dict{MOI.ConstraintIndex, T}
    primal_sensitivities::Dict{MOI.VariableIndex, T}
    dual_sensitivities::Dict{MOI.ConstraintIndex, T}
    objective_sensitivity::Union{Nothing, T}
    # Populated on first `forward_differentiate!`; VT (device vector type) is
    # only known then, so store without a VT parameter. Nothing/Union{} keeps
    # Julia's field-tag specialisation cheap — much better than `::Any`.
    jvp_result::Union{Nothing, MadDiff.JVPResult}
    param_direction::Union{Nothing, AbstractVector{T}}
end

ForwardModeData{T}() where {T} = ForwardModeData{T}(
    Dict{MOI.ConstraintIndex, T}(), Dict{MOI.VariableIndex, T}(),
    Dict{MOI.ConstraintIndex, T}(), nothing, nothing, nothing,
)

mutable struct ReverseModeData{T}
    primal_seeds::Dict{MOI.VariableIndex, T}
    dual_seeds::Dict{MOI.ConstraintIndex, T}
    param_outputs::Dict{MOI.ConstraintIndex, T}
    dobj::Union{Nothing, T}
end

ReverseModeData{T}() where {T} = ReverseModeData{T}(
    Dict{MOI.VariableIndex, T}(), Dict{MOI.ConstraintIndex, T}(),
    Dict{MOI.ConstraintIndex, T}(), nothing,
)

# Resizeable scratch for the CPU-side staging buffers used to marshal seeds
# into MadDiff and unmarshal results back. All buffers are reused across
# `forward_differentiate!` / `reverse_differentiate!` calls to avoid per-call
# `zeros(T, n)` allocations on the happy path.
mutable struct WorkBuffers{T}
    dp::Vector{T}        # forward:  dense perturbation Δp
    dy::Vector{T}        # forward:  sign-adjusted dual sensitivities
    dL_dx::Vector{T}     # reverse:  primal seeds staging
    dL_dy::Vector{T}     # reverse:  dual  seeds staging
    dL_dzl::Vector{T}    # reverse:  lower-bound dual seeds staging
    dL_dzu::Vector{T}    # reverse:  upper-bound dual seeds staging
end

WorkBuffers{T}() where {T} = WorkBuffers{T}(
    T[], T[], T[], T[], T[], T[],
)

# ---------- wrapper ----------

mutable struct DiffOptWrapper{OT <: MOI.AbstractOptimizer, T}
    inner::OT
    param_ci_to_vi::Dict{MOI.ConstraintIndex, MOI.VariableIndex}
    forward::ForwardModeData{T}
    reverse::ReverseModeData{T}
    work::WorkBuffers{T}
    sensitivity_config::MadDiff.MadDiffConfig
    sensitivity_solver::Union{Nothing, MadDiff.MadDiffSolver}
    diff_time::T
    # `true` whenever a seed has been set since the last `_clear_outputs!`.
    # Batched setters mark dirty; `*_differentiate!` flushes once before solving.
    outputs_dirty::Bool
end

DiffOptWrapper(inner::OT; T::Type = Float64) where {OT <: MOI.AbstractOptimizer} =
    DiffOptWrapper{OT, T}(
        inner, Dict{MOI.ConstraintIndex, MOI.VariableIndex}(),
        ForwardModeData{T}(), ReverseModeData{T}(), WorkBuffers{T}(),
        MadDiff.MadDiffConfig(), nothing, zero(T), false,
    )

@inline _mark_outputs_dirty!(wrapper::DiffOptWrapper) =
    (wrapper.outputs_dirty = true; nothing)
@inline _outputs_dirty(wrapper::DiffOptWrapper) = wrapper.outputs_dirty

include("moi_wrapper.jl")
include("forward_mode.jl")
include("reverse_mode.jl")

end # module
