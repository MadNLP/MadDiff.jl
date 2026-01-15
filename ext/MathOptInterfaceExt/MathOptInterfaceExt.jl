module MathOptInterfaceExt

import MadDiff
import MadNLP; const NLPModels = MadNLP.NLPModels
import MathOptInterface as MOI

include("moi_evaluator.jl")

mutable struct ForwardModeIO{T}
    param_perturbations::Dict{MOI.ConstraintIndex, T}
    primal_sensitivities::Dict{MOI.VariableIndex, T}
    dual_sensitivities::Dict{MOI.ConstraintIndex, T}
end
ForwardModeIO{T}() where {T} = ForwardModeIO{T}(
    Dict{MOI.ConstraintIndex, T}(),
    Dict{MOI.VariableIndex, T}(),
    Dict{MOI.ConstraintIndex, T}(),
)

mutable struct ReverseModeIO{T}
    primal_inputs::Dict{MOI.VariableIndex, T}
    dual_inputs::Dict{MOI.ConstraintIndex, T}
    param_outputs::Dict{MOI.ConstraintIndex, T}
end
ReverseModeIO{T}() where {T} = ReverseModeIO{T}(
    Dict{MOI.VariableIndex, T}(),
    Dict{MOI.ConstraintIndex, T}(),
    Dict{MOI.ConstraintIndex, T}(),
)

mutable struct WorkVectors{T}
    y_cache::Vector{T}
    dλ_cache::Vector{T}
    dL_dx::Vector{T}
    dL_dλ::Vector{T}
    dL_dzl::Vector{T}
    dL_dzu::Vector{T}
end
WorkVectors{T}() where {T} = WorkVectors{T}(
    Vector{T}(undef, 0),
    Vector{T}(undef, 0),
    Vector{T}(undef, 0),
    Vector{T}(undef, 0),
    Vector{T}(undef, 0),
    Vector{T}(undef, 0),
)

mutable struct Optimizer{OT <: MOI.AbstractOptimizer, T} <: MOI.AbstractOptimizer
    inner::OT
    param_ci_to_vi::Dict{MOI.ConstraintIndex, MOI.VariableIndex}
    forward::ForwardModeIO{T}
    reverse::ReverseModeIO{T}
    work::WorkVectors{T}
    sensitivity_config::MadDiff.MadDiffConfig
    sensitivity_solver::Union{Nothing, MadDiff.MadDiffSolver}
    sensitivity_context::Union{Nothing, SensitivityContext}
    vi_to_lb_idx::Dict{MOI.VariableIndex, Int}
    vi_to_ub_idx::Dict{MOI.VariableIndex, Int}
    diff_time::T
end

function Optimizer(inner::OT; T::Type = Float64) where {OT <: MOI.AbstractOptimizer}
    return Optimizer{OT, T}(
        inner,
        Dict{MOI.ConstraintIndex, MOI.VariableIndex}(),
        ForwardModeIO{T}(),
        ReverseModeIO{T}(),
        WorkVectors{T}(),
        MadDiff.MadDiffConfig(),
        nothing,
        nothing,
        Dict{MOI.VariableIndex, Int}(),
        Dict{MOI.VariableIndex, Int}(),
        zero(T),
    )
end

function MadDiff.diff_optimizer(optimizer_constructor; kwargs...)
    return () -> Optimizer(optimizer_constructor(; kwargs...))
end

include("moi_wrapper.jl")
include("moi_jvp.jl")
include("moi_vjp.jl")
include("forward_mode.jl")
include("reverse_mode.jl")

end # module
