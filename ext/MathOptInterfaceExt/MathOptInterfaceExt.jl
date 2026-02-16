module MathOptInterfaceExt

import MadDiff
import MadNLP; const NLPModels = MadNLP.NLPModels
import ParametricNLPModels
import MathOptInterface as MOI

mutable struct ForwardModeData{T}
    param_perturbations::Dict{MOI.ConstraintIndex, T}
    primal_sensitivities::Dict{MOI.VariableIndex, T}
    dual_sensitivities::Dict{MOI.ConstraintIndex, T}
    objective_sensitivity::T
end
ForwardModeData{T}() where {T} = ForwardModeData{T}(
    Dict{MOI.ConstraintIndex, T}(),
    Dict{MOI.VariableIndex, T}(),
    Dict{MOI.ConstraintIndex, T}(),
    zero(T),
)

mutable struct ReverseModeData{T}
    primal_seeds::Dict{MOI.VariableIndex, T}
    dual_seeds::Dict{MOI.ConstraintIndex, T}
    param_outputs::Dict{MOI.ConstraintIndex, T}
    dobj::Union{Nothing, T}
end
ReverseModeData{T}() where {T} = ReverseModeData{T}(
    Dict{MOI.VariableIndex, T}(),
    Dict{MOI.ConstraintIndex, T}(),
    Dict{MOI.ConstraintIndex, T}(),
    nothing,
)

mutable struct WorkBuffers{T}
    y_cache::Vector{T}
    dy_cache::Vector{T}
    dL_dx::Vector{T}
    dL_dy::Vector{T}
    dL_dzl::Vector{T}
    dL_dzu::Vector{T}
end
WorkBuffers{T}() where {T} = WorkBuffers{T}(
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
    forward::ForwardModeData{T}
    reverse::ReverseModeData{T}
    work::WorkBuffers{T}
    sensitivity_config::MadDiff.MadDiffConfig
    sensitivity_solver::Union{Nothing, MadDiff.MadDiffSolver}
    diff_time::T
end

function Optimizer(inner::OT; T::Type = Float64) where {OT <: MOI.AbstractOptimizer}
    return Optimizer{OT, T}(
        inner,
        Dict{MOI.ConstraintIndex, MOI.VariableIndex}(),
        ForwardModeData{T}(),
        ReverseModeData{T}(),
        WorkBuffers{T}(),
        MadDiff.MadDiffConfig(),
        nothing,
        zero(T),
    )
end

function MadDiff.diff_optimizer(optimizer_constructor; kwargs...)
    return () -> Optimizer(optimizer_constructor(; kwargs...))
end

include("moi_wrapper.jl")
include("forward_mode.jl")
include("reverse_mode.jl")

end # module
