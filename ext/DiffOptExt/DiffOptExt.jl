module DiffOptExt

import MadDiff
import DiffOpt
import MadNLP
const MOI = DiffOpt.MOI
const POI = DiffOpt.POI

MOIExt = Base.get_extension(MadDiff, :MathOptInterfaceExt)

mutable struct DiffOptModel <: MOI.AbstractOptimizer
    wrapper::Union{Nothing,MOIExt.DiffOptWrapper}
    source_to_inner::MOI.Utilities.IndexMap
    sensitivity_config::MadDiff.MadDiffConfig
end

function DiffOptModel(; sensitivity_config::MadDiff.MadDiffConfig = MadDiff.MadDiffConfig())
    return DiffOptModel(nothing, MOI.Utilities.IndexMap(), sensitivity_config)
end

_backend(model::DiffOptModel) = model.wrapper::MOIExt.DiffOptWrapper

MOI.supports_incremental_interface(::DiffOptModel) = true
MOI.supports_add_constrained_variable(::DiffOptModel, ::Type{<:MOI.AbstractScalarSet}) = true
MOI.supports_add_constrained_variables(::DiffOptModel, ::Type{<:MOI.AbstractVectorSet}) = true
MOI.supports_add_constrained_variables(::DiffOptModel, ::Type{MOI.Reals}) = true
MOI.supports_constraint(::DiffOptModel, ::Type{<:MOI.AbstractFunction}, ::Type{<:MOI.AbstractSet}) = true
MOI.is_empty(model::DiffOptModel) = isnothing(model.wrapper)

function MOI.empty!(model::DiffOptModel)
    model.wrapper = nothing
    model.source_to_inner = MOI.Utilities.IndexMap()
    return
end

function _madnlp_optimizer_type()
    if isdefined(MadNLP, :Optimizer)
        return getproperty(MadNLP, :Optimizer)
    end
    ext = Base.get_extension(MadNLP, :MathOptInterfaceExt)
    if isnothing(ext)
        return nothing
    end
    return getproperty(ext, :Optimizer)
end

function _is_madnlp_optimizer(optimizer)
    optimizer_type = _madnlp_optimizer_type()
    return !isnothing(optimizer_type) && optimizer isa optimizer_type
end

function _compose_index_maps(
    source_to_mid::MOI.Utilities.IndexMap,
    mid_to_dest::MOI.Utilities.IndexMap,
)
    output = MOI.Utilities.IndexMap()
    for (source, mid) in source_to_mid
        output[source] = mid_to_dest[mid]
    end
    return output
end

function _has_active_bridges(model::MOI.Bridges.LazyBridgeOptimizer)
    return !isempty(MOI.Bridges.Variable.bridges(model)) ||
           !isempty(MOI.Bridges.Constraint.bridges(model)) ||
           !isempty(MOI.Bridges.Objective.bridges(model))
end

function _unwrap_to_madnlp(
    root_optimizer,
    optimizer,
    source_to_optimizer::Union{Nothing,MOI.Utilities.IndexMap},
)
    _is_madnlp_optimizer(optimizer) &&
        return optimizer, something(source_to_optimizer, MOI.Utilities.identity_index_map(root_optimizer))
    error("MadDiff requires a wrapper chain ending in MadNLP. Got $(typeof(optimizer)).")
end

function _unwrap_to_madnlp(
    root_optimizer,
    optimizer::MOI.Utilities.CachingOptimizer,
    source_to_optimizer::Union{Nothing,MOI.Utilities.IndexMap},
)
    map_step = deepcopy(optimizer.model_to_optimizer_map)
    source_to_inner = isnothing(source_to_optimizer) ?
        map_step : _compose_index_maps(source_to_optimizer, map_step)
    return _unwrap_to_madnlp(root_optimizer, optimizer.optimizer, source_to_inner)
end

function _unwrap_to_madnlp(
    root_optimizer,
    optimizer::MOI.Bridges.LazyBridgeOptimizer,
    source_to_optimizer::Union{Nothing,MOI.Utilities.IndexMap},
)
    _has_active_bridges(optimizer) &&
        error("MadDiff does not support active MOI bridges in the DiffOpt chain.")
    return _unwrap_to_madnlp(root_optimizer, optimizer.model, source_to_optimizer)
end

function _unwrap_to_madnlp(
    root_optimizer,
    optimizer::POI.Optimizer,
    source_to_optimizer::Union{Nothing,MOI.Utilities.IndexMap},
)
    return _unwrap_to_madnlp(root_optimizer, optimizer.optimizer, source_to_optimizer)
end

_unwrap_to_madnlp(root_optimizer) = _unwrap_to_madnlp(root_optimizer, root_optimizer, nothing)

function _refresh_parameter_map!(
    optimizer::MOIExt.DiffOptWrapper,
    source_optimizer,
    source_to_madnlp::MOI.Utilities.IndexMap,
)
    empty!(optimizer.param_ci_to_vi)
    for source_ci in MOI.get(
        source_optimizer,
        MOI.ListOfConstraintIndices{
            MOI.VariableIndex,
            MOI.Parameter{Float64},
        }(),
    )
        source_vi = MOI.get(source_optimizer, MOI.ConstraintFunction(), source_ci)
        optimizer.param_ci_to_vi[source_to_madnlp[source_ci]] =
            source_to_madnlp[source_vi]
    end
    return optimizer
end

function MOI.copy_to(model::DiffOptModel, src::MOI.ModelLike)
    madnlp_optimizer, source_to_madnlp = _unwrap_to_madnlp(src)
    backend = MOIExt.DiffOptWrapper(madnlp_optimizer)
    backend.sensitivity_config = deepcopy(model.sensitivity_config)
    _refresh_parameter_map!(backend, src, source_to_madnlp)
    model.wrapper = backend
    model.source_to_inner = source_to_madnlp
    return MOI.Utilities.identity_index_map(src)
end

function MOI.copy_to(
    model::MOI.Bridges.LazyBridgeOptimizer{<:DiffOptModel},
    src::MOI.ModelLike,
)
    return MOI.copy_to(model.model, src)
end

_map_source_to_inner(model::DiffOptModel, idx) = model.source_to_inner[idx]

DiffOpt.forward_differentiate!(model::DiffOptModel) =
    MadDiff.forward_differentiate!(_backend(model))
DiffOpt.reverse_differentiate!(model::DiffOptModel) =
    MadDiff.reverse_differentiate!(_backend(model))
DiffOpt.empty_input_sensitivities!(model::DiffOptModel) =
    MadDiff.empty_input_sensitivities!(_backend(model))

MOI.supports(::DiffOptModel, ::DiffOpt.NonLinearKKTJacobianFactorization) = true
MOI.supports(::DiffOptModel, ::DiffOpt.AllowObjectiveAndSolutionInput) = true
MOI.set(::DiffOptModel, ::DiffOpt.NonLinearKKTJacobianFactorization, _) = nothing
MOI.set(::DiffOptModel, ::DiffOpt.AllowObjectiveAndSolutionInput, _) = nothing

function MOI.set(
    model::DiffOptModel,
    ::DiffOpt.ForwardConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    set::MOI.Parameter{T},
) where {T}
    inner_ci = _map_source_to_inner(model, ci)
    return MOI.set(_backend(model), MadDiff.ForwardConstraintSet(), inner_ci, set)
end

function MOI.get(
    model::DiffOptModel,
    ::DiffOpt.ForwardVariablePrimal,
    vi::MOI.VariableIndex,
)
    inner_vi = _map_source_to_inner(model, vi)
    return MOI.get(_backend(model), MadDiff.ForwardVariablePrimal(), inner_vi)
end

function MOI.get(
    model::DiffOptModel,
    ::DiffOpt.ForwardConstraintDual,
    ci::MOI.ConstraintIndex,
)
    inner_ci = _map_source_to_inner(model, ci)
    return MOI.get(_backend(model), MadDiff.ForwardConstraintDual(), inner_ci)
end

MOI.get(model::DiffOptModel, ::DiffOpt.ForwardObjectiveSensitivity) =
    MOI.get(_backend(model), MadDiff.ForwardObjectiveSensitivity())

function MOI.set(
    model::DiffOptModel,
    ::DiffOpt.ReverseVariablePrimal,
    vi::MOI.VariableIndex,
    value,
)
    inner_vi = _map_source_to_inner(model, vi)
    return MOI.set(_backend(model), MadDiff.ReverseVariablePrimal(), inner_vi, value)
end

function MOI.set(
    model::DiffOptModel,
    ::DiffOpt.ReverseConstraintDual,
    ci::MOI.ConstraintIndex,
    value,
)
    inner_ci = _map_source_to_inner(model, ci)
    return MOI.set(_backend(model), MadDiff.ReverseConstraintDual(), inner_ci, value)
end

MOI.set(model::DiffOptModel, ::DiffOpt.ReverseObjectiveSensitivity, value) =
    MOI.set(_backend(model), MadDiff.ReverseObjectiveSensitivity(), value)

function MOI.get(
    model::DiffOptModel,
    ::DiffOpt.ReverseConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    inner_ci = _map_source_to_inner(model, ci)
    return MOI.get(_backend(model), MadDiff.ReverseConstraintSet(), inner_ci)
end

MOI.get(model::DiffOptModel, ::DiffOpt.DifferentiateTimeSec) =
    MOI.get(_backend(model), MadDiff.DifferentiateTimeSec())

DiffOpt.get_reverse_parameter(
    model::DiffOptModel,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T} = MadDiff.get_reverse_parameter(
    _backend(model),
    _map_source_to_inner(model, ci),
)

MOI.get(model::DiffOptModel, ::MOI.SolverName) = "MadDiff[MadNLP]"

DiffOpt._copy_dual(::DiffOptModel, ::MOI.ModelLike, _) = nothing
DiffOpt._copy_dual(
    ::MOI.Bridges.LazyBridgeOptimizer{<:DiffOptModel},
    ::MOI.ModelLike,
    _,
) = nothing

"""
    MadDiff.diffopt_model_constructor(; config = MadDiff.MadDiffConfig())

Return a DiffOpt `ModelConstructor` callable that reuses a solved MadNLP
optimizer for differentiation.
"""
function MadDiff.diffopt_model_constructor(;
    config::MadDiff.MadDiffConfig = MadDiff.MadDiffConfig(),
)
    return () -> DiffOptModel(; sensitivity_config = config)
end

function MadDiff.diff_model(
    optimizer_constructor;
    config::MadDiff.MadDiffConfig = MadDiff.MadDiffConfig(),
    kwargs...,
)
    model = DiffOpt.diff_model(optimizer_constructor; kwargs...)
    MOI.set(model, DiffOpt.AllowObjectiveAndSolutionInput(), true)
    MOI.set(
        model,
        DiffOpt.ModelConstructor(),
        MadDiff.diffopt_model_constructor(config = config),
    )
    return model
end

end # module DiffOptExt
