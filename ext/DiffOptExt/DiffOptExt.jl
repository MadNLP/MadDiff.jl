module DiffOptExt

import MadDiff
import DiffOpt
import MadNLP
const MOI = DiffOpt.MOI
const POI = DiffOpt.POI

MOIExt = Base.get_extension(MadDiff, :MathOptInterfaceExt)

mutable struct MadDiffOptimizer{OT<:MOI.ModelLike} <: MOI.AbstractOptimizer
    inner::OT
    wrapper::Union{Nothing,MOIExt.DiffOptWrapper}
    source_to_inner::MOI.Utilities.IndexMap
    sensitivity_config::MadDiff.MadDiffConfig
end

function MadDiffOptimizer(
    inner::OT;
    config::MadDiff.MadDiffConfig = MadDiff.MadDiffConfig(),
) where {OT<:MOI.ModelLike}
    return MadDiffOptimizer{OT}(
        inner,
        nothing,
        MOI.Utilities.IndexMap(),
        config,
    )
end

function _ensure_backend!(model::MadDiffOptimizer)
    if isnothing(model.wrapper)
        madnlp_optimizer, source_to_madnlp = _unwrap_to_madnlp(model.inner)
        backend = MOIExt.DiffOptWrapper(madnlp_optimizer)
        backend.sensitivity_config = deepcopy(model.sensitivity_config)
        _refresh_parameter_map!(backend, model.inner, source_to_madnlp)
        model.wrapper = backend
        model.source_to_inner = source_to_madnlp
    end
    return model.wrapper
end

_backend(model::MadDiffOptimizer) = _ensure_backend!(model)

_map_source_to_inner(model::MadDiffOptimizer, idx) = model.source_to_inner[idx]

# ============================================================================
# Standard MOI forwarding
# ============================================================================

# Structural operations
function MOI.empty!(model::MadDiffOptimizer)
    MOI.empty!(model.inner)
    model.wrapper = nothing
    model.source_to_inner = MOI.Utilities.IndexMap()
    return
end

MOI.is_empty(model::MadDiffOptimizer) = MOI.is_empty(model.inner)

function MOI.optimize!(model::MadDiffOptimizer)
    model.wrapper = nothing
    return MOI.optimize!(model.inner)
end

MOI.add_variable(model::MadDiffOptimizer) = MOI.add_variable(model.inner)
MOI.add_variables(model::MadDiffOptimizer, n) = MOI.add_variables(model.inner, n)

function MOI.add_constraint(
    model::MadDiffOptimizer,
    f::MOI.AbstractFunction,
    s::MOI.AbstractSet,
)
    return MOI.add_constraint(model.inner, f, s)
end

MOI.delete(model::MadDiffOptimizer, vi::MOI.VariableIndex) = MOI.delete(model.inner, vi)
MOI.delete(model::MadDiffOptimizer, ci::MOI.ConstraintIndex) = MOI.delete(model.inner, ci)
MOI.is_valid(model::MadDiffOptimizer, vi::MOI.VariableIndex) = MOI.is_valid(model.inner, vi)
MOI.is_valid(model::MadDiffOptimizer, ci::MOI.ConstraintIndex) = MOI.is_valid(model.inner, ci)

function MOI.modify(
    model::MadDiffOptimizer,
    ci::MOI.ConstraintIndex,
    chg::MOI.AbstractFunctionModification,
)
    return MOI.modify(model.inner, ci, chg)
end

function MOI.supports_incremental_interface(model::MadDiffOptimizer)
    return MOI.supports_incremental_interface(model.inner)
end

function MOI.copy_to(model::MadDiffOptimizer, src::MOI.ModelLike)
    return MOI.copy_to(model.inner, src)
end

function MOI.supports_constraint(
    model::MadDiffOptimizer,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports_constraint(model.inner, F, S)
end

function MOI.supports_add_constrained_variable(
    model::MadDiffOptimizer,
    ::Type{S},
) where {S<:MOI.AbstractScalarSet}
    return MOI.supports_add_constrained_variable(model.inner, S)
end

function MOI.supports_add_constrained_variables(
    model::MadDiffOptimizer,
    ::Type{S},
) where {S<:MOI.AbstractVectorSet}
    return MOI.supports_add_constrained_variables(model.inner, S)
end

# Model attributes
MOI.get(model::MadDiffOptimizer, attr::MOI.AbstractModelAttribute) = MOI.get(model.inner, attr)
MOI.set(model::MadDiffOptimizer, attr::MOI.AbstractModelAttribute, value) = MOI.set(model.inner, attr, value)
MOI.supports(model::MadDiffOptimizer, attr::MOI.AbstractModelAttribute) = MOI.supports(model.inner, attr)

# Optimizer attributes
MOI.get(model::MadDiffOptimizer, attr::MOI.AbstractOptimizerAttribute) = MOI.get(model.inner, attr)
MOI.set(model::MadDiffOptimizer, attr::MOI.AbstractOptimizerAttribute, value) = MOI.set(model.inner, attr, value)
MOI.supports(model::MadDiffOptimizer, attr::MOI.AbstractOptimizerAttribute) = MOI.supports(model.inner, attr)

# Variable attributes
function MOI.get(model::MadDiffOptimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex)
    return MOI.get(model.inner, attr, vi)
end

function MOI.get(model::MadDiffOptimizer, attr::MOI.AbstractVariableAttribute, vis::Vector{MOI.VariableIndex})
    return MOI.get(model.inner, attr, vis)
end

function MOI.set(model::MadDiffOptimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex, value)
    return MOI.set(model.inner, attr, vi, value)
end

function MOI.supports(model::MadDiffOptimizer, attr::MOI.AbstractVariableAttribute, ::Type{MOI.VariableIndex})
    return MOI.supports(model.inner, attr, MOI.VariableIndex)
end

# Constraint attributes
function MOI.get(model::MadDiffOptimizer, attr::MOI.AbstractConstraintAttribute, ci::MOI.ConstraintIndex)
    return MOI.get(model.inner, attr, ci)
end

function MOI.set(model::MadDiffOptimizer, attr::MOI.AbstractConstraintAttribute, ci::MOI.ConstraintIndex, value)
    return MOI.set(model.inner, attr, ci, value)
end

function MOI.supports(
    model::MadDiffOptimizer,
    attr::MOI.AbstractConstraintAttribute,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F,S}
    return MOI.supports(model.inner, attr, MOI.ConstraintIndex{F,S})
end

# SolverName
MOI.get(::MadDiffOptimizer, ::MOI.SolverName) = "MadDiff[MadNLP]"

# ============================================================================
# DiffOpt native differentiation support
# ============================================================================

# Support native differentiation
MOI.supports(::MadDiffOptimizer, ::DiffOpt.BackwardDifferentiate) = true
MOI.supports(::MadDiffOptimizer, ::DiffOpt.ForwardDifferentiate) = true

# Trigger differentiation
function MOI.set(model::MadDiffOptimizer, ::DiffOpt.BackwardDifferentiate, ::Nothing)
    MadDiff.reverse_differentiate!(_backend(model))
    return
end

function MOI.set(model::MadDiffOptimizer, ::DiffOpt.ForwardDifferentiate, ::Nothing)
    MadDiff.forward_differentiate!(_backend(model))
    return
end

# Input attributes: reverse mode
function MOI.set(
    model::MadDiffOptimizer,
    ::DiffOpt.ReverseVariablePrimal,
    vi::MOI.VariableIndex,
    value,
)
    inner_vi = _map_source_to_inner(model, vi)
    return MOI.set(_backend(model), MadDiff.ReverseVariablePrimal(), inner_vi, value)
end

function MOI.supports(
    ::MadDiffOptimizer,
    ::DiffOpt.ReverseVariablePrimal,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    model::MadDiffOptimizer,
    ::DiffOpt.ReverseConstraintDual,
    ci::MOI.ConstraintIndex,
    value,
)
    inner_ci = _map_source_to_inner(model, ci)
    return MOI.set(_backend(model), MadDiff.ReverseConstraintDual(), inner_ci, value)
end

function MOI.set(model::MadDiffOptimizer, ::DiffOpt.ReverseObjectiveSensitivity, value)
    return MOI.set(_backend(model), MadDiff.ReverseObjectiveSensitivity(), value)
end

# Input attributes: forward mode
function MOI.set(
    model::MadDiffOptimizer,
    ::DiffOpt.ForwardConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    set::MOI.Parameter{T},
) where {T}
    inner_ci = _map_source_to_inner(model, ci)
    return MOI.set(_backend(model), MadDiff.ForwardConstraintSet(), inner_ci, set)
end

# Output attributes: reverse mode
function MOI.get(
    model::MadDiffOptimizer,
    ::DiffOpt.ReverseConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    inner_ci = _map_source_to_inner(model, ci)
    return MOI.get(_backend(model), MadDiff.ReverseConstraintSet(), inner_ci)
end

# Output attributes: forward mode
function MOI.get(
    model::MadDiffOptimizer,
    ::DiffOpt.ForwardVariablePrimal,
    vi::MOI.VariableIndex,
)
    inner_vi = _map_source_to_inner(model, vi)
    return MOI.get(_backend(model), MadDiff.ForwardVariablePrimal(), inner_vi)
end

function MOI.get(
    model::MadDiffOptimizer,
    ::DiffOpt.ForwardConstraintDual,
    ci::MOI.ConstraintIndex,
)
    inner_ci = _map_source_to_inner(model, ci)
    return MOI.get(_backend(model), MadDiff.ForwardConstraintDual(), inner_ci)
end

function MOI.get(model::MadDiffOptimizer, ::DiffOpt.ForwardObjectiveSensitivity)
    return MOI.get(_backend(model), MadDiff.ForwardObjectiveSensitivity())
end

# DifferentiateTimeSec
function MOI.get(model::MadDiffOptimizer, ::DiffOpt.DifferentiateTimeSec)
    return MOI.get(_backend(model), MadDiff.DifferentiateTimeSec())
end

# Ignored attributes (not needed for MadDiff)
MOI.supports(::MadDiffOptimizer, ::DiffOpt.NonLinearKKTJacobianFactorization) = true
MOI.supports(::MadDiffOptimizer, ::DiffOpt.AllowObjectiveAndSolutionInput) = true
MOI.set(::MadDiffOptimizer, ::DiffOpt.NonLinearKKTJacobianFactorization, _) = nothing
MOI.set(::MadDiffOptimizer, ::DiffOpt.AllowObjectiveAndSolutionInput, _) = nothing

# empty_input_sensitivities!
function DiffOpt.empty_input_sensitivities!(model::MadDiffOptimizer)
    if !isnothing(model.wrapper)
        MadDiff.empty_input_sensitivities!(model.wrapper)
    end
    return
end

# get_reverse_parameter
function DiffOpt.get_reverse_parameter(
    model::MadDiffOptimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    inner_ci = _map_source_to_inner(model, ci)
    return MadDiff.get_reverse_parameter(_backend(model), inner_ci)
end

# ============================================================================
# Unwrapping to find MadNLP
# ============================================================================

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

# ============================================================================
# Public API
# ============================================================================

function MadDiff.diff_optimizer(
    optimizer_constructor;
    config::MadDiff.MadDiffConfig = MadDiff.MadDiffConfig(),
    kwargs...,
)
    # Build the standard optimizer chain
    inner = DiffOpt.diff_optimizer(optimizer_constructor; kwargs...)
    # Wrap inner.optimizer with MadDiffOptimizer
    mad_opt = MadDiffOptimizer(inner.optimizer; config)
    return DiffOpt.Optimizer(mad_opt)
end

end # module DiffOptExt
