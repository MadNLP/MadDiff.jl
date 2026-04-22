module DiffOptExt

# ============================================================================
# MadDiff × DiffOpt.
#
# `DiffOptModel` is the inner solver DiffOpt instantiates (via its
# `ModelConstructor`) on top of a solved MadNLP. It re-routes DiffOpt's
# attribute-based API (`ForwardDifferentiate`, `ReverseDifferentiate`, the
# sensitivity getters/setters) to MadDiff's MOI extension, which runs the
# actual JVP/VJP kernels.
#
# The key wiring is `_unwrap_to_madnlp`: walk DiffOpt's layered optimizer
# (caching → bridges → POI → MadNLP), compose the index maps, and yield the
# MadNLP optimizer plus a source→MadNLP index map.
# ============================================================================

import DiffOpt
import MadDiff
import MadNLP

const MOI     = DiffOpt.MOI
const POI     = DiffOpt.POI
const MOIExt  = Base.get_extension(MadDiff, :MathOptInterfaceExt)

# ---------- DiffOptModel ----------

mutable struct DiffOptModel <: MOI.AbstractOptimizer
    wrapper::Union{Nothing, MOIExt.DiffOptWrapper}
    source_to_inner::MOI.Utilities.IndexMap
    sensitivity_config::MadDiff.MadDiffConfig
end

DiffOptModel(; sensitivity_config::MadDiff.MadDiffConfig = MadDiff.MadDiffConfig()) =
    DiffOptModel(nothing, MOI.Utilities.IndexMap(), sensitivity_config)

_backend(model::DiffOptModel) = model.wrapper::MOIExt.DiffOptWrapper
_map(model::DiffOptModel, idx) = model.source_to_inner[idx]

# ---------- MOI plumbing ----------

MOI.supports_incremental_interface(::DiffOptModel) = true
MOI.supports_add_constrained_variable(::DiffOptModel, ::Type{<:MOI.AbstractScalarSet}) = true
MOI.supports_add_constrained_variables(::DiffOptModel, ::Type{<:MOI.AbstractVectorSet}) = true
MOI.supports_add_constrained_variables(::DiffOptModel, ::Type{MOI.Reals}) = true
MOI.supports_constraint(::DiffOptModel, ::Type{<:MOI.AbstractFunction}, ::Type{<:MOI.AbstractSet}) = true
MOI.is_empty(model::DiffOptModel) = model.wrapper === nothing

function MOI.empty!(model::DiffOptModel)
    model.wrapper = nothing
    model.source_to_inner = MOI.Utilities.IndexMap()
    return nothing
end

MOI.get(::DiffOptModel, ::MOI.SolverName) = "MadDiff[MadNLP]"

# ---------- MadNLP unwrapping ----------

function _madnlp_optimizer_type()
    isdefined(MadNLP, :Optimizer) && return getproperty(MadNLP, :Optimizer)
    ext = Base.get_extension(MadNLP, :MathOptInterfaceExt)
    ext === nothing ? nothing : getproperty(ext, :Optimizer)
end

_is_madnlp_optimizer(opt) =
    let T = _madnlp_optimizer_type(); T !== nothing && opt isa T end

function _compose(source_to_mid::MOI.Utilities.IndexMap,
                   mid_to_dest::MOI.Utilities.IndexMap)
    out = MOI.Utilities.IndexMap()
    for (src, mid) in source_to_mid
        out[src] = mid_to_dest[mid]
    end
    return out
end

_has_active_bridges(m::MOI.Bridges.LazyBridgeOptimizer) =
    !isempty(MOI.Bridges.Variable.bridges(m))  ||
    !isempty(MOI.Bridges.Constraint.bridges(m)) ||
    !isempty(MOI.Bridges.Objective.bridges(m))

function _unwrap_to_madnlp(root, optimizer, src_to_opt)
    _is_madnlp_optimizer(optimizer) &&
        return optimizer, something(src_to_opt, MOI.Utilities.identity_index_map(root))
    error("MadDiff: DiffOpt chain must end in MadNLP; got $(typeof(optimizer)).")
end

function _unwrap_to_madnlp(root, optimizer::MOI.Utilities.CachingOptimizer, src_to_opt)
    step = deepcopy(optimizer.model_to_optimizer_map)
    next = src_to_opt === nothing ? step : _compose(src_to_opt, step)
    return _unwrap_to_madnlp(root, optimizer.optimizer, next)
end

function _unwrap_to_madnlp(root, optimizer::MOI.Bridges.LazyBridgeOptimizer, src_to_opt)
    _has_active_bridges(optimizer) &&
        error("MadDiff: active MOI bridges in the DiffOpt chain are not supported.")
    return _unwrap_to_madnlp(root, optimizer.model, src_to_opt)
end

_unwrap_to_madnlp(root, optimizer::POI.Optimizer, src_to_opt) =
    _unwrap_to_madnlp(root, optimizer.optimizer, src_to_opt)

_unwrap_to_madnlp(root) = _unwrap_to_madnlp(root, root, nothing)

function _refresh_parameter_map!(wrapper::MOIExt.DiffOptWrapper, src, src_to_mad)
    empty!(wrapper.param_ci_to_vi)
    for src_ci in MOI.get(src, MOI.ListOfConstraintIndices{
            MOI.VariableIndex, MOI.Parameter{Float64}}())
        src_vi = MOI.get(src, MOI.ConstraintFunction(), src_ci)
        wrapper.param_ci_to_vi[src_to_mad[src_ci]] = src_to_mad[src_vi]
    end
    return wrapper
end

# ---------- copy_to: install the MadNLP-backed wrapper ----------

function MOI.copy_to(model::DiffOptModel, src::MOI.ModelLike)
    madnlp, src_to_mad = _unwrap_to_madnlp(src)
    wrapper = MOIExt.DiffOptWrapper(madnlp)
    wrapper.sensitivity_config = deepcopy(model.sensitivity_config)
    _refresh_parameter_map!(wrapper, src, src_to_mad)
    model.wrapper = wrapper
    model.source_to_inner = src_to_mad
    return MOI.Utilities.identity_index_map(src)
end

MOI.copy_to(model::MOI.Bridges.LazyBridgeOptimizer{<:DiffOptModel},
             src::MOI.ModelLike) = MOI.copy_to(model.model, src)

# ---------- trigger attributes ----------
#
# DiffOpt now routes `forward_differentiate!(::JuMP.Model)` through
# `MOI.set(model, DiffOpt.ForwardDifferentiate(), nothing)`; same for reverse.

MOI.supports(::DiffOptModel, ::DiffOpt.ForwardDifferentiate)           = true
MOI.supports(::DiffOptModel, ::DiffOpt.ReverseDifferentiate)           = true
MOI.supports(::DiffOptModel, ::DiffOpt.NonLinearKKTJacobianFactorization) = true
MOI.supports(::DiffOptModel, ::DiffOpt.AllowObjectiveAndSolutionInput)   = true

MOI.set(m::DiffOptModel, ::DiffOpt.ForwardDifferentiate, ::Nothing) =
    (MadDiff.forward_differentiate!(_backend(m)); nothing)
MOI.set(m::DiffOptModel, ::DiffOpt.ReverseDifferentiate, ::Nothing) =
    (MadDiff.reverse_differentiate!(_backend(m)); nothing)
MOI.set(::DiffOptModel, ::DiffOpt.NonLinearKKTJacobianFactorization, _) = nothing
MOI.set(::DiffOptModel, ::DiffOpt.AllowObjectiveAndSolutionInput,   _) = nothing

# Direct function-level dispatch (used by backend callers; redundant once the
# attribute triggers above exist, but kept for the direct API).
DiffOpt.forward_differentiate!(m::DiffOptModel)     = MadDiff.forward_differentiate!(_backend(m))
DiffOpt.reverse_differentiate!(m::DiffOptModel)     = MadDiff.reverse_differentiate!(_backend(m))
DiffOpt.empty_input_sensitivities!(m::DiffOptModel) = MadDiff.empty_input_sensitivities!(_backend(m))

# ---------- sensitivity in/out: forward ----------

MOI.set(m::DiffOptModel, ::DiffOpt.ForwardConstraintSet,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}},
        set::MOI.Parameter{T}) where {T} =
    MOI.set(_backend(m), MadDiff.ForwardConstraintSet(), _map(m, ci), set)

MOI.get(m::DiffOptModel, ::DiffOpt.ForwardVariablePrimal, vi::MOI.VariableIndex) =
    MOI.get(_backend(m), MadDiff.ForwardVariablePrimal(), _map(m, vi))

MOI.get(m::DiffOptModel, ::DiffOpt.ForwardConstraintDual, ci::MOI.ConstraintIndex) =
    MOI.get(_backend(m), MadDiff.ForwardConstraintDual(), _map(m, ci))

MOI.get(m::DiffOptModel, ::DiffOpt.ForwardObjectiveSensitivity) =
    MOI.get(_backend(m), MadDiff.ForwardObjectiveSensitivity())

# ---------- sensitivity in/out: reverse ----------

MOI.set(m::DiffOptModel, ::DiffOpt.ReverseVariablePrimal,
        vi::MOI.VariableIndex, value) =
    MOI.set(_backend(m), MadDiff.ReverseVariablePrimal(), _map(m, vi), value)

MOI.set(m::DiffOptModel, ::DiffOpt.ReverseConstraintDual,
        ci::MOI.ConstraintIndex, value) =
    MOI.set(_backend(m), MadDiff.ReverseConstraintDual(), _map(m, ci), value)

MOI.set(m::DiffOptModel, ::DiffOpt.ReverseObjectiveSensitivity, value) =
    MOI.set(_backend(m), MadDiff.ReverseObjectiveSensitivity(), value)

MOI.get(m::DiffOptModel, ::DiffOpt.ReverseConstraintSet,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}}) where {T} =
    MOI.get(_backend(m), MadDiff.ReverseConstraintSet(), _map(m, ci))

DiffOpt.get_reverse_parameter(m::DiffOptModel,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}}) where {T} =
    MadDiff.get_reverse_parameter(_backend(m), _map(m, ci))

MOI.get(m::DiffOptModel, ::DiffOpt.DifferentiateTimeSec) =
    MOI.get(_backend(m), MadDiff.DifferentiateTimeSec())

# DiffOpt attempts to round-trip duals; our wrapper doesn't need that.
DiffOpt._copy_dual(::DiffOptModel, ::MOI.ModelLike, _) = nothing
DiffOpt._copy_dual(::MOI.Bridges.LazyBridgeOptimizer{<:DiffOptModel},
                   ::MOI.ModelLike, _) = nothing

# ---------- public entry points ----------

"""
    MadDiff.diffopt_model_constructor(; config = MadDiffConfig())

Return a `DiffOpt.ModelConstructor` callable that installs a `DiffOptModel`
(MadNLP-backed) as DiffOpt's inner sensitivity optimizer.
"""
MadDiff.diffopt_model_constructor(; config::MadDiff.MadDiffConfig = MadDiff.MadDiffConfig()) =
    () -> DiffOptModel(; sensitivity_config = config)

"""
    MadDiff.diff_model(optimizer_constructor; config = MadDiffConfig(), kwargs...)

Build a JuMP model whose optimizer chain ends in MadNLP and whose
differentiation backend is MadDiff.
"""
function MadDiff.diff_model(
    optimizer_constructor;
    config::MadDiff.MadDiffConfig = MadDiff.MadDiffConfig(),
    kwargs...,
)
    model = DiffOpt.diff_model(optimizer_constructor; kwargs...)
    MOI.set(model, DiffOpt.AllowObjectiveAndSolutionInput(), true)
    MOI.set(model, DiffOpt.ModelConstructor(),
            MadDiff.diffopt_model_constructor(config = config))
    return model
end

end # module
