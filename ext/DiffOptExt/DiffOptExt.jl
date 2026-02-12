module DiffOptExt

import MadDiff
import DiffOpt
const MOI = DiffOpt.MOI

MOIExt = Base.get_extension(MadDiff, :MathOptInterfaceExt)
Optimizer = MOIExt.Optimizer

DiffOpt.forward_differentiate!(model::Optimizer) = MadDiff.forward_differentiate!(model)
DiffOpt.reverse_differentiate!(model::Optimizer) = MadDiff.reverse_differentiate!(model)
DiffOpt.empty_input_sensitivities!(model::Optimizer) = MadDiff.empty_input_sensitivities!(model)
MadDiff.nonlinear_diff_model(optimizer_constructor; kwargs...) = DiffOpt.JuMP.Model(MadDiff.diff_optimizer(optimizer_constructor; kwargs...))

MOI.set(m::Optimizer, ::DiffOpt.ForwardConstraintSet, ci::MOI.ConstraintIndex, set) =
    MOI.set(m, MadDiff.ForwardConstraintSet(), ci, set)
MOI.get(m::Optimizer, ::DiffOpt.ForwardVariablePrimal, vi::MOI.VariableIndex) =
    MOI.get(m, MadDiff.ForwardVariablePrimal(), vi)
MOI.get(m::Optimizer, ::DiffOpt.ForwardConstraintDual, ci::MOI.ConstraintIndex) =
    MOI.get(m, MadDiff.ForwardConstraintDual(), ci)
MOI.get(m::Optimizer, ::DiffOpt.ForwardObjectiveSensitivity) =
    MOI.get(m, MadDiff.ForwardObjectiveSensitivity())

MOI.set(m::Optimizer, ::DiffOpt.ReverseVariablePrimal, vi::MOI.VariableIndex, value) =
    MOI.set(m, MadDiff.ReverseVariablePrimal(), vi, value)
MOI.set(m::Optimizer, ::DiffOpt.ReverseConstraintDual, ci::MOI.ConstraintIndex, value) =
    MOI.set(m, MadDiff.ReverseConstraintDual(), ci, value)
MOI.set(m::Optimizer, ::DiffOpt.ReverseObjectiveSensitivity, value) =
    MOI.set(m, MadDiff.ReverseObjectiveSensitivity(), value)
MOI.get(m::Optimizer, ::DiffOpt.ReverseConstraintSet, ci::MOI.ConstraintIndex) =
    MOI.get(m, MadDiff.ReverseConstraintSet(), ci)

MOI.get(m::Optimizer, ::DiffOpt.DifferentiateTimeSec) =
    MOI.get(m, MadDiff.DifferentiateTimeSec())

DiffOpt.get_reverse_parameter(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}},
) where {T} = MadDiff.get_reverse_parameter(model, ci)

end # module
