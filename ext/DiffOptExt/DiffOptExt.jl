module DiffOptExt

import MadDiff
import DiffOpt
const MOI = DiffOpt.MOI

const MathOptInterfaceExt = Base.get_extension(MadDiff, :MathOptInterfaceExt)
const Optimizer = MathOptInterfaceExt.Optimizer

DiffOpt.forward_differentiate!(model::Optimizer) = MadDiff.forward_differentiate!(model)
DiffOpt.reverse_differentiate!(model::Optimizer) = MadDiff.reverse_differentiate!(model)
DiffOpt.empty_input_sensitivities!(model::Optimizer) = MadDiff.empty_input_sensitivities!(model)

MOI.set(m::Optimizer, ::DiffOpt.ForwardConstraintSet, ci::MOI.ConstraintIndex, set) =
    MOI.set(m, MadDiff.ForwardConstraintSet(), ci, set)
MOI.get(m::Optimizer, ::DiffOpt.ForwardVariablePrimal, vi::MOI.VariableIndex) =
    MOI.get(m, MadDiff.ForwardVariablePrimal(), vi)
MOI.get(m::Optimizer, ::DiffOpt.ForwardConstraintDual, ci::MOI.ConstraintIndex) =
    MOI.get(m, MadDiff.ForwardConstraintDual(), ci)

MOI.set(m::Optimizer, ::DiffOpt.ReverseVariablePrimal, vi::MOI.VariableIndex, value) =
    MOI.set(m, MadDiff.ReverseVariablePrimal(), vi, value)
MOI.set(m::Optimizer, ::DiffOpt.ReverseConstraintDual, ci::MOI.ConstraintIndex, value) =
    MOI.set(m, MadDiff.ReverseConstraintDual(), ci, value)
MOI.get(m::Optimizer, ::DiffOpt.ReverseConstraintSet, ci::MOI.ConstraintIndex) =
    MOI.get(m, MadDiff.ReverseConstraintSet(), ci)

MOI.get(m::Optimizer, ::DiffOpt.DifferentiateTimeSec) =
    MOI.get(m, MadDiff.DifferentiateTimeSec())

end # module
