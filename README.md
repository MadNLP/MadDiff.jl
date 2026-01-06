# MadDiff

MadDiff is a Julia package for differentiating MadSuite solvers. MadDiff relies on MadNLP's KKT system and linear solver infrastructure, allowing both the solve and the differentiation to run on the GPU, and to re-use the existing KKT system. It also integrates with the DiffOpt API for CPU.

## NLPModels API

```julia
using JuMP
using NLPModelsJuMP
using MadNLP
using MadDiff

model = Model()
@variable(model, x)
@constraint(model, x >= 1.0)  # the RHS is the parameter (implicit)
@objective(model, Min, x^2)

nlp = MathOptNLPModel(model)
solver = MadNLP.MadNLPSolver(nlp)
result = MadNLP.solve!(solver)

sens = MadDiff.MadDiffSolver(solver)
# NOTE: when using NLPModelsJuMP with MadDiff's NLPModels API,
#       be careful about variable/constraint ordering wrt JuMP.

∇xpL = [0.0]   # (∂²L/∂x∂p) * Δp
∇pg = [-1.0]   # (∂g/∂p) * Δp for x >= 1 + p
fwd = MadDiff.forward_differentiate!(sens, ∇xpL, ∇pg)

rev = MadDiff.reverse_differentiate!(sens, [1.0])   # ∂f/∂x * Δx for f = x
```

## DiffOpt API

```julia
using JuMP
using DiffOpt
using MadDiff
using MadNLP
const MOI = MathOptInterface

# just switch DiffOpt.diff_optimizer for MadDiff.diff_optimizer!
model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
@variable(model, x)
@variable(model, p in MOI.Parameter(1.0))
@constraint(model, x >= p)  # can use explicit parameters
@objective(model, Min, x^2)
optimize!(model)

DiffOpt.empty_input_sensitivities!(model)
MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), MOI.Parameter(1.0))
DiffOpt.forward_differentiate!(model)
dx = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x)

DiffOpt.empty_input_sensitivities!(model)
MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)
DiffOpt.reverse_differentiate!(model)
dp = MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)).value
```
