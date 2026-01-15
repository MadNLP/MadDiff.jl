# MadDiff

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://klamike.github.io/MadDiff.jl/dev/)
[![Build Status](https://github.com/klamike/MadDiff.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/klamike/MadDiff.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/klamike/MadDiff.jl/branch/main/graph/badge.svg?token=ERB8DC2NZE)](https://codecov.io/gh/klamike/MadDiff.jl)

MadDiff is a Julia package for differentiating MadSuite solvers. MadDiff relies on MadNLP's KKT system and linear solver infrastructure, allowing both the solve and the differentiation to run on the GPU, and to re-use the existing KKT system. It also integrates with the DiffOpt API for CPU.

## NLPModels API

```julia
using JuMP
using NLPModelsJuMP
using MadNLP
using MadDiff
using LinearAlgebra

model = Model()
@variable(model, x)
@constraint(model, x >= 2.0)  # the RHS is 2*p
@objective(model, Min, x^2)

nlp = MathOptNLPModel(model)
solver = MadNLP.MadNLPSolver(nlp)
result = MadNLP.solve!(solver)

sens = MadDiff.MadDiffSolver(solver)
# NOTE: when using NLPModelsJuMP with MadDiff's NLPModels API,
#       be careful about variable/constraint ordering wrt JuMP.

Dp_lcon = [2.0;;]  # ∂lcon/∂p (n_con × n_params matrix)
Δp = [1.0]  # parameter perturbation direction
fwd = MadDiff.forward_differentiate!(sens; Dp_lcon=Dp_lcon * Δp)

dL_dx = [1.0]  # gradient of loss w.r.t. x*
pullback = MadDiff.make_param_pullback(Dp_lcon=Dp_lcon)
sens_rev = MadDiff.MadDiffSolver(solver; param_pullback, n_params=1)
rev = MadDiff.reverse_differentiate!(sens_rev; dL_dx)

@assert dot(dL_dx, fwd.dx) ≈ dot(rev.grad_p, Δp)
```

## DiffOpt API

```julia
using JuMP
using DiffOpt
using MadDiff
using MadNLP
const MOI = MathOptInterface

# just switch DiffOpt.diff_optimizer for MadDiff.diff_optimizer, everything else stays the same!
model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
@variable(model, x)
@variable(model, p in MOI.Parameter(1.0))
@constraint(model, x >= 2p)  # can use explicit parameters
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
