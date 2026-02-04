# MadDiff

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://klamike.github.io/MadDiff.jl/dev/)
[![Build Status](https://github.com/klamike/MadDiff.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/klamike/MadDiff.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/klamike/MadDiff.jl/branch/main/graph/badge.svg?token=ERB8DC2NZE)](https://codecov.io/gh/klamike/MadDiff.jl)

MadDiff is a Julia package for differentiating MadSuite solvers. MadDiff relies on MadNLP's KKT system and linear solver infrastructure, allowing both the solve and the differentiation to run on the GPU, and to re-use the existing KKT system. It supports LP, QP, and NLP using KKT systems from MadNLP, MadIPM, MadNCL, and HybridKKT. For reverse mode with custom KKT systems, define `adjoint_mul!` and `adjoint_solve!`; forward mode requires no additional modifications besides setting the `kkt_system` option. MadDiff also has an experimental integration with the DiffOpt API.

## NLPModels API

The NLPModels API requires that your `AbstractNLPModel` implementation includes the `ParametricNLPModels` API. Currently, only the MadNLP MOIModel is supported. Support for ExaModels, ADNLPModels, and NLPModelsJuMP is planned.

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
