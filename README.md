# MadDiff

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://klamike.github.io/MadDiff.jl/dev/)
[![Build Status](https://github.com/klamike/MadDiff.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/klamike/MadDiff.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/klamike/MadDiff.jl/branch/main/graph/badge.svg?token=ERB8DC2NZE)](https://codecov.io/gh/klamike/MadDiff.jl)

MadDiff implements forward and reverse mode implicit differentiation for MadSuite solvers. MadDiff leverages MadNLP's modular KKT and linear solver infrastructure, supporting LP, QP, and NLP using KKT systems from [MadNLP](https://github.com/MadNLP/MadNLP.jl), [MadIPM](https://github.com/MadNLP/MadIPM.jl), [MadNCL](https://github.com/MadNLP/MadNCL.jl), and [HybridKKT](https://github.com/MadNLP/HybridKKT.jl).

> [!WARNING]
> MadDiff is a work-in-progress. Proceed with caution and verify correctness before use.

## NLPModels API

> [!NOTE]
> The NLPModels API requires that your `AbstractNLPModel` implementation includes the `ParametricNLPModels` API. Currently, only the MadNLP MOIModel is supported. Support for ExaModels, ADNLPModels, and NLPModelsJuMP is planned.


```julia
nlp = ...  # must implement ParametricNLPModels API
solver = MadNLP.MadNLPSolver(nlp)
solution = MadNLP.solve!(solver)

diff = MadDiff.MadDiffSolver(solver)

dL_dx, dL_dy, dL_dzl, dL_dzu = ...  # loss sensitivity vectors
rev = MadDiff.reverse_differentiate!(diff; dL_dx, dL_dy, dL_dzl, dL_dzu)
rev.grad_p  # gradient of the loss with respect to the parameters
```

## DiffOpt API

MadDiff aims to be a drop-in replacement for DiffOpt with MadNLP. Simply switch `DiffOpt.diff_optimizer(MadNLP.Optimizer)` for `MadDiff.diff_optimizer(MadNLP.Optimizer)` and enjoy the speedup!

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
