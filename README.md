# MadDiff

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://madnlp.github.io/MadDiff.jl/dev/)
[![Build Status](https://github.com/MadNLP/MadDiff.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MadNLP/MadDiff.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MadNLP/MadDiff.jl/branch/main/graph/badge.svg?token=ERB8DC2NZE)](https://codecov.io/gh/MadNLP/MadDiff.jl)

MadDiff implements forward and reverse mode implicit differentiation for MadSuite solvers. MadDiff leverages MadNLP's modular KKT and linear solver infrastructure, supporting LP, QP, and NLP using KKT systems from [MadNLP](https://github.com/MadNLP/MadNLP.jl), [MadIPM](https://github.com/MadNLP/MadIPM.jl), and [HybridKKT](https://github.com/MadNLP/HybridKKT.jl).

> [!WARNING]
> MadDiff is a work-in-progress and requires installing [forks of several dependencies](https://github.com/MadNLP/MadDiff.jl/blob/271ceec7f623789229e8c04a8bc6efa657867c38/test/Project.toml#L22-L28). Proceed with caution and verify correctness before use.

## NLPModels interface

> [!NOTE]
> The NLPModels interface requires that your `AbstractNLPModel` implementation includes the [parametric AD API](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/pull/557). Currently, this is automated only for the case when using MadNLP through JuMP ([fork](https://github.com/klamike/MadNLP.jl/tree/mk/moi_param)) or when using ExaModels ([fork](https://github.com/klamike/ExaModels.jl/tree/mk/param_ad)); support for other solvers and modelers is planned.


```julia
nlp = ...  # must implement parametric AD API
solver = MadNLP.MadNLPSolver(nlp)
solution = MadNLP.solve!(solver)

diff = MadDiff.MadDiffSolver(solver)

dL_dx, dL_dy, dL_dzl, dL_dzu = ...  # loss sensitivity vectors
rev = MadDiff.vector_jacobian_product!(diff; dL_dx, dL_dy, dL_dzl, dL_dzu)
rev.grad_p  # gradient of the loss with respect to the parameters
```

## JuMP interface

MadDiff aims to be a drop-in replacement for [DiffOpt](https://github.com/jump-dev/DiffOpt.jl) with MadNLP. Simply switch `DiffOpt.diff_model(MadNLP.Optimizer)` for `MadDiff.diff_model(MadNLP.Optimizer)` and enjoy the speedup!

```julia
using JuMP, DiffOpt
using MadDiff, MadNLP

model = MadDiff.diff_model(MadNLP.Optimizer)
@variable(model, x)
@variable(model, p in MOI.Parameter(1.0))
@constraint(model, x >= 2p)
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
