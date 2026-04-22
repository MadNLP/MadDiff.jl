module MadDiff

# ============================================================================
# MadDiff — implicit differentiation through MadNLP.
#
# Given a solved `MadNLP.AbstractMadNLPSolver` whose NLP implements the
# `ParametricNLPModels` parametric-sensitivity API, MadDiff computes
#
#   dx/dp, dy/dp, dzl/dp, dzu/dp
#
# by linearising the KKT system at the optimum and reusing MadNLP's factored
# KKT matrix (or a fresh one, per `MadDiffConfig`). The public surface is a
# pair of kernels — `jacobian_vector_product!` (forward) and
# `vector_jacobian_product!` (reverse) — plus a `ChainRulesCore` wrapper
# (`differentiable_solve`) and DiffOpt / MOI adapters that live in extensions.
# ============================================================================

import LinearAlgebra
import LinearAlgebra: axpy!, dot, mul!, norm, Symmetric
import SparseArrays

import MadNLP
import MadNLP:
    AbstractCallback, AbstractDenseKKTSystem, AbstractKKTSystem,
    AbstractKKTVector, AbstractMadNLPSolver, AbstractUnreducedKKTSystem,
    CompactLBFGS, DenseCondensedKKTSystem, DenseKKTSystem, MadNLPSolver,
    MadNLPCounters, MadNLPLogger, MakeParameter, PrimalVector,
    RichardsonIterator, RichardsonOptions, ScaledSparseKKTSystem,
    SparseCallback, SparseCondensedKKTSystem, SparseKKTSystem,
    SparseUnreducedKKTSystem, SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL,
    UnreducedKKTVector,
    _eval_grad_f_wrapper!, _madnlp_unsafe_wrap, _symv!, @debug, @sprintf,
    create_array, create_kkt_system, dual, dual_inf_perturbation!, dual_lb,
    dual_ub, eval_jac_wrapper!, eval_lag_hess_wrapper!, full,
    get_slack_regularization, improve!, inertia_correction!, initialize!,
    num_variables, primal, primal_dual, set_aug_diagonal!, set_aug_rhs!,
    slack, solve_kkt!, solve_linear_system!, solve_refine!, unpack_x!,
    unpack_y!, unpack_z!, variable

import NLPModels: @lencheck, get_ncon, get_nvar, grad!
import ParametricNLPModels:
    get_nnzgp, get_nnzhp, get_nnzjp, get_nnzjplcon, get_nnzjplvar,
    get_nnzjpucon, get_nnzjpuvar, get_nparam,
    grad_param!, hpprod!, hptprod!, jpprod!, jptprod!,
    lcon_jpprod!, lcon_jptprod!, lvar_jpprod!, lvar_jptprod!,
    ucon_jpprod!, ucon_jptprod!, uvar_jpprod!, uvar_jptprod!

# ---------- core implementation ----------

include("utils/packing.jl")
include("KKT/adjoint.jl")
include("KKT/Sparse/augmented.jl")
include("KKT/Sparse/scaled_augmented.jl")
include("KKT/Sparse/unreduced.jl")
include("KKT/Sparse/condensed.jl")
include("KKT/Dense/augmented.jl")
include("KKT/Dense/condensed.jl")
include("api.jl")
include("utils/cache.jl")
include("utils/utils.jl")
include("KKT/kkt.jl")
include("jvp.jl")
include("vjp.jl")
include("chainrules_api.jl")
include("extension_stubs.jl")

export MadDiffSolver, MadDiffConfig,
    JVPResult, VJPResult,
    jacobian_vector_product!, vector_jacobian_product!,
    compute_objective_sensitivity!, reset_sensitivity_cache!,
    differentiable_solve, SolverSpec, Components,
    batch_differentiable_solve, BatchSolverSpec, BatchMadDiffSolver

end # module MadDiff
