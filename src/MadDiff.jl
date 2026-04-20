module MadDiff

import MadNLP
import MadNLP: AbstractMadNLPSolver, MadNLPSolver, _madnlp_unsafe_wrap,
    set_aug_diagonal!, set_aug_rhs!, get_slack_regularization, dual_inf_perturbation!,
    inertia_correction!, solve_kkt!, solve_linear_system!, solve_refine!, improve!, RichardsonIterator,
    full, primal, variable, slack, dual, dual_lb, dual_ub, primal_dual, num_variables,
    SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL,
    create_kkt_system, initialize!,
    AbstractKKTSystem, AbstractDenseKKTSystem, AbstractUnreducedKKTSystem,
    SparseUnreducedKKTSystem, CompactLBFGS,
    SparseCondensedKKTSystem, DenseCondensedKKTSystem,
    ScaledSparseKKTSystem, SparseKKTSystem, DenseKKTSystem, 
    AbstractKKTVector, UnreducedKKTVector, PrimalVector,
    unpack_x!, unpack_y!, unpack_z!,
    eval_jac_wrapper!, eval_lag_hess_wrapper!,
    AbstractCallback, SparseCallback, MakeParameter, create_array,
    @debug, @sprintf, _symv!, _eval_grad_f_wrapper!

import NLPModels: @lencheck, get_nvar, get_ncon, get_x0, get_y0, grad!
import ParametricNLPModels:
    get_nparam, get_nnzgp, get_nnzjp, get_nnzhp, get_nnzjplcon, get_nnzjpucon, get_nnzjplvar, get_nnzjpuvar,
    grad_param!, hpprod!, hptprod!, jpprod!, jptprod!,
    lvar_jpprod!, lvar_jptprod!, uvar_jpprod!, uvar_jptprod!,
    lcon_jpprod!, lcon_jptprod!, ucon_jpprod!, ucon_jptprod!
import LinearAlgebra: dot, mul!, norm, axpy!, Symmetric

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

export MadDiffSolver, MadDiffConfig
export jacobian_vector_product!, vector_jacobian_product!
export compute_objective_sensitivity!
export reset_sensitivity_cache!
export differentiable_solve, SolverSpec, Components

"""
    BatchMadDiffSolver(batch_solver::MadIPM.UniformBatchMPCSolver)

Construct a batched sensitivity solver wrapping a solved `UniformBatchMPCSolver`.
Provides `jacobian_vector_product!` / `vector_jacobian_product!` methods that
operate on `(nparam, batch_size)` matrices and return per-instance results.

!!! note
    This constructor lives in the `MadIPMExt` extension. Load `MadIPM` (and
    ensure `BatchQuadraticModels` is available) for the batch path. The
    current MadIPM release refactored the batch solver's internal field
    layout; the batch JVP/VJP adapters are on the `mk/batch` branch of this
    repository and are in the process of being forward-ported — see the
    project memory note `project_maddiff_batch.md` for status.
"""
function BatchMadDiffSolver end

"""
    diff_model(optimizer_constructor; kwargs...)

Create a JuMP Model with MadDiff wrapping `optimizer_constructor`.
"""
function diff_model(args...; kwargs...)
    error(
        "`MadDiff.diff_model` requires the `DiffOpt` extension. " *
        "Make sure both `DiffOpt` and `MadDiff` are loaded when using the JuMP API.",
    )
end

function forward_differentiate! end
function reverse_differentiate! end
function empty_input_sensitivities! end
function diffopt_model_constructor end
function get_reverse_parameter end
struct ForwardConstraintSet end
struct ForwardVariablePrimal end
struct ForwardConstraintDual end
struct ReverseVariablePrimal end
struct ReverseConstraintDual end
struct ReverseConstraintSet end
struct ForwardObjectiveSensitivity end
struct ReverseObjectiveSensitivity end
struct DifferentiateTimeSec end
const MADDIFF_KKTSYSTEM = "MadDiffKKTSystem"
const MADDIFF_KKTSYSTEM_OPTIONS = "MadDiffKKTSystemOptions"
const MADDIFF_LINEARSOLVER = "MadDiffLinearSolver"
const MADDIFF_LINEARSOLVER_OPTIONS = "MadDiffLinearSolverOptions"
const MADDIFF_SKIP_KKT_REFACTORIZATION = "MadDiffSkipKKTRefactorization"

end # module MadDiff
