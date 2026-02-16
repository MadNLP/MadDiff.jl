module MadDiff

import MadNLP
import MadNLP: AbstractMadNLPSolver, MadNLPSolver, _madnlp_unsafe_wrap,
    set_aug_diagonal!, set_aug_rhs!, get_slack_regularization, dual_inf_perturbation!,
    inertia_correction!, solve_kkt_system!, solve_linear_system!, multi_solve!, solve_refine!, improve!, RichardsonIterator,
    full, primal, variable, slack, dual, dual_lb, dual_ub, primal_dual, num_variables,
    SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL,
    create_kkt_system, initialize!,
    AbstractKKTSystem, AbstractCondensedKKTSystem, AbstractDenseKKTSystem, AbstractUnreducedKKTSystem,
    SparseUnreducedKKTSystem, CompactLBFGS,
    SparseCondensedKKTSystem, DenseCondensedKKTSystem,
    ScaledSparseKKTSystem, SparseKKTSystem, DenseKKTSystem, 
    AbstractKKTVector, UnreducedKKTVector, PrimalVector,
    unpack_x!, unpack_y!, unpack_z!,
    eval_jac_wrapper!, eval_lag_hess_wrapper!,
    AbstractCallback, SparseCallback, MakeParameter, create_array,
    @debug, @sprintf, _symv!, _eval_grad_f_wrapper!, _get_sparse_csc

import NLPModels: @lencheck, get_nvar, get_ncon, get_x0, get_y0, grad!
import ParametricNLPModels: hpprod!, jpprod!,
                            lvar_jpprod!, uvar_jpprod!, lcon_jpprod!, ucon_jpprod!,
                            grad_param!, hess_param!, jac_param!, lvar_jac_param!, uvar_jac_param!, lcon_jac_param!, ucon_jac_param!,
                            hptprod!, jptprod!,
                            lvar_jptprod!, uvar_jptprod!, lcon_jptprod!, ucon_jptprod!
import LinearAlgebra: dot, mul!, norm, axpy!, Symmetric

include("utils/packing.jl")
include("utils/batch_packing.jl")
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
include("jacobian.jl")
include("vjp.jl")
include("jacobian_transpose.jl")

export MadDiffSolver, MadDiffConfig
export jacobian_vector_product!, vector_jacobian_product!
export jacobian!, jacobian_transpose!
export reset_sensitivity_cache!

# implemented in MathOptInterfaceExt
function diff_optimizer end
function forward_differentiate! end
function reverse_differentiate! end
function empty_input_sensitivities! end
function nonlinear_diff_model end
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
