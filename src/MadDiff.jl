module MadDiff

import MadNLP
import MadNLP: AbstractMadNLPSolver, MadNLPSolver, _madnlp_unsafe_wrap,
    set_aug_diagonal!, set_aug_rhs!, get_slack_regularization, dual_inf_perturbation!,
    inertia_correction!, solve_linear_system!, multi_solve!, solve_refine!, improve!, RichardsonIterator,
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
    @debug, @sprintf, _symv!

import NLPModels: @lencheck, get_nvar, get_ncon, get_x0, get_y0, grad!
import LinearAlgebra: dot, mul!, norm, axpy!, Symmetric
import ParametricNLPModels
import SparseArrays: spzeros

include("packing.jl")
include("adjoint.jl")
include("api.jl")
include("cache.jl")
include("utils.jl")
include("kkt.jl")
include("forward.jl")
include("jacobian_forward.jl")
include("reverse.jl")
include("jacobian_reverse.jl")

export MadDiffSolver, MadDiffConfig
export forward_differentiate!, reverse_differentiate!
export forward_jacobian!, reverse_jacobian_transpose!
export reset_sensitivity_cache!

# implemented in MathOptInterfaceExt
function diff_optimizer end
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
