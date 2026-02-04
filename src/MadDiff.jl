module MadDiff

import MadNLP
import MadNLP: AbstractMadNLPSolver, MadNLPSolver, _madnlp_unsafe_wrap,
    set_aug_diagonal!, set_aug_rhs!, get_slack_regularization, dual_inf_perturbation!,
    inertia_correction!, solve!, multi_solve!, solve_refine!, improve!, RichardsonIterator,
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

const NLPModels = MadNLP.NLPModels
import MadNLP.NLPModels: @lencheck
import LinearAlgebra: dot, mul!, norm, axpy!, Symmetric
import ParametricNLPModels

include("packing.jl")
include("adjoint.jl")
include("api.jl")
include("cache.jl")
include("utils.jl")
include("kkt.jl")
include("forward.jl")
include("reverse.jl")

export MadDiffSolver, MadDiffConfig
export forward_differentiate!, reverse_differentiate!
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
struct DifferentiateTimeSec end
const MADDIFF_KKTSYSTEM = "MadDiffKKTSystem"
const MADDIFF_KKTSYSTEM_OPTIONS = "MadDiffKKTSystemOptions"
const MADDIFF_LINEARSOLVER = "MadDiffLinearSolver"
const MADDIFF_LINEARSOLVER_OPTIONS = "MadDiffLinearSolverOptions"
const MADDIFF_SKIP_KKT_REFACTORIZATION = "MadDiffSkipKKTRefactorization"

end # module MadDiff
