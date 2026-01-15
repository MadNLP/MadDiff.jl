module MadDiff

import MadNLP
const NLPModels = MadNLP.NLPModels
import MadNLP.NLPModels: @lencheck
import LinearAlgebra: dot, mul!

include("api.jl")
include("kkt.jl")
include("forward.jl")
include("reverse.jl")

export MadDiffSolver, MadDiffConfig
export forward_differentiate!, reverse_differentiate!
export reset_sensitivity_cache!, make_param_pullback

# implemented in MathOptInterfaceExt
function diff_optimizer end
function empty_input_sensitivities! end
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
const MADDIFF_REGULARIZATION = "MadDiffRegularization"

end # module MadDiff
