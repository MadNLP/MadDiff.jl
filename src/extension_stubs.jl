# ============================================================================
# Stubs filled in by package extensions.
#
# DiffOpt + MOI adapters: `DiffOptExt` / `MathOptInterfaceExt`.
# MadIPM batch adapters:  `MadIPMExt`.
# ChainRulesCore rrules:  `MadDiffChainRulesCoreExt` (see `chainrules_api.jl`).
# ============================================================================

# ---------- DiffOpt / MOI ----------

"""
    diff_model(optimizer_constructor; config = MadDiffConfig(), kwargs...)

Build a JuMP model whose optimizer chain ends in MadNLP and whose
differentiation backend is MadDiff. Supplied by the `DiffOptExt` extension
(`using DiffOpt` to activate).
"""
function diff_model end

function diffopt_model_constructor end
function forward_differentiate! end
function reverse_differentiate! end
function empty_input_sensitivities! end
function get_reverse_parameter end

# MadDiff-local attribute tags mirroring DiffOpt's; the MOI extension dispatches
# on these and the DiffOpt extension forwards from DiffOpt's attribute types.
struct ForwardConstraintSet end
struct ForwardVariablePrimal end
struct ForwardConstraintDual end
struct ForwardObjectiveSensitivity end
struct ReverseVariablePrimal end
struct ReverseConstraintDual end
struct ReverseConstraintSet end
struct ReverseObjectiveSensitivity end
struct DifferentiateTimeSec end

# Option-name constants consumed by the MOI extension.
const MADDIFF_KKTSYSTEM                = "MadDiffKKTSystem"
const MADDIFF_KKTSYSTEM_OPTIONS        = "MadDiffKKTSystemOptions"
const MADDIFF_LINEARSOLVER             = "MadDiffLinearSolver"
const MADDIFF_LINEARSOLVER_OPTIONS     = "MadDiffLinearSolverOptions"
const MADDIFF_SKIP_KKT_REFACTORIZATION = "MadDiffSkipKKTRefactorization"

# ---------- MadIPM batch ----------

"""
    BatchMadDiffSolver(batch_solver)

Batched sensitivity wrapper around a solved `MadIPM.UniformBatchMPCSolver`.
Provides `jacobian_vector_product!` / `vector_jacobian_product!` methods that
operate on `(nparam, batch_size)` matrices.

Activated by the `MadIPMExt` extension.
"""
function BatchMadDiffSolver end
