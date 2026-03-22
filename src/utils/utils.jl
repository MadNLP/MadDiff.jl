const SUCCESSFUL_STATUSES = (SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL)
function assert_solved_and_feasible(solver::AbstractMadNLPSolver)
    solver.status ∉ SUCCESSFUL_STATUSES &&
        error("Solver did not converge successfully: $(solver.status)")
    return nothing
end

_get_wrapper_type(x) = Base.typename(typeof(x)).wrapper

struct _SensitivitySolverShim{T, S<:AbstractMadNLPSolver{T}, K<:AbstractKKTSystem{T}} <: AbstractMadNLPSolver{T}
    inner::S
    kkt::K
end

function Base.getproperty(s:: _SensitivitySolverShim, name::Symbol)
    name === :inner && return getfield(s, :inner)
    name === :kkt && return getfield(s, :kkt)
    return getproperty(getfield(s, :inner), name)
end

function Base.setproperty!(s:: _SensitivitySolverShim, name::Symbol, value)
    name === :inner && return setfield!(s, :inner, value)
    name === :kkt && return setfield!(s, :kkt, value)
    return setproperty!(getfield(s, :inner), name, value)
end


function _needs_new_kkt(config)
    return !isnothing(config.kkt_system) ||
        !isnothing(config.kkt_options) ||
        !isnothing(config.linear_solver) ||
        !isnothing(config.linear_solver_options)
end

has_hess_param(nlp) = get_nnzhp(nlp) != 0
has_jac_param(nlp) = get_nnzjp(nlp) != 0
has_lvar_param(nlp) = get_nnzjplvar(nlp) != 0
has_uvar_param(nlp) = get_nnzjpuvar(nlp) != 0
has_lcon_param(nlp) = get_nnzjplcon(nlp) != 0
has_ucon_param(nlp) = get_nnzjpucon(nlp) != 0
has_grad_param(nlp) = get_nnzgp(nlp) != 0
