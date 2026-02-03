const SUCCESSFUL_STATUSES = (SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL)
function assert_solved_and_feasible(solver::AbstractMadNLPSolver)
    solver.status ∉ SUCCESSFUL_STATUSES &&
        error("Solver did not converge successfully: $(solver.status)")
    return nothing
end

_pullback_add!(out, ::Nothing, v) = nothing
function _pullback_add!(out, M, v)
    @lencheck size(M, 1) v
    out .+= M' * v
    return nothing
end

_pullback_sub!(out, ::Nothing, v) = nothing
function _pullback_sub!(out, M, v)
    @lencheck size(M, 1) v
    out .-= M' * v
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
