const SUCCESSFUL_STATUSES = (SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL)
function assert_solved_and_feasible(solver::AbstractMadNLPSolver)
    solver.status ∉ SUCCESSFUL_STATUSES &&
        error("Solver did not converge successfully: $(solver.status)")
    return nothing
end

_get_fixed_idx(cb::AbstractCallback, ::Any) = nothing
function _get_fixed_idx(cb::SparseCallback{T,VT,VI,NLP,FH}, ref_array) where {T,VT,VI,NLP,FH<:MakeParameter}
    return cb.fixed_handler.fixed
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
