# ============================================================================
# ChainRulesCore surface
#
# Public types (`SolverSpec`, `Components`) and entry points
# (`differentiable_solve`, `_solve_forward`) live here; the actual rrules are
# in `MadDiffChainRulesCoreExt`. Design mirrors `Moreau.jl`: a `SolverSpec`
# wraps the user's `p -> (solver, obj)` closure; `differentiable_solve`
# returns a `p -> NamedTuple` with a `ChainRulesCore.rrule` registered that
# back-solves the KKT system at the optimum.
# ============================================================================

"""
    Components{C}

Type-level tag carrying the solution components to return from
[`differentiable_solve`](@ref). `C` is a tuple drawn from
`(:x, :y, :zl, :zu, :obj)`. The adjoint elides back-solves for any component
not listed.

User-facing tuples are canonicalised at construction — `Components((:x, :y))`
and `Components((:y, :x))` resolve to the same type, so distinct permutations
of the same set do not force recompilation of the rrule closure.
"""
struct Components{C} end
function Components(c::Tuple{Vararg{Symbol}})
    # Canonical order (`_COMPONENT_ORDER`) so permutations unify into one type.
    return Components{_canonical_components(c)}()
end

const _COMPONENT_ORDER = (:x, :y, :zl, :zu, :obj)

function _canonical_components(c::Tuple{Vararg{Symbol}})
    # Emits each valid component at most once, in `_COMPONENT_ORDER`.
    return Tuple(s for s in _COMPONENT_ORDER if s in c)
end

"""
    SolverSpec(forward; kind = :build)

Wrap a parameter-to-solution pipeline:

- `forward::Function` — unary `p -> (solver, obj_val)` where `solver` is an
  already-solved `MadNLP.AbstractMadNLPSolver` whose underlying NLP implements
  `ParametricNLPModels`.
- `kind::Symbol` — informational tag: `:build` (rebuild the NLP each call),
  `:update` (reuse a persistent workspace).
"""
struct SolverSpec{F<:Function}
    forward::F
    kind::Symbol
end
SolverSpec(forward::Function; kind::Symbol = :build) = SolverSpec(forward, kind)

function _run_forward(spec::SolverSpec, p)
    solver, obj_val = _unpack_forward(spec.forward(p))
    return MadDiffSolver(solver), obj_val
end

_unpack_forward(t::Tuple{Any, Any}) = t
_unpack_forward(solver)             = (solver, zero(eltype(full(solver.x))))

"""
    differentiable_solve(spec::SolverSpec; components = (:x,))

Return a closure `p -> solution::NamedTuple` whose fields are `components`.
The closure carries a `ChainRulesCore.rrule` registered by
`MadDiffChainRulesCoreExt`; the adjoint uses the implicit-function theorem at
the optimum and only projects cotangents for the requested components.

`components` must be a subset of `(:x, :y, :zl, :zu, :obj)` without duplicates.
Requires `using ChainRulesCore`.
"""
function differentiable_solve end
function _solve_forward end

# ---------- batch variant ----------

"""
    BatchSolverSpec(forward; kind = :build)

Batched analogue of [`SolverSpec`](@ref). `forward` is a unary callable
`p_matrix -> (batch_solver, obj_vec)` where `batch_solver` is an already-solved
`UniformBatchMPCSolver` and `obj_vec::AbstractVector` carries one objective
value per batch instance. `p_matrix` is `(nparam, batch_size)`.
"""
struct BatchSolverSpec{F<:Function}
    forward::F
    kind::Symbol
end
BatchSolverSpec(forward::Function; kind::Symbol = :build) = BatchSolverSpec(forward, kind)

function _run_batch_forward(spec::BatchSolverSpec, p)
    out = spec.forward(p)
    batch_solver, obj_vec = _unpack_batch_forward(out)
    return BatchMadDiffSolver(batch_solver), obj_vec
end

_unpack_batch_forward(t::Tuple{Any, Any}) = t
function _unpack_batch_forward(batch_solver)
    bs = batch_solver.problem.batch_size
    T  = eltype(batch_solver.state.workspace.bx)
    return batch_solver, zeros(T, bs)
end

"""
    batch_differentiable_solve(spec::BatchSolverSpec; components = (:x,))

Batched analogue of [`differentiable_solve`](@ref). Returns a closure
`p_matrix -> solution::NamedTuple` whose fields are per-instance matrices
(or a length-`bs` vector for `:obj`). The closure has a `ChainRulesCore.rrule`
that runs the batch VJP with only the requested components seeded.

Requires `using ChainRulesCore`.
"""
function batch_differentiable_solve end

function _batch_solve_forward end
