# =============================================================================
# Public API surface for the ChainRulesCore extension
# -----------------------------------------------------------------------------
# These stubs define the symbols that `MadDiffChainRulesCoreExt` fills in when
# `ChainRulesCore` is loaded. If the user calls `differentiable_solve` without
# having loaded ChainRulesCore, we raise an instructive error instead of a
# MethodError.
#
# The API is deliberately close to `Moreau.jl`'s pattern: a `SolverSpec` packs
# a forward-pass closure ("given `p`, solve the problem and return the
# `MadDiffSolver`") plus metadata; `differentiable_solve(spec; components)`
# returns a ChainRulesCore-friendly function `p -> NamedTuple`. `components`
# is a compile-time tag so the adjoint only computes the KKT back-solves that
# are actually needed.
# =============================================================================

"""
    Components{tuple_of_symbols}

Type-level tag carrying the set of solution components that the caller cares
about. Valid symbols are `:x`, `:y`, `:zl`, `:zu`, `:obj`. Used internally by
[`differentiable_solve`](@ref) so that the ChainRulesCore `rrule` can elide
KKT back-solves for components the user didn't request.
"""
struct Components{C} end

Components(c::Tuple{Vararg{Symbol}}) = Components{c}()

"""
    SolverSpec(forward; kind = :build)

Describe a parameter-to-solution pipeline.

- `forward::Function` — a unary callable `p -> (solver, obj_val)`. `solver`
  must be a **solved** `MadNLP.AbstractMadNLPSolver` (or subtype, e.g.
  `MadIPM.MPCSolver`) whose underlying NLP implements the `ParametricNLPModels`
  parametric sensitivity API. `obj_val` is the scalar objective at the
  optimum (used if the caller requests the `:obj` component).
- `kind` is an informational tag: `:build` means "rebuild the NLP from `p`
  every time" and `:update` means "reuse an existing workspace and call
  `update_standard_form!`" — it is not load-bearing for dispatch, but it is
  propagated into error messages and docstrings.

See [`differentiable_solve`](@ref) for how to consume a `SolverSpec`.
"""
struct SolverSpec{F<:Function}
    forward::F
    kind::Symbol
end
SolverSpec(forward::Function; kind::Symbol = :build) = SolverSpec(forward, kind)

# Internal: run the user's forward function and wrap the solver in a
# MadDiffSolver. We keep `obj_val` separate (rather than reading it back off
# the solver) so the user can tell us the exact scalar — MadNLP's internal
# `obj_val` already includes any scale/sign flips.
function _run_forward(spec::SolverSpec, p)
    out = spec.forward(p)
    solver, obj_val = _unpack_forward(out)
    sens = MadDiffSolver(solver)
    return sens, obj_val
end
_unpack_forward(t::Tuple{Any,Any}) = (t[1], t[2])
_unpack_forward(solver) = (solver, zero(_T(solver)))
_T(solver) = eltype(full(solver.x))

# Stub / error paths — filled in by the ChainRulesCore extension.
"""
    differentiable_solve(spec::SolverSpec; components = (:x,))

Build a ChainRulesCore-friendly closure `p -> NamedTuple` that solves the
parametric optimization problem described by `spec` and whose adjoint is the
implicit-function-theorem VJP through the KKT system at the optimum.

See the documentation of [`SolverSpec`](@ref) for how to build `spec`, and
[`Components`](@ref) for the meaning of `components`.

!!! note
    Requires `ChainRulesCore` to be loaded (Julia's package-extension
    mechanism activates `MadDiffChainRulesCoreExt` on import).
"""
function differentiable_solve end

# Internal entry point the extension supplies; declared here as an
# undefined generic function so the extension can add the single method
# without triggering the "method overwriting during precompilation" error.
function _solve_forward end

# Fallback error path: if the user calls `differentiable_solve` without
# having loaded ChainRulesCore, the extension's method won't be active and
# we fall into this stub.
function differentiable_solve(::Any; kwargs...)
    error(
        "`MadDiff.differentiable_solve` requires `ChainRulesCore`. Load it " *
        "(`using ChainRulesCore`) so the extension `MadDiffChainRulesCoreExt` " *
        "is registered."
    )
end
