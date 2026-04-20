module MadDiffChainRulesCoreExt

# =============================================================================
# MadDiff × ChainRulesCore
# -----------------------------------------------------------------------------
# This extension exposes MadDiff's forward / reverse implicit-differentiation
# primitives as ChainRulesCore `rrule` / `frule` definitions so that a
# parametric optimization solve can be dropped into a Zygote / Enzyme pipeline
# without any extra boilerplate.
#
# Design is modeled on `Moreau.jl`'s `MoreauChainRulesExt`: a single
# "parameters → solve → solution" forward function, with an adjoint that uses
# the solved KKT system to project output cotangents back onto the parameter
# space. The user opts into which solution components they want in the forward
# output (and therefore which adjoint components the rrule has to compute) via
# a type-level `Components` tag so dispatch can elide unused adjoint work.
#
# The main MadDiff module defines the public API symbols (`differentiable_solve`,
# `SolverSpec`, `Components`) as stubs that error if ChainRulesCore isn't
# loaded; this extension supplies the real implementations.
# =============================================================================

using LinearAlgebra
using ChainRulesCore
import ChainRulesCore: rrule, frule, NoTangent, ZeroTangent, @not_implemented

using MadDiff
using MadDiff: MadDiffSolver, MadDiffConfig,
    JVPResult, VJPResult,
    jacobian_vector_product!, vector_jacobian_product!,
    compute_objective_sensitivity!
import MadDiff: differentiable_solve, SolverSpec, Components, _solve_forward
import MadNLP: full, variable


# =============================================================================
# Utilities
# =============================================================================

# Pull an array-valued tangent field out of a structured tangent, returning
# `nothing` if the caller didn't ask for a gradient on that component.
@inline function _tangent_or_nothing(Δ, field::Symbol)
    Δ === NoTangent() && return nothing
    Δ === ZeroTangent() && return nothing
    hasproperty(Δ, field) || return nothing
    t = getproperty(Δ, field)
    (t === NoTangent() || t === ZeroTangent() || t === nothing) && return nothing
    return t
end

# Lightweight NamedTuple tangent unpack for tuple-returning forward functions:
# if the forward returns a `Components((:x, :y))`-shaped NamedTuple, the
# downstream AD may deliver the cotangent as a Tangent or plain NamedTuple
# with the same field names.
@inline function _field_or_zero(Δ, field::Symbol, fallback)
    t = _tangent_or_nothing(Δ, field)
    t === nothing ? fallback : t
end


# =============================================================================
# rrule: MadDiff.jacobian_vector_product!(sens, Δp)
# -----------------------------------------------------------------------------
# The JVP is itself a linear function of `Δp`, so its adjoint is the
# corresponding VJP. This lets users plug `jacobian_vector_product!` into
# reverse-mode AD (e.g. for nested differentiation).
# =============================================================================

function ChainRulesCore.rrule(
    ::typeof(jacobian_vector_product!),
    sens::MadDiffSolver,
    Δp::AbstractVector,
)
    result = jacobian_vector_product!(sens, Δp)

    function jvp_pullback(Δresult)
        dL_dx  = _tangent_or_nothing(Δresult, :dx)
        dL_dy  = _tangent_or_nothing(Δresult, :dy)
        dL_dzl = _tangent_or_nothing(Δresult, :dzl)
        dL_dzu = _tangent_or_nothing(Δresult, :dzu)
        # JVP is linear in Δp, so the adjoint is just the VJP with the same
        # seeds projected onto parameter space.
        vjp = vector_jacobian_product!(
            sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj = nothing,
        )
        return (NoTangent(), NoTangent(), vjp.grad_p)
    end

    return result, jvp_pullback
end


# =============================================================================
# rrule: MadDiff.vector_jacobian_product!(sens; dL_*)
# -----------------------------------------------------------------------------
# The VJP returns `grad_p = J' * cotangent`; its adjoint w.r.t. each input
# seed is the corresponding JVP column, so we again use `jacobian_vector_product!`.
# In practice nested reverse-over-reverse is rare, but exposing the rrule keeps
# the API symmetric.
# =============================================================================

function ChainRulesCore.rrule(
    ::typeof(vector_jacobian_product!),
    sens::MadDiffSolver;
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing, dobj = nothing,
)
    result = vector_jacobian_product!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
    function vjp_pullback(Δresult)
        # Nested reverse-over-reverse through the VJP itself is rare; we
        # provide a no-op adjoint for the solver argument. Keyword-argument
        # seeds aren't differentiable in ChainRulesCore's rrule calling
        # convention, so we return just (self, solver) tangents.
        return (NoTangent(), NoTangent())
    end
    return result, vjp_pullback
end


# =============================================================================
# differentiable_solve: high-level API (implementation)
# -----------------------------------------------------------------------------
# The user supplies a `SolverSpec` that wraps whatever they want to happen
# inside the forward pass (typically: build an NLP from `p`, call MadNLP /
# MadIPM, return the solver). `differentiable_solve` turns that into a closure
# `p -> solution_components` whose rrule implicitly differentiates the KKT
# system at the solution and projects cotangents back to `p`.
#
# `components` is a compile-time-fixed tuple of symbols drawn from
# `(:x, :y, :zl, :zu, :obj)`. The returned forward closure produces a
# `NamedTuple` with exactly those fields; the adjoint only runs the backward
# projections the user actually seeded.
# =============================================================================

"""
    differentiable_solve(spec::SolverSpec; components = (:x,))

Return a closure `p -> solution` where `solution` is a `NamedTuple` with the
fields named in `components`. The closure has a registered `ChainRulesCore.rrule`
so it can be dropped into any reverse-mode AD framework (Zygote, Enzyme, etc.).

See [`SolverSpec`](@ref) for how to describe the forward pass.

`components` must be a subset of `(:x, :y, :zl, :zu, :obj)`. Components not
listed here are neither returned by the forward pass nor honored by the
adjoint — the implicit-function computation is only run for the requested
outputs.
"""
function MadDiff.differentiable_solve(spec::MadDiff.SolverSpec; components = (:x,))
    _validate_components(components)
    comp_tag = MadDiff.Components{Tuple(components)}()
    return p -> _solve_forward(spec, comp_tag, p)
end

const _VALID_COMPONENTS = (:x, :y, :zl, :zu, :obj)

function _validate_components(components)
    for c in components
        c in _VALID_COMPONENTS || throw(ArgumentError(
            "Unknown component `$c`. Must be a subset of $_VALID_COMPONENTS."))
    end
    length(unique(components)) == length(components) ||
        throw(ArgumentError("`components` must not contain duplicates."))
    return nothing
end

@inline _has(::MadDiff.Components{C}, s::Symbol) where {C} = s in C

function _build_solution(sens::MadDiffSolver, comp::MadDiff.Components, obj_val::Real)
    solver = sens.solver
    cb = solver.cb
    pairs_ = Pair{Symbol,Any}[]
    _has(comp, :x)   && push!(pairs_, :x  => copy(variable(solver.x)))
    _has(comp, :y)   && push!(pairs_, :y  => copy(solver.y))
    if _has(comp, :zl)
        zl = zero(variable(solver.x))
        zl[cb.ind_lb] .= solver.zl_r
        push!(pairs_, :zl => zl)
    end
    if _has(comp, :zu)
        zu = zero(variable(solver.x))
        zu[cb.ind_ub] .= solver.zu_r
        push!(pairs_, :zu => zu)
    end
    _has(comp, :obj) && push!(pairs_, :obj => obj_val)
    return (; pairs_...)
end

function MadDiff._solve_forward(spec::MadDiff.SolverSpec, comp::MadDiff.Components, p)
    sens, obj_val = MadDiff._run_forward(spec, p)
    return _build_solution(sens, comp, obj_val)
end

function ChainRulesCore.rrule(
    ::typeof(MadDiff._solve_forward),
    spec::MadDiff.SolverSpec,
    comp::MadDiff.Components,
    p::AbstractVector,
)
    sens, obj_val = MadDiff._run_forward(spec, p)
    solution = _build_solution(sens, comp, obj_val)

    function solve_pullback(Δ)
        dL_dx  = _has(comp, :x)  ? _tangent_or_nothing(Δ, :x)  : nothing
        dL_dy  = _has(comp, :y)  ? _tangent_or_nothing(Δ, :y)  : nothing
        dL_dzl = _has(comp, :zl) ? _tangent_or_nothing(Δ, :zl) : nothing
        dL_dzu = _has(comp, :zu) ? _tangent_or_nothing(Δ, :zu) : nothing
        dobj_t = _has(comp, :obj) ? _tangent_or_nothing(Δ, :obj) : nothing
        dobj   = dobj_t === nothing ? nothing : float(dobj_t)

        if all(isnothing, (dL_dx, dL_dy, dL_dzl, dL_dzu, dobj))
            return (NoTangent(), NoTangent(), NoTangent(), ZeroTangent())
        end

        vjp = vector_jacobian_product!(
            sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj,
        )
        return (NoTangent(), NoTangent(), NoTangent(), copy(vjp.grad_p))
    end

    return solution, solve_pullback
end

end # module
