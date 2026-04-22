module MadDiffChainRulesCoreExt

# ============================================================================
# MadDiff × ChainRulesCore.
#
# Wires MadDiff's implicit-differentiation primitives into reverse-mode AD:
#
#   _solve_forward(spec, comp, p)   : forward pass — solve, emit the requested
#                                     components as a `NamedTuple`.
#   rrule(_solve_forward, ...)      : primary entry. Pullback seeds
#                                     `vector_jacobian_product!` with only the
#                                     cotangents the user asked for and skips
#                                     the adjoint solves for unlisted components.
#   rrule(jacobian_vector_product!) : adjoint-of-linear-map trick (the JVP's
#                                     adjoint is the VJP with the output
#                                     cotangent as seed) — enables nested AD.
#   rrule(vector_jacobian_product!) : no-op pullback for reverse-over-reverse.
# ============================================================================

using ChainRulesCore
using MadDiff

import ChainRulesCore: NoTangent, ZeroTangent, rrule
import MadDiff: BatchSolverSpec, Components, MadDiffSolver, SolverSpec,
    _batch_solve_forward, _run_batch_forward, _run_forward, _solve_forward,
    batch_differentiable_solve, differentiable_solve,
    jacobian_vector_product!, vector_jacobian_product!
import MadNLP: full, variable

const _VALID_COMPONENTS = (:x, :y, :zl, :zu, :obj)

# ---------- tangent / component helpers ----------

# Collapse `NoTangent` / `ZeroTangent` / missing / `nothing` to `nothing` so
# `vector_jacobian_product!` can skip the corresponding adjoint back-solve.
@inline function _tangent(Δ, field::Symbol)
    (Δ === NoTangent() || Δ === ZeroTangent()) && return nothing
    hasproperty(Δ, field) || return nothing
    t = getproperty(Δ, field)
    (t === NoTangent() || t === ZeroTangent() || t === nothing) && return nothing
    return t
end

@inline _has(::Components{C}, s::Symbol) where {C} = s in C

@inline _tangent_if(comp, Δ, field) = _has(comp, field) ? _tangent(Δ, field) : nothing

function _validate_components(components)
    for c in components
        c in _VALID_COMPONENTS || throw(ArgumentError(
            "MadDiff: unknown component `$c`; must be one of $_VALID_COMPONENTS."))
    end
    length(unique(components)) == length(components) ||
        throw(ArgumentError("MadDiff: `components` must not contain duplicates."))
    return nothing
end

# ---------- forward: assemble the solution NamedTuple ----------

# Build the requested-components NamedTuple without an `Any`-typed accumulator.
# Each branch returns a small NamedTuple; they merge into the final tuple with
# type-stable concrete value types (inferrable field-by-field).
function _build_solution(sens::MadDiffSolver, comp::Components, obj_val::Real)
    solver = sens.solver
    cb     = solver.cb
    out    = (;)

    _has(comp, :x) && (out = merge(out, (; x = copy(variable(solver.x)))))
    _has(comp, :y) && (out = merge(out, (; y = copy(solver.y))))

    # `ind_lb`/`ind_ub` index into the full primal-plus-slack space, not the
    # variable subspace — slacks carry their own bound multipliers.
    if _has(comp, :zl)
        zl = zero(full(solver.x)); zl[cb.ind_lb] .= solver.zl_r
        out = merge(out, (; zl))
    end
    if _has(comp, :zu)
        zu = zero(full(solver.x)); zu[cb.ind_ub] .= solver.zu_r
        out = merge(out, (; zu))
    end

    _has(comp, :obj) && (out = merge(out, (; obj = obj_val)))
    return out
end

function MadDiff._solve_forward(spec::SolverSpec, comp::Components, p)
    sens, obj_val = _run_forward(spec, p)
    return _build_solution(sens, comp, obj_val)
end

# ---------- public entry ----------

function MadDiff.differentiable_solve(spec::SolverSpec; components = (:x,))
    _validate_components(components)
    comp = Components(Tuple(components))
    return p -> _solve_forward(spec, comp, p)
end

# ---------- rrules ----------

function ChainRulesCore.rrule(
    ::typeof(MadDiff._solve_forward),
    spec::SolverSpec, comp::Components, p::AbstractVector,
)
    sens, obj_val = _run_forward(spec, p)
    solution      = _build_solution(sens, comp, obj_val)

    function solve_pullback(Δ)
        dL_dx  = _tangent_if(comp, Δ, :x)
        dL_dy  = _tangent_if(comp, Δ, :y)
        dL_dzl = _tangent_if(comp, Δ, :zl)
        dL_dzu = _tangent_if(comp, Δ, :zu)
        dobj_t = _tangent_if(comp, Δ, :obj)
        dobj   = dobj_t === nothing ? nothing : float(dobj_t)

        all(isnothing, (dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)) &&
            return (NoTangent(), NoTangent(), NoTangent(), ZeroTangent())

        vjp = vector_jacobian_product!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
        return (NoTangent(), NoTangent(), NoTangent(), copy(vjp.grad_p))
    end

    return solution, solve_pullback
end

function ChainRulesCore.rrule(
    ::typeof(jacobian_vector_product!), sens::MadDiffSolver, Δp::AbstractVector,
)
    result = jacobian_vector_product!(sens, Δp)

    function jvp_pullback(Δresult)
        vjp = vector_jacobian_product!(sens;
            dL_dx  = _tangent(Δresult, :dx),
            dL_dy  = _tangent(Δresult, :dy),
            dL_dzl = _tangent(Δresult, :dzl),
            dL_dzu = _tangent(Δresult, :dzu),
            dobj   = nothing,
        )
        return (NoTangent(), NoTangent(), copy(vjp.grad_p))
    end

    return result, jvp_pullback
end

function ChainRulesCore.rrule(
    ::typeof(vector_jacobian_product!),
    sens::MadDiffSolver;
    dL_dx = nothing, dL_dy = nothing, dL_dzl = nothing, dL_dzu = nothing,
    dobj  = nothing,
)
    result = vector_jacobian_product!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
    return result, _ -> (NoTangent(), NoTangent())
end

# ============================================================================
# Batch variant — same shape, matrix-valued seeds.
# ============================================================================

function _build_batch_solution(sens, comp::Components, obj_vec)
    solver = sens.solver
    out    = (;)
    _has(comp, :x)   && (out = merge(out, (; x   = copy(MadNLP.variable(solver.state.x)))))
    _has(comp, :y)   && (out = merge(out, (; y   = copy(MadNLP.full(solver.state.y)))))
    _has(comp, :zl)  && (out = merge(out, (; zl  = copy(MadNLP.full(solver.state.zl)))))
    _has(comp, :zu)  && (out = merge(out, (; zu  = copy(MadNLP.full(solver.state.zu)))))
    _has(comp, :obj) && (out = merge(out, (; obj = copy(obj_vec))))
    return out
end

function MadDiff._batch_solve_forward(spec::BatchSolverSpec, comp::Components, p)
    sens, obj_vec = _run_batch_forward(spec, p)
    return _build_batch_solution(sens, comp, obj_vec)
end

function MadDiff.batch_differentiable_solve(spec::BatchSolverSpec; components = (:x,))
    _validate_components(components)
    comp = Components(Tuple(components))
    return p -> _batch_solve_forward(spec, comp, p)
end

function ChainRulesCore.rrule(
    ::typeof(MadDiff._batch_solve_forward),
    spec::BatchSolverSpec, comp::Components, p::AbstractMatrix,
)
    sens, obj_vec = _run_batch_forward(spec, p)
    solution      = _build_batch_solution(sens, comp, obj_vec)

    function batch_pullback(Δ)
        dL_dx  = _tangent_if(comp, Δ, :x)
        dL_dy  = _tangent_if(comp, Δ, :y)
        dL_dzl = _tangent_if(comp, Δ, :zl)
        dL_dzu = _tangent_if(comp, Δ, :zu)
        dobj_t = _tangent_if(comp, Δ, :obj)
        dobj   = dobj_t === nothing ? nothing : float.(dobj_t)

        all(isnothing, (dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)) &&
            return (NoTangent(), NoTangent(), NoTangent(), ZeroTangent())

        vjp = vector_jacobian_product!(sens; dL_dx, dL_dy, dL_dzl, dL_dzu, dobj)
        return (NoTangent(), NoTangent(), NoTangent(), copy(vjp.grad_p))
    end

    return solution, batch_pullback
end

end # module
