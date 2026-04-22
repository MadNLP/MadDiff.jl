# ============================================================================
# Internal helpers.
# ============================================================================

const SUCCESSFUL_STATUSES = (SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL)

function assert_solved_and_feasible(solver::AbstractMadNLPSolver)
    solver.status in SUCCESSFUL_STATUSES ||
        error("MadDiff: solver did not converge (status = $(solver.status)).")
    return nothing
end

_get_wrapper_type(x) = Base.typename(typeof(x)).wrapper

_needs_new_kkt(cfg::MadDiffConfig) =
    cfg.kkt_system    !== nothing ||
    cfg.kkt_options   !== nothing ||
    cfg.linear_solver !== nothing ||
    cfg.linear_solver_options !== nothing

# Transparent shim around a MadNLPSolver that swaps its `kkt` field. Used when
# MadDiff refactorises into a fresh KKT system while wanting MadNLP internals
# (inertia correction, counters) to drive the factorisation.
struct _SensitivitySolverShim{T, S<:AbstractMadNLPSolver{T}, K<:AbstractKKTSystem{T}} <: AbstractMadNLPSolver{T}
    inner::S
    kkt::K
end

Base.getproperty(s::_SensitivitySolverShim, name::Symbol) =
    name === :inner ? getfield(s, :inner) :
    name === :kkt   ? getfield(s, :kkt)   :
                      getproperty(getfield(s, :inner), name)

Base.setproperty!(s::_SensitivitySolverShim, name::Symbol, value) =
    name === :inner ? setfield!(s, :inner, value) :
    name === :kkt   ? setfield!(s, :kkt,   value) :
                      setproperty!(getfield(s, :inner), name, value)

# ---------- device adaptation ----------
# Used by the JVP/VJP entry points to marshal host-typed seeds onto the
# solver's device (no-op when already matching) so users can pass plain
# `Vector{Float64}` seeds against a GPU-backed `MadDiffSolver`.
#
# `_solver_proto(sens)` returns an array whose `similar` yields the solver's
# native backend. `_adapt_device(proto, x)` builds an x-shaped array on that
# backend (or returns `x` verbatim when already matching).

@inline _solver_proto(sens::MadDiffSolver) = full(sens.solver.x)

@inline _adapt_device(::VT, x::VT) where {VT <: AbstractArray} = x
@inline _adapt_device(proto::AbstractArray, x::AbstractArray) =
    (y = similar(proto, eltype(x), size(x)...); copyto!(y, x); y)

# ---------- CSC diagonal extraction ----------
# Used by `AdjointKKT(::SparseCondensedKKTSystem)` to cache `diag(hess_com)`
# for the GPU-safe `Symmetric(L,:L)·x` emulation in `adjoint_mul!` (see
# comment at the top of `KKT/Sparse/condensed.jl`). We only need it once, at
# construction time, so we do the structural scan on the host (cheap —
# `colptr`/`rowval` are small) and gather `nzval` on whichever backend `A`
# lives on. Works uniformly for `SparseMatrixCSC` and `CuSparseMatrixCSC`.

function _csc_diag(A::AbstractMatrix{T}) where {T}
    n = LinearAlgebra.checksquare(A)
    colptr = Array(SparseArrays.getcolptr(A))
    rowval = Array(SparseArrays.rowvals(A))
    idx_h  = zeros(Int, n)   # 0 ⇒ missing diagonal entry
    @inbounds for j in 1:n
        for k in colptr[j]:colptr[j+1]-1
            if rowval[k] == j
                idx_h[j] = k
                break
            end
        end
    end

    nz = SparseArrays.nonzeros(A)  # same backend as A
    d  = similar(nz, n)
    nz_h = Array(nz)
    d_h  = [i == 0 ? zero(T) : nz_h[i] for i in idx_h]
    copyto!(d, d_h)
    return d
end

# ---------- parametric-capability probes ----------

has_hess_param(nlp) = get_nnzhp(nlp)      != 0
has_jac_param(nlp)  = get_nnzjp(nlp)      != 0
has_lvar_param(nlp) = get_nnzjplvar(nlp)  != 0
has_uvar_param(nlp) = get_nnzjpuvar(nlp)  != 0
has_lcon_param(nlp) = get_nnzjplcon(nlp)  != 0
has_ucon_param(nlp) = get_nnzjpucon(nlp)  != 0
has_grad_param(nlp) = get_nnzgp(nlp)      != 0
