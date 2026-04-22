# ============================================================================
# Adjoint (reverse-mode) KKT system — interface + shared reductions.
#
# Each concrete KKT system provides:
#   adjoint_mul!(w, a::AdjointKKT, x, α, β)     —  w ← α·Kᵀ·x + β·w
#   adjoint_solve_kkt!(a::AdjointKKT, w)        —  w ← K⁻ᵀ·w
#
# `AdjointRichardsonIterator` mirrors MadNLP's `RichardsonIterator`, holding
# an `AdjointKKT` so IR can reach whatever adjoint-only scratch `adjoint_mul!`
# needs (see `AdjointKKT` docstring). `adjoint_solve_refine!` is the
# reverse-mode analogue of MadNLP's `solve_refine!`.
# ============================================================================

function adjoint_solve_kkt! end
function adjoint_mul! end

# ---------- AdjointKKT: kkt + adjoint-only scratch ---------------------------

"""
    AdjointKKT{T, VT, KKT}

Bundles an `AbstractKKTSystem` with scratch that `adjoint_mul!` /
`adjoint_solve_kkt!` may need but that MadNLP's KKT types don't expose.

Current entries:

* `hess_diag :: VT` — cached `diag(kkt.hess_com)`. `adjoint_mul!` needs
  it to emulate `Symmetric(L,:L)·x` as `L·x + Lᵀ·x − diag(L).·x` — the
  GPU-safe replacement for the default `Symmetric(:L)` SpMV, which
  silently drops the upper-triangle reflection on `CuSparseMatrixCSC`
  (MadNLP has the same TODO on the forward side). CPU-equivalent.

* `scratch :: VT` — length-`n` scratch used by `_sym_hess_mul!` to hold
  `Lᵀ·x` before broadcast-accumulating. Needed because CUSPARSE's
  `mul!(dest, Transpose(CuSparseMatrixCSC), src, α, 1)` accumulation is
  unreliable (MadNLP's forward `solve_kkt!` has the same workaround).

For KKT types whose `adjoint_mul!` doesn't need either, `AdjointKKT` is
a near-zero-overhead wrapper: both fields are empty vectors and the
default delegations below forward to the bare-kkt methods.
"""
struct AdjointKKT{T, VT, KKT <: AbstractKKTSystem{T}}
    kkt::KKT
    hess_diag::VT
    scratch::VT
end

AdjointKKT(kkt::AbstractKKTSystem{T, VT}) where {T, VT} =
    AdjointKKT{T, VT, typeof(kkt)}(kkt, VT(undef, 0), VT(undef, 0))

# Default delegations — adjoint_mul!/solve_kkt! methods that don't need
# extra scratch keep their original `kkt`-only dispatches, and we forward
# from the wrapper. Concrete kkts whose adjoint *does* need scratch
# (every sparse kkt that applies `Symmetric(hess_com, :L)·x` — condensed,
# augmented, scaled_augmented, unreduced, hybrid_condensed) provide a
# specialized method on `AdjointKKT{T, VT, <:TheirKKT}`.
@inline adjoint_mul!(w, a::AdjointKKT, x, α = one(eltype(full(x))), β = zero(eltype(full(x)))) =
    adjoint_mul!(w, a.kkt, x, α, β)
@inline adjoint_solve_kkt!(a::AdjointKKT, w) = adjoint_solve_kkt!(a.kkt, w)

# ---------- AdjointRichardsonIterator ----------------------------------------

"""
    AdjointRichardsonIterator{T, A<:AdjointKKT{T}}

Reverse-mode counterpart to `MadNLP.RichardsonIterator`. Structurally
identical (opt / cnt / logger) but holds an [`AdjointKKT`](@ref) under the
`kkt` field, so IR loops dispatched on this iterator pick up the
adjoint-side `adjoint_mul!` / `adjoint_solve_kkt!` specialisations.
"""
struct AdjointRichardsonIterator{T, A <: AdjointKKT{T}}
    kkt::A
    opt::RichardsonOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function AdjointRichardsonIterator(a::AdjointKKT;
        opt::RichardsonOptions,
        cnt::MadNLPCounters,
        logger::MadNLPLogger,
    )
    return AdjointRichardsonIterator(a, opt, cnt, logger)
end

# ---------- symmetric Hessian SpMV, GPU-safe -------------------------------
#
# `_sym_hess_mul!(y, H, hess_diag, scratch, x, α, β)` computes
# `y ← α·Sym(L)·x + β·y` where `H` stores only the lower triangle (as both
# `SparseMatrixCSC` and `CuSparseMatrixCSC` do for MadNLP's `hess_com`).
#
# We *cannot* use `mul!(y, Symmetric(H, :L), x, α, β)`: on GPU, CUSPARSE's
# SpMV against the `Symmetric` wrapper silently drops the upper-triangle
# reflection — so the residual operator used by adjoint IR would be wrong
# (MadNLP has the same TODO on its forward side). The identity
#   Sym(L)·x = L·x + Lᵀ·x − diag(L).·x
# is backend-uniform: two plain SpMVs plus a broadcast subtraction.
#
# We also can't express `Lᵀ·x` as `mul!(y, H', x, α, 1)` and let it accumulate
# into `y` alongside `L·x`: CUSPARSE's `mul!(dest, Transpose(CuSparseMatrixCSC),
# src, α, 1)` is unreliable (same reason forward `solve_kkt!` in condensed.jl
# writes its jt_csc' SpMV to a fresh buffer). We therefore route `Lᵀ·x` through
# the caller-provided `scratch` buffer with `β = 0`, then broadcast-accumulate.
#
# Callers pre-allocate `hess_diag = diag(L)` and `scratch::VT` of length `n`
# once at `AdjointKKT` construction time; both live in the wrapper so the hot
# loop is allocation-free.

@inline function _sym_hess_mul!(y, H, hess_diag, scratch, x, alpha, beta)
    mul!(scratch, H', x, alpha, zero(alpha))   # scratch ← α·Lᵀ·x
    mul!(y,       H,  x, alpha, beta)          # y ← α·L·x + β·y
    y .+= scratch .- alpha .* hess_diag .* x
    return y
end

# ---------- shared bound-dual manipulations ----------

function _adjoint_kktmul!(w, x, reg, du_diag, l_lower, u_lower,
                          l_diag, u_diag, alpha, beta)
    primal(w) .+= alpha .* reg      .* primal(x)
    dual(w)   .+= alpha .* du_diag  .* dual(x)
    w.xp_lr   .+= alpha .* l_lower  .* dual_lb(x)
    w.xp_ur   .+= alpha .* u_lower  .* dual_ub(x)
    dual_lb(w) .= beta .* dual_lb(w) .+ alpha .* (.-x.xp_lr .- l_diag .* dual_lb(x))
    dual_ub(w) .= beta .* dual_ub(w) .+ alpha .* ( x.xp_ur .+ u_diag .* dual_ub(x))
    return nothing
end

function _adjoint_finish_bounds!(kkt, w)
    dlb, dub = dual_lb(w), dual_ub(w)
    w.xp_lr .+= (kkt.l_lower ./ kkt.l_diag) .* dlb
    w.xp_ur .-= (kkt.u_lower ./ kkt.u_diag) .* dub
    dlb .= .-dlb ./ kkt.l_diag
    dub .=   dub ./ kkt.u_diag
    return nothing
end

function _adjoint_reduce_rhs!(kkt, w)
    dual_lb(w) .-= w.xp_lr ./ kkt.l_diag
    dual_ub(w) .-= w.xp_ur ./ kkt.u_diag
    return nothing
end

# ---------- refined adjoint solve ----------

function adjoint_solve_refine!(
    x::VT, iterator::AdjointRichardsonIterator{T}, b::VT, w::VT,
) where {T, VT}
    @debug(iterator.logger, "Adjoint iterative solver initiated")

    norm_b = norm(full(b), Inf)
    residual_ratio = zero(T)
    fill!(full(x), zero(T))

    if norm_b != zero(T)
        copyto!(full(w), full(b))
        iterator.cnt.ir = 0

        while true
            adjoint_solve_kkt!(iterator.kkt, w)
            axpy!(one(T), full(w), full(x))
            copyto!(full(w), full(b))
            adjoint_mul!(w, iterator.kkt, x, -one(T), one(T))

            norm_w = norm(full(w), Inf)
            norm_x = norm(full(x), Inf)
            residual_ratio = norm_w / (min(norm_x, 1e6 * norm_b) + norm_b)

            mod(iterator.cnt.ir, 10) == 0 &&
                @debug(iterator.logger, "iterator.cnt.ir ||res||")
            @debug(iterator.logger,
                @sprintf("%4i %6.2e", iterator.cnt.ir, residual_ratio))
            iterator.cnt.ir += 1

            iterator.cnt.ir >= iterator.opt.richardson_max_iter && break
            residual_ratio < iterator.opt.richardson_tol && break
        end
    end

    @debug(iterator.logger,
        @sprintf("Adjoint iterative solver terminated with %4i refinement steps, residual = %6.2e",
                 iterator.cnt.ir, residual_ratio))
    return residual_ratio < iterator.opt.richardson_acceptable_tol
end
