module MadIPMExt

using LinearAlgebra: mul!
import MadDiff
import MadNLP
import MadIPM: NormalKKTSystem

MadDiff._get_bound_scale(kkt::NormalKKTSystem) = (kkt.l_lower, kkt.u_lower)

function MadDiff._vjp_solve!(kkt::NormalKKTSystem, w::MadNLP.AbstractKKTVector)
    # no reduce_rhs!
    r1 = kkt.buffer_n
    r2 = kkt.buffer_m
    Σ = kkt.pr_diag

    wx = MadNLP.primal(w)
    wy = MadNLP.dual(w)

    # Build RHS
    r1 .= wx ./ Σ                          # Σ⁻¹ r₁
    r2 .= wy                               # r₂
    mul!(r2, kkt.AT', r1, 1.0, -1.0)       # A Σ⁻¹ r₁ - r₂
    # Solve normal KKT system
    MadNLP.solve!(kkt.linear_solver, r2)   # Δy
    # Unpack solution
    wy .= r2                               # Δy
    r1 .= wx                               # r₁
    mul!(r1, kkt.AT, wy, -1.0, 1.0)        # r₁ - Aᵀ Δy
    wx .= r1 ./ Σ                          # Σ⁻¹ (r₁ - Aᵀ Δy)

    # no finish_aug_solve!
    return w
end

end # module
