using MadNLP, MadDiff, LinearAlgebra, Random
using NLPModels, QuadraticModels

# ─── Helper: QP with active and inactive constraints ────────────────────
# min 0.5 x^2  s.t.  x >= 1 (active),  x >= -10 (inactive)
function _make_qp_with_inactive()
    c = [0.0]
    Hrows = [1]; Hcols = [1]; Hvals = [1.0]
    Arows = [1, 2]; Acols = [1, 1]; Avals = [1.0, 1.0]
    lcon = [1.0, -10.0]
    ucon = [Inf, Inf]
    lvar = [-Inf]
    uvar = [Inf]
    x0 = [1.0]
    return QuadraticModel(c, Hrows, Hcols, Hvals; Arows, Acols, Avals,
                          lcon, ucon, lvar, uvar, x0, name="qp_active_inactive")
end

# ─── Helper: simple QP (all constraints active) ────────────────────────
# min 0.5 x^2  s.t.  x >= 1
function _make_simple_qp()
    c = [0.0]
    Hrows = [1]; Hcols = [1]; Hvals = [1.0]
    Arows = [1]; Acols = [1]; Avals = [1.0]
    lcon = [1.0]
    ucon = [Inf]
    lvar = [-Inf]
    uvar = [Inf]
    x0 = [1.0]
    return QuadraticModel(c, Hrows, Hcols, Hvals; Arows, Acols, Avals,
                          lcon, ucon, lvar, uvar, x0, name="qp_simple")
end

function setup_kkt(KKTType, solver, ls)
    kkt = MadNLP.create_kkt_system(KKTType, solver.cb, ls; solver=solver)
    MadNLP.initialize!(kkt)
    MadDiff.eval_jac_wrapper!(solver, kkt, solver.x)
    MadDiff.eval_lag_hess_wrapper!(solver, kkt, solver.x, solver.y)
    MadNLP.set_aug_diagonal!(kkt, solver)
    MadDiff._ensure_quasi_definite!(kkt)
    MadNLP.compress_hessian!(kkt)
    MadNLP.compress_jacobian!(kkt)
    MadNLP.build_kkt!(kkt)
    MadNLP.factorize_kkt!(kkt)
    return kkt
end

# ─── Test 1: Construction ─────────────────────────────────────────────
println("=== Test 1: DenseCondensedActiveSetKKT construction ===")
nlp = _make_simple_qp()
solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR)
MadNLP.solve!(solver)
println("  x* = ", solver.x.x)

kkt_c = setup_kkt(MadDiff.DenseCondensedActiveSetKKTSystem, solver, MadNLP.LapackCPUSolver)
println("  n_active=$(kkt_c.n_active), ns_active=$(kkt_c.ns_active), n_eq_active=$(kkt_c.n_eq_active)")
println("  aug_com size: ", size(kkt_c.aug_com))
println("  PASS")

# ─── Test 2: Forward solve matches DenseActiveSetKKT ──────────────────
println("\n=== Test 2: Forward solve matches DenseActiveSet ===")
kkt_ref = setup_kkt(MadDiff.DenseActiveSetKKTSystem, solver, MadNLP.LapackCPUSolver)

Random.seed!(42)
rhs_c = MadNLP.UnreducedKKTVector(kkt_c)
rhs_r = MadNLP.UnreducedKKTVector(kkt_ref)
seed = randn(length(MadNLP.full(rhs_c)))
copyto!(MadNLP.full(rhs_c), seed)
copyto!(MadNLP.full(rhs_r), seed)

MadNLP.solve_kkt!(kkt_c, rhs_c)
MadNLP.solve_kkt!(kkt_ref, rhs_r)

diff = maximum(abs.(MadNLP.full(rhs_c) .- MadNLP.full(rhs_r)))
println("  Forward solve max diff: ", diff)
@assert diff < 1e-6 "Forward solve mismatch: $diff"
println("  PASS")

# ─── Test 3: Adjoint solve identity ──────────────────────────────────
println("\n=== Test 3: Adjoint solve identity ===")
v = MadNLP.UnreducedKKTVector(kkt_c)
w = MadNLP.UnreducedKKTVector(kkt_c)
Random.seed!(123)
randn!(MadNLP.full(v))
randn!(MadNLP.full(w))

x_fwd = copy(w)
y_adj = copy(v)
MadNLP.solve_kkt!(kkt_c, x_fwd)
MadDiff.adjoint_solve_kkt!(kkt_c, y_adj)

lhs = dot(v, x_fwd)
rhs_val = dot(y_adj, w)
diff3 = abs(lhs - rhs_val)
println("  <v, K^-1 w> = ", lhs)
println("  <K^-T v, w> = ", rhs_val)
println("  diff = ", diff3)
@assert diff3 < 1e-6 "Adjoint solve identity failed: $diff3"
println("  PASS")

# ─── Test 4: With inactive constraints ──────────────────────────────
println("\n=== Test 4: Inactive constraints ===")
nlp2 = _make_qp_with_inactive()
solver2 = MadNLP.MadNLPSolver(nlp2; print_level=MadNLP.ERROR)
MadNLP.solve!(solver2)
println("  x* = ", solver2.x.x)

kkt_c2 = setup_kkt(MadDiff.DenseCondensedActiveSetKKTSystem, solver2, MadNLP.LapackCPUSolver)
println("  n_active=$(kkt_c2.n_active), ns_active=$(kkt_c2.ns_active), n_eq_active=$(kkt_c2.n_eq_active)")
println("  aug_com size: ", size(kkt_c2.aug_com))
println("  PASS")

# ─── Test 5: SparseCondensedActiveSetKKT ──────────────────────────────
println("\n=== Test 5: SparseCondensedActiveSetKKT construction ===")
kkt_sc = setup_kkt(MadDiff.SparseCondensedActiveSetKKTSystem, solver, MadNLP.LDLSolver)
println("  n_active=$(kkt_sc.n_active), ns_active=$(kkt_sc.ns_active), n_eq_active=$(kkt_sc.n_eq_active)")
println("  aug_com size: ", size(kkt_sc.aug_com))
println("  PASS")

# ─── Test 6: Sparse forward solve matches Sparse active set ──────────
println("\n=== Test 6: Sparse forward solve ===")
kkt_sr = setup_kkt(MadDiff.SparseActiveSetKKTSystem, solver, MadNLP.LDLSolver)

Random.seed!(42)
rhs_sc = MadNLP.UnreducedKKTVector(kkt_sc)
rhs_sr = MadNLP.UnreducedKKTVector(kkt_sr)
seed2 = randn(length(MadNLP.full(rhs_sc)))
copyto!(MadNLP.full(rhs_sc), seed2)
copyto!(MadNLP.full(rhs_sr), seed2)

MadNLP.solve_kkt!(kkt_sc, rhs_sc)
MadNLP.solve_kkt!(kkt_sr, rhs_sr)

diff6 = maximum(abs.(MadNLP.full(rhs_sc) .- MadNLP.full(rhs_sr)))
println("  Sparse forward solve diff: ", diff6)
@assert diff6 < 1e-6 "Sparse forward solve mismatch: $diff6"
println("  PASS")

# ─── Test 7: Sparse adjoint solve identity ────────────────────────────
println("\n=== Test 7: Sparse adjoint solve identity ===")
v7 = MadNLP.UnreducedKKTVector(kkt_sc)
w7 = MadNLP.UnreducedKKTVector(kkt_sc)
Random.seed!(456)
randn!(MadNLP.full(v7))
randn!(MadNLP.full(w7))

x_fwd7 = copy(w7)
y_adj7 = copy(v7)
MadNLP.solve_kkt!(kkt_sc, x_fwd7)
MadDiff.adjoint_solve_kkt!(kkt_sc, y_adj7)

lhs7 = dot(v7, x_fwd7)
rhs7 = dot(y_adj7, w7)
diff7 = abs(lhs7 - rhs7)
println("  diff = ", diff7)
@assert diff7 < 1e-6 "Sparse adjoint identity failed: $diff7"
println("  PASS")

# ─── Test 8: Sparse with inactive constraints ────────────────────────
println("\n=== Test 8: Sparse with inactive constraints ===")
kkt_sc2 = setup_kkt(MadDiff.SparseCondensedActiveSetKKTSystem, solver2, MadNLP.LDLSolver)
println("  n_active=$(kkt_sc2.n_active), ns_active=$(kkt_sc2.ns_active), n_eq_active=$(kkt_sc2.n_eq_active)")
println("  PASS")

# ─── Test 9: QP with equality constraint ─────────────────────────────
# min 0.5(x1^2 + x2^2) s.t. x1 + x2 = 2, x1 >= 0.5
println("\n=== Test 9: Dense with equality constraint ===")
function _make_qp_with_eq()
    c = [0.0, 0.0]
    Hrows = [1, 2]; Hcols = [1, 2]; Hvals = [1.0, 1.0]
    Arows = [1, 1, 2]; Acols = [1, 2, 1]; Avals = [1.0, 1.0, 1.0]
    lcon = [2.0, 0.5]
    ucon = [2.0, Inf]
    lvar = [-Inf, -Inf]
    uvar = [Inf, Inf]
    x0 = [1.0, 1.0]
    QuadraticModel(c, Hrows, Hcols, Hvals; Arows, Acols, Avals, lcon, ucon, lvar, uvar, x0)
end
nlp_eq = _make_qp_with_eq()
solver_eq = MadNLP.MadNLPSolver(nlp_eq; print_level=MadNLP.ERROR)
MadNLP.solve!(solver_eq)
println("  x* = ", solver_eq.x.x)

kkt_eq = setup_kkt(MadDiff.DenseCondensedActiveSetKKTSystem, solver_eq, MadNLP.LapackCPUSolver)
println("  n_active=$(kkt_eq.n_active), ns_active=$(kkt_eq.ns_active), n_eq_active=$(kkt_eq.n_eq_active)")
println("  aug_com size: ", size(kkt_eq.aug_com))

# Adjoint identity
v9 = MadNLP.UnreducedKKTVector(kkt_eq)
w9 = MadNLP.UnreducedKKTVector(kkt_eq)
Random.seed!(789)
randn!(MadNLP.full(v9))
randn!(MadNLP.full(w9))
x_fwd9 = copy(w9)
y_adj9 = copy(v9)
MadNLP.solve_kkt!(kkt_eq, x_fwd9)
MadDiff.adjoint_solve_kkt!(kkt_eq, y_adj9)
lhs9 = dot(v9, x_fwd9)
rhs9 = dot(y_adj9, w9)
diff9 = abs(lhs9 - rhs9)
println("  adjoint identity diff = ", diff9)
@assert diff9 < 1e-5 "Eq adjoint identity failed: $diff9"
println("  PASS")

# ─── Test 10: Sparse with equality constraint ─────────────────────────
println("\n=== Test 10: Sparse with equality constraint ===")
kkt_seq = setup_kkt(MadDiff.SparseCondensedActiveSetKKTSystem, solver_eq, MadNLP.LDLSolver)
println("  n_active=$(kkt_seq.n_active), ns_active=$(kkt_seq.ns_active), n_eq_active=$(kkt_seq.n_eq_active)")
println("  aug_com size: ", size(kkt_seq.aug_com))

v10 = MadNLP.UnreducedKKTVector(kkt_seq)
w10 = MadNLP.UnreducedKKTVector(kkt_seq)
Random.seed!(321)
randn!(MadNLP.full(v10))
randn!(MadNLP.full(w10))
x_fwd10 = copy(w10)
y_adj10 = copy(v10)
MadNLP.solve_kkt!(kkt_seq, x_fwd10)
MadDiff.adjoint_solve_kkt!(kkt_seq, y_adj10)
lhs10 = dot(v10, x_fwd10)
rhs10 = dot(y_adj10, w10)
diff10 = abs(lhs10 - rhs10)
println("  adjoint identity diff = ", diff10)
@assert diff10 < 1e-5 "Sparse eq adjoint identity failed: $diff10"
println("  PASS")

# ─── Test 11: QP with both active eq and ineq ─────────────────────────
# min 0.5(x1^2 + x2^2) s.t. x1 + x2 = 2, x1 >= 1 (active at x*=[1,1])
println("\n=== Test 11: Dense with active eq + ineq ===")
function _make_qp_mixed_active()
    c = [0.0, 0.0]
    Hrows = [1, 2]; Hcols = [1, 2]; Hvals = [1.0, 1.0]
    Arows = [1, 1, 2]; Acols = [1, 2, 1]; Avals = [1.0, 1.0, 1.0]
    lcon = [2.0, 1.0]   # con1: x1+x2 = 2 (eq), con2: x1 >= 1 (ineq)
    ucon = [2.0, Inf]
    lvar = [-Inf, -Inf]
    uvar = [Inf, Inf]
    x0 = [1.0, 1.0]
    QuadraticModel(c, Hrows, Hcols, Hvals; Arows, Acols, Avals, lcon, ucon, lvar, uvar, x0)
end
nlp_mix = _make_qp_mixed_active()
solver_mix = MadNLP.MadNLPSolver(nlp_mix; print_level=MadNLP.ERROR)
MadNLP.solve!(solver_mix)
println("  x* = ", solver_mix.x.x)

kkt_mix = setup_kkt(MadDiff.DenseCondensedActiveSetKKTSystem, solver_mix, MadNLP.LapackCPUSolver)
println("  n_active=$(kkt_mix.n_active), ns_active=$(kkt_mix.ns_active), n_eq_active=$(kkt_mix.n_eq_active)")
println("  aug_com size: ", size(kkt_mix.aug_com))

# Forward solve: compare with DenseActiveSetKKTSystem
kkt_mix_ref = setup_kkt(MadDiff.DenseActiveSetKKTSystem, solver_mix, MadNLP.LapackCPUSolver)

Random.seed!(999)
rhs_mix = MadNLP.UnreducedKKTVector(kkt_mix)
rhs_mix_ref = MadNLP.UnreducedKKTVector(kkt_mix_ref)
seed_mix = randn(length(MadNLP.full(rhs_mix)))
copyto!(MadNLP.full(rhs_mix), seed_mix)
copyto!(MadNLP.full(rhs_mix_ref), seed_mix)

MadNLP.solve_kkt!(kkt_mix, rhs_mix)
MadNLP.solve_kkt!(kkt_mix_ref, rhs_mix_ref)

diff11 = maximum(abs.(MadNLP.full(rhs_mix) .- MadNLP.full(rhs_mix_ref)))
println("  forward solve diff: ", diff11)
@assert diff11 < 1e-5 "Mixed forward solve mismatch: $diff11"

# Adjoint identity
v11 = MadNLP.UnreducedKKTVector(kkt_mix)
w11 = MadNLP.UnreducedKKTVector(kkt_mix)
Random.seed!(111)
randn!(MadNLP.full(v11))
randn!(MadNLP.full(w11))
x_fwd11 = copy(w11)
y_adj11 = copy(v11)
MadNLP.solve_kkt!(kkt_mix, x_fwd11)
MadDiff.adjoint_solve_kkt!(kkt_mix, y_adj11)
lhs11 = dot(v11, x_fwd11)
rhs11 = dot(y_adj11, w11)
diff11a = abs(lhs11 - rhs11)
println("  adjoint identity diff = ", diff11a)
@assert diff11a < 1e-5 "Mixed adjoint identity failed: $diff11a"
println("  PASS")

println("\n*** All tests passed ***")
