using Test, LinearAlgebra, SparseArrays
using MadDiff, MadNLP, MadIPM, NLPModels, QuadraticModels
using CUDA

if !CUDA.functional()
    @info "CUDA not functional, skipping batch CUDA diff tests"
else

include("test_batch_diff.jl")  # reuse TestBatchParametricQP and helpers

# Build a CuArray-backed batch model from the CPU one
function gpu_batch_parametric_qp(c, H, A, F, B, θ_batch; lvar, uvar, lcon, ucon)
    T = eltype(c)
    nvar = length(c)
    ncon = size(A, 1)
    nparam = size(F, 2)
    nbatch = size(θ_batch, 2)
    MT = CuMatrix{T}

    meta = NLPModels.BatchNLPModelMeta{T, MT}(
        nbatch, nvar;
        ncon = ncon,
        lvar = CuMatrix(lvar), uvar = CuMatrix(uvar),
        lcon = CuMatrix(lcon), ucon = CuMatrix(ucon),
        nnzj = nnz(A),
        nnzh = nnz(H),
        islp = (nnz(H) == 0),
        nparam = nparam,
        nnzgp = nparam,
        nnzjp = length(B),
        nnzhp = length(F),
        grad_param_available = true,
        jac_param_available = ncon > 0,
        hess_param_available = true,
        jpprod_available = ncon > 0,
        jptprod_available = ncon > 0,
        hpprod_available = true,
        hptprod_available = true,
    )

    c_eff = CuMatrix(repeat(c, 1, nbatch) .+ F * θ_batch)
    Bθ = CuMatrix(B * θ_batch)
    _HX = CUDA.zeros(T, nvar, nbatch)

    return TestBatchParametricQP{T, MT}(
        meta, CuSparseMatrixCSC(H), CuSparseMatrixCSC(A),
        F, B,  # F, B stay on CPU (used via mul! with CuMatrix)
        CuMatrix(θ_batch), copy(c), c_eff, Bθ, _HX,
    )
end

@testset "Batch JVP/VJP CUDA" begin
    T = Float64
    nvar = 2; ncon = 1; nparam = 1; nbatch = 3

    c = zeros(T, nvar)
    H = sparse([1, 2], [1, 2], [1.0, 1.0], nvar, nvar)
    A = sparse([1, 1], [1, 2], [1.0, 1.0], ncon, nvar)
    F = [1.0; 0.0;;]
    B = [1.0;;]
    θ_batch = [0.0 0.5 1.0]

    lcon = fill(1.0, ncon, nbatch) .+ B * θ_batch
    ucon = fill(T(Inf), ncon, nbatch)
    lvar = fill(0.0, nvar, nbatch)
    uvar = fill(10.0, nvar, nbatch)

    # CPU reference
    bqp_cpu = TestBatchParametricQP(c, H, A, F, B, θ_batch;
                                     lvar = lvar, uvar = uvar, lcon = lcon, ucon = ucon)
    batch_solver_cpu = MadIPM.UniformBatchMPCSolver(bqp_cpu)
    MadIPM.solve!(batch_solver_cpu)
    bsens_cpu = MadDiff.MadIPMExt.BatchMadDiffSolver(batch_solver_cpu)

    Δp_cpu = ones(T, nparam, nbatch)
    jvp_cpu = MadDiff.jacobian_vector_product!(bsens_cpu, Δp_cpu)

    # GPU solve
    bqp_gpu = gpu_batch_parametric_qp(c, H, A, F, B, θ_batch;
                                       lvar = lvar, uvar = uvar, lcon = lcon, ucon = ucon)
    batch_solver_gpu = MadIPM.UniformBatchMPCSolver(bqp_gpu)
    MadIPM.solve!(batch_solver_gpu)
    bsens_gpu = MadDiff.MadIPMExt.BatchMadDiffSolver(batch_solver_gpu)

    @testset "JVP" begin
        Δp_gpu = CuMatrix(Δp_cpu)
        jvp_gpu = MadDiff.jacobian_vector_product!(bsens_gpu, Δp_gpu)

        @test Array(jvp_gpu.dx) ≈ jvp_cpu.dx atol=1e-5
        @test Array(jvp_gpu.dy) ≈ jvp_cpu.dy atol=1e-5
        @test Array(jvp_gpu.dobj) ≈ jvp_cpu.dobj atol=1e-5
    end

    @testset "VJP with dobj" begin
        dobj_cpu = ones(T, nbatch)
        vjp_cpu = MadDiff.vector_jacobian_product!(bsens_cpu; dobj = dobj_cpu)

        dobj_gpu = CuVector(dobj_cpu)
        vjp_gpu = MadDiff.vector_jacobian_product!(bsens_gpu; dobj = dobj_gpu)

        @test Array(vjp_gpu.grad_p) ≈ vjp_cpu.grad_p atol=1e-5
    end

    @testset "VJP with dL_dx" begin
        nvar_nlp = bqp_cpu.meta.nvar
        dL_dx_cpu = ones(T, nvar_nlp, nbatch)
        vjp_cpu = MadDiff.vector_jacobian_product!(bsens_cpu; dL_dx = dL_dx_cpu)

        dL_dx_gpu = CuMatrix(dL_dx_cpu)
        vjp_gpu = MadDiff.vector_jacobian_product!(bsens_gpu; dL_dx = dL_dx_gpu)

        @test Array(vjp_gpu.grad_p) ≈ vjp_cpu.grad_p atol=1e-5
    end
end

end  # if CUDA.functional()
