using Test, LinearAlgebra, SparseArrays
using MadDiff, MadNLP, MadIPM, NLPModels, QuadraticModels

const BatchMadDiffSolver = Base.get_extension(MadDiff, :MadIPMExt).BatchMadDiffSolver

@testset "Batch JVP/VJP" begin
    T = Float64
    # Simple parametric QP:
    #   min  ½x₁² + ½x₂² + (F*θ)ᵀx
    #   s.t. x₁ + x₂ ≥ 1 + B*θ
    #        0 ≤ x₁, x₂ ≤ 10
    nvar = 2; ncon = 1; nparam = 1; nbatch = 3

    c = zeros(T, nvar)
    H = sparse([1, 2], [1, 2], [1.0, 1.0], nvar, nvar)
    A = sparse([1, 1], [1, 2], [1.0, 1.0], ncon, nvar)
    F = [1.0; 0.0;;]   # nvar × nparam
    B = [1.0;;]         # ncon × nparam

    # Different parameter values per instance
    θ_vals = [0.0, 0.1, 0.2]

    # Build sequential models
    lpqps = [LinearParametricQuadraticModel(
        c, H, A, F, B, [θ];
        lvar = [0.0, 0.0], uvar = [10.0, 10.0],
        lcon = [1.0], ucon = [Inf],
    ) for θ in θ_vals]

    # Build batch model from sequential models
    bqp = BatchLinearParametricQuadraticModel(lpqps)

    # Solve batch
    batch_solver = MadIPM.UniformBatchMPCSolver(bqp; tol=1e-8, max_iter=200)
    MadIPM.solve!(batch_solver)

    # Build batch diff solver
    bsens = BatchMadDiffSolver(batch_solver)

    # Parameter perturbation
    Δp = ones(T, nparam, nbatch)

    @testset "Batch JVP" begin
        result = MadDiff.jacobian_vector_product!(bsens, Δp)

        for j in 1:nbatch
            seq_solver = MadIPM.MPCSolver(lpqps[j]; print_level=MadNLP.ERROR, tol=1e-8, max_iter=200)
            MadIPM.solve!(seq_solver)
            seq_sens = MadDiffSolver(seq_solver)
            seq_result = MadDiff.jacobian_vector_product!(seq_sens, Δp[:, j])

            @test result.dx[:, j] ≈ seq_result.dx atol=1e-6
            @test result.dy[:, j] ≈ seq_result.dy atol=1e-6
            @test result.dobj[j]  ≈ seq_result.dobj[] atol=1e-6
        end
    end

    @testset "Batch VJP" begin
        dobj = ones(T, nbatch)
        result = MadDiff.vector_jacobian_product!(bsens; dobj = dobj)

        for j in 1:nbatch
            seq_solver = MadIPM.MPCSolver(lpqps[j]; print_level=MadNLP.ERROR, tol=1e-8, max_iter=200)
            MadIPM.solve!(seq_solver)
            seq_sens = MadDiffSolver(seq_solver)
            seq_result = MadDiff.vector_jacobian_product!(seq_sens; dobj = dobj[j])

            @test result.grad_p[:, j] ≈ seq_result.grad_p atol=1e-6
        end
    end

    @testset "Batch VJP with dL_dx" begin
        nvar_nlp = bqp.meta.nvar
        dL_dx = ones(T, nvar_nlp, nbatch)
        result = MadDiff.vector_jacobian_product!(bsens; dL_dx = dL_dx)

        for j in 1:nbatch
            seq_solver = MadIPM.MPCSolver(lpqps[j]; print_level=MadNLP.ERROR, tol=1e-8, max_iter=200)
            MadIPM.solve!(seq_solver)
            seq_sens = MadDiffSolver(seq_solver)
            seq_result = MadDiff.vector_jacobian_product!(seq_sens; dL_dx = dL_dx[:, j])

            @test result.grad_p[:, j] ≈ seq_result.grad_p atol=1e-6
        end
    end
end
