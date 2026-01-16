using Test
using MadDiff
using MadNLP
using JuMP
using LinearAlgebra
using CUDA
using KernelAbstractions
using ExaModels
using MadNLPGPU

const HAS_CUDA = CUDA.functional()

if !HAS_CUDA
    @info "Skipping CUDA tests..."
else

    @info "Running CUDA tests..."
    CUDA.versioninfo()

    function build_test_model()
        model = Model()
        @variable(model, x, start = 1.0)
        @variable(model, y, start = 1.0)
        @constraint(model, x + y == 2)
        @objective(model, Min, x^2 + y^2)
        return model
    end

    @testset "CUDA API" begin
        @testset "Forward" begin
            jm = build_test_model()
            backend = CUDABackend()
            em = ExaModels.ExaModel(jm; backend = backend)

            solver = MadNLP.MadNLPSolver(em; print_level = MadNLP.ERROR)
            MadNLP.solve!(solver)

            sens = MadDiff.MadDiffSolver(solver)

            # ∇xpL = zeros (no parameter-primal cross terms)
            # ∇pg = -1 (perturbation of constraint RHS)
            Dxp_L = CuArray([0.0, 0.0])
            Dp_g = CuArray([-1.0])
            result = MadDiff.forward_differentiate!(sens; Dxp_L, Dp_g)

            # Verify results are CuArrays
            @test result.dx isa CuArray
            @test result.dλ isa CuArray
            @test result.dzl isa CuArray
            @test result.dzu isa CuArray

            dx = Vector(result.dx)
            @test isapprox(dx, [0.5, 0.5]; atol = 1.0e-4)
        end

        @testset "Reverse" begin
            jm = build_test_model()
            backend = CUDABackend()
            em = ExaModels.ExaModel(jm; backend = backend)

            solver = MadNLP.MadNLPSolver(em; print_level = MadNLP.ERROR)
            MadNLP.solve!(solver)

            sens = MadDiff.MadDiffSolver(solver)

            dL_dx = CuArray([1.0, 1.0])

            result = MadDiff.reverse_differentiate!(sens; dL_dx)

            # Verify results are CuArrays
            @test result.adj_x isa CuArray
            @test result.adj_λ isa CuArray
            @test result.adj_zl isa CuArray
            @test result.adj_zu isa CuArray

            adj_x = Vector(result.adj_x)
            adj_λ = Vector(result.adj_λ)  # equality constraint dual adjoint

            @test isapprox(adj_x, [0.0, 0.0]; atol = 1.0e-4)
            @test isapprox(adj_λ, [1.0]; atol = 1.0e-4)
        end

        @testset "Forward-Reverse" begin
            jm = build_test_model()
            backend = CUDABackend()
            em = ExaModels.ExaModel(jm; backend = backend)

            solver = MadNLP.MadNLPSolver(em; print_level = MadNLP.ERROR)
            MadNLP.solve!(solver)

            sens = MadDiff.MadDiffSolver(solver)

            Dxp_L = CuArray([0.0, 0.0])
            Dp_g = CuArray([-1.0])

            fwd = MadDiff.forward_differentiate!(sens; Dxp_L, Dp_g)
            dx_dp = Vector(fwd.dx)

            dL_dx = CuArray([1.0, 1.0])
            rev = MadDiff.reverse_differentiate!(sens; dL_dx)

            dloss_dp_fwd = dot([1.0, 1.0], dx_dp)

            adj_x = Vector(rev.adj_x)
            adj_λ = Vector(rev.adj_λ)  # equality constraint dual adjoint
            # dL/dp = -(adj_x · Dxp_L + adj_λ · Dp_g) relates forward and reverse sensitivities
            dloss_dp_rev = -(dot(adj_x, [0.0, 0.0]) + dot(adj_λ, [-1.0]))

            @test isapprox(dloss_dp_fwd, dloss_dp_rev; atol = 1.0e-6)
        end
    end

end  # if HAS_CUDA
