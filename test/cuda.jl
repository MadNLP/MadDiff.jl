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

            # d²L/dxdp = zeros (no parameter-primal cross terms)
            # dg/dp = -1 (perturbation of constraint RHS)
            d2L_dxdp = CuArray([0.0, 0.0])
            dg_dp = CuArray([-1.0])
            result = MadDiff.forward_differentiate!(sens; d2L_dxdp, dg_dp)

            # Verify results are CuArrays
            @test result.dx isa CuArray
            @test result.dy isa CuArray
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
            @test result.dx isa CuArray
            @test result.dy isa CuArray
            @test result.dzl isa CuArray
            @test result.dzu isa CuArray

            dx = Vector(result.dx)
            dy = Vector(result.dy)  # equality constraint dual adjoint

            @test isapprox(dx, [0.0, 0.0]; atol = 1.0e-4)
            @test isapprox(dy, [1.0]; atol = 1.0e-4)
        end

        @testset "Forward-Reverse" begin
            jm = build_test_model()
            backend = CUDABackend()
            em = ExaModels.ExaModel(jm; backend = backend)

            solver = MadNLP.MadNLPSolver(em; print_level = MadNLP.ERROR)
            MadNLP.solve!(solver)

            sens = MadDiff.MadDiffSolver(solver)

            d2L_dxdp = CuArray([0.0, 0.0])
            dg_dp = CuArray([-1.0])

            fwd = MadDiff.forward_differentiate!(sens; d2L_dxdp, dg_dp)
            dx_dp = Vector(fwd.dx)

            dL_dx = CuArray([1.0, 1.0])
            rev = MadDiff.reverse_differentiate!(sens; dL_dx)

            dloss_dp_fwd = dot([1.0, 1.0], dx_dp)

            dx = Vector(rev.dx)
            dy = Vector(rev.dy)  # equality constraint dual adjoint
            # dL/dp = -(dx · d2L_dxdp + dy · dg_dp) relates forward and reverse sensitivities
            dloss_dp_rev = -(dot(dx, [0.0, 0.0]) + dot(dy, [-1.0]))

            @test isapprox(dloss_dp_fwd, dloss_dp_rev; atol = 1.0e-6)
        end
    end

end  # if HAS_CUDA
