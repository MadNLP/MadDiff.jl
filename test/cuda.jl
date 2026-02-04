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

    @testset "CUDA API" begin
        @testset "Forward" begin
            model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
            set_silent(model)
            @variable(model, x, start = 1.0)
            @variable(model, y, start = 1.0)
            @variable(model, p in MOI.Parameter(2.0))
            @constraint(model, x + y == p)
            @objective(model, Min, x^2 + y^2)
            optimize!(model)

            DiffOpt.empty_input_sensitivities!(model)
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), MOI.Parameter(1.0))
            DiffOpt.forward_differentiate!(model)
            dx = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x)
            dy_val = MOI.get(model, DiffOpt.ForwardVariablePrimal(), y)

            @test isapprox(dx, 0.5; atol = 1.0e-4)
            @test isapprox(dy_val, 0.5; atol = 1.0e-4)
        end

        @testset "Reverse" begin
            model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
            set_silent(model)
            @variable(model, x, start = 1.0)
            @variable(model, y, start = 1.0)
            @variable(model, p in MOI.Parameter(2.0))
            @constraint(model, x + y == p)
            @objective(model, Min, x^2 + y^2)
            optimize!(model)

            DiffOpt.empty_input_sensitivities!(model)
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), y, 1.0)
            DiffOpt.reverse_differentiate!(model)
            dp = MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)).value

            @test isapprox(dp, 1.0; atol = 1.0e-4)
        end

        @testset "Forward-Reverse" begin
            model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
            set_silent(model)
            @variable(model, x, start = 1.0)
            @variable(model, y, start = 1.0)
            @variable(model, p in MOI.Parameter(2.0))
            @constraint(model, x + y == p)
            @objective(model, Min, x^2 + y^2)
            optimize!(model)

            DiffOpt.empty_input_sensitivities!(model)
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), MOI.Parameter(1.0))
            DiffOpt.forward_differentiate!(model)
            dx_fwd = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x)
            dy_fwd = MOI.get(model, DiffOpt.ForwardVariablePrimal(), y)

            DiffOpt.empty_input_sensitivities!(model)
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), y, 1.0)
            DiffOpt.reverse_differentiate!(model)
            dp_rev = MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)).value

            # Forward: dL/dp = dL/dx * dx/dp + dL/dy * dy/dp
            dloss_dp_fwd = 1.0 * dx_fwd + 1.0 * dy_fwd

            @test isapprox(dloss_dp_fwd, dp_rev; atol = 1.0e-6)
        end
    end

end  # if HAS_CUDA
