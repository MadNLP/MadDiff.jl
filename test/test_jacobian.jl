using NLPModels

@testset "jac/jact/jvp/vjp" begin
    function _check_consistency(sens; atol = 1e-8)
        jf = MadDiff.jacobian!(sens)
        jr = MadDiff.jacobian_transpose!(sens)
        @test isapprox(jr.dx, transpose(jf.dx); atol = atol)
        @test isapprox(jr.dy, transpose(jf.dy); atol = atol)
        @test isapprox(jr.dzl, transpose(jf.dzl); atol = atol)
        @test isapprox(jr.dzu, transpose(jf.dzu); atol = atol)
        @test isapprox(jr.dobj, jf.dobj; atol = atol)

        n_x = NLPModels.get_nvar(sens.solver.nlp)
        n_con = NLPModels.get_ncon(sens.solver.nlp)
        for j in 1:sens.n_p
            dp = zeros(Float64, sens.n_p)
            dp[j] = 1.0
            col = MadDiff.jacobian_vector_product!(sens, dp)

            for i in 1:n_x
                dL_dx = zeros(Float64, n_x)
                dL_dx[i] = 1.0
                row = MadDiff.vector_jacobian_product!(sens; dL_dx)
                @test isapprox(col.dx[i], row.grad_p[j]; atol = atol)
            end

            for i in 1:n_con
                dL_dy = zeros(Float64, n_con)
                dL_dy[i] = 1.0
                row = MadDiff.vector_jacobian_product!(sens; dL_dy)
                @test isapprox(col.dy[i], row.grad_p[j]; atol = atol)
            end

            for i in 1:n_x
                dL_dzl = zeros(Float64, n_x)
                dL_dzl[i] = 1.0
                row = MadDiff.vector_jacobian_product!(sens; dL_dzl)
                @test isapprox(col.dzl[i], row.grad_p[j]; atol = atol)
            end

            for i in 1:n_x
                dL_dzu = zeros(Float64, n_x)
                dL_dzu[i] = 1.0
                row = MadDiff.vector_jacobian_product!(sens; dL_dzu)
                @test isapprox(col.dzu[i], row.grad_p[j]; atol = atol)
            end

            row = MadDiff.vector_jacobian_product!(sens; dobj = 1.0)
            @test isapprox(col.dobj[], row.grad_p[j]; atol = atol)
        end
    end

    model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer; linear_solver = MadNLP.MumpsSolver))
    set_silent(model)
    @variable(model, x)
    @variable(model, y)
    @variable(model, p1 in MOI.Parameter(1.0))
    @variable(model, p2 in MOI.Parameter(2.0))
    @constraint(model, x + y == p1 + p2)
    @objective(model, Min, x^2 + y^2)
    optimize!(model)
    _check_consistency(MadDiff.MadDiffSolver(unsafe_backend(model).inner.solver); atol = 1e-8)

    model_mp = Model(MadDiff.diff_optimizer(
        MadNLP.Optimizer;
        fixed_variable_treatment = MadNLP.MakeParameter,
    ))
    set_silent(model_mp)
    @variable(model_mp, x >= 0.0)
    @variable(model_mp, y <= 1.0)
    @variable(model_mp, z)
    fix(z, 0.5; force = true)
    @variable(model_mp, p1 in MOI.Parameter(1.0))
    @variable(model_mp, p2 in MOI.Parameter(2.0))
    @variable(model_mp, p3 in MOI.Parameter(1.5))
    @constraint(model_mp, x + y + z >= p3)
    @objective(model_mp, Min, (x + p1)^2 + (y - p2)^4 + z^2)
    optimize!(model_mp)
    _check_consistency(MadDiff.MadDiffSolver(unsafe_backend(model_mp).inner.solver, config = MadDiffConfig(kkt_system = MadNLP.SparseUnreducedKKTSystem)); atol = 1e-8)

    model_rect = Model(MadDiff.diff_optimizer(MadNLP.Optimizer; linear_solver = MadNLP.MumpsSolver))
    set_silent(model_rect)
    @variable(model_rect, x >= 0.0)
    @variable(model_rect, y)
    @variable(model_rect, z)
    @variable(model_rect, p1 in MOI.Parameter(1.0))
    @variable(model_rect, p2 in MOI.Parameter(0.5))
    @constraint(model_rect, x + y + z == p1)
    @constraint(model_rect, y - z >= p2)
    @objective(model_rect, Min, (x - 1)^2 + (y + p2)^2 + (z - p1)^2)
    optimize!(model_rect)
    _check_consistency(MadDiff.MadDiffSolver(unsafe_backend(model_rect).inner.solver); atol = 1e-8)
end
