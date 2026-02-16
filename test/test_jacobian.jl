@testset "jacobian_forward" begin
    model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, y)
    @variable(model, p1 in MOI.Parameter(1.0))
    @variable(model, p2 in MOI.Parameter(2.0))
    @constraint(model, x + y == p1 + p2)
    @objective(model, Min, x^2 + y^2)
    optimize!(model)

    sens = MadDiff.MadDiffSolver(unsafe_backend(model).inner.solver)
    jac = MadDiff.jacobian!(sens)

    for j in 1:sens.n_p
        dp = zeros(Float64, sens.n_p)
        dp[j] = 1.0
        col = MadDiff.forward_differentiate!(sens, dp)
        @test isapprox(col.dx, jac.dx[:, j]; atol=1e-8)
        @test isapprox(col.dy, jac.dy[:, j]; atol=1e-8)
        @test isapprox(col.dzl, jac.dzl[:, j]; atol=1e-8)
        @test isapprox(col.dzu, jac.dzu[:, j]; atol=1e-8)
        @test isapprox(col.dobj[], jac.dobj[j]; atol=1e-8)
    end

end

@testset "jacobian_forward MakeParameter" begin
    model = Model(MadDiff.diff_optimizer(
        MadNLP.Optimizer;
        fixed_variable_treatment = MadNLP.MakeParameter,
    ))
    set_silent(model)
    @variable(model, x >= 0.0)
    @variable(model, y <= 1.0)
    @variable(model, z)
    fix(z, 0.5; force = true)
    @variable(model, p1 in MOI.Parameter(1.0))
    @variable(model, p2 in MOI.Parameter(2.0))
    @variable(model, p3 in MOI.Parameter(1.5))
    @constraint(model, x + y + z >= p3)
    @objective(model, Min, (x + p1)^2 + (y - p2)^2 + z^2)
    optimize!(model)

    @test isapprox(value(x), 0.0; atol=1e-8)
    @test isapprox(value(y), 1.0; atol=1e-8)
    @test isapprox(value(x) + value(y) + value(z), value(p3); atol=1e-8)

    sens = MadDiff.MadDiffSolver(unsafe_backend(model).inner.solver)
    jac = MadDiff.jacobian!(sens)

    for j in 1:sens.n_p
        dp = zeros(Float64, sens.n_p)
        dp[j] = 1.0
        col = MadDiff.forward_differentiate!(sens, dp)
        @test isapprox(col.dx, jac.dx[:, j]; atol=1e-8)
        @test isapprox(col.dy, jac.dy[:, j]; atol=1e-8)
        @test isapprox(col.dzl, jac.dzl[:, j]; atol=1e-8)
        @test isapprox(col.dzu, jac.dzu[:, j]; atol=1e-8)
        @test isapprox(col.dobj[], jac.dobj[j]; atol=1e-8)
    end

end

@testset "jacobian_transpose!" begin
    function _test_reverse_rows(sens; atol = 1e-8)
        jac = MadDiff.jacobian_transpose!(sens)
        n_x = NLPModels.get_nvar(sens.solver.nlp)
        n_con = NLPModels.get_ncon(sens.solver.nlp)

        for i in 1:n_x
            dL_dx = zeros(Float64, n_x)
            dL_dx[i] = 1.0
            row = MadDiff.reverse_differentiate!(sens; dL_dx)
            @test isapprox(row.grad_p, jac.dx[:, i]; atol = atol)
        end

        for i in 1:n_con
            dL_dy = zeros(Float64, n_con)
            dL_dy[i] = 1.0
            row = MadDiff.reverse_differentiate!(sens; dL_dy)
            @test isapprox(row.grad_p, jac.dy[:, i]; atol = atol)
        end

        for i in 1:n_x
            dL_dzl = zeros(Float64, n_x)
            dL_dzl[i] = 1.0
            row = MadDiff.reverse_differentiate!(sens; dL_dzl)
            @test isapprox(row.grad_p, jac.dzl[:, i]; atol = atol)
        end

        for i in 1:n_x
            dL_dzu = zeros(Float64, n_x)
            dL_dzu[i] = 1.0
            row = MadDiff.reverse_differentiate!(sens; dL_dzu)
            @test isapprox(row.grad_p, jac.dzu[:, i]; atol = atol)
        end

        row = MadDiff.reverse_differentiate!(sens; dobj = 1.0)
        @test isapprox(row.grad_p, jac.dobj; atol = atol)
    end

    model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, y)
    @variable(model, p1 in MOI.Parameter(1.0))
    @variable(model, p2 in MOI.Parameter(2.0))
    @constraint(model, x + y == p1 + p2)
    @objective(model, Min, x^2 + y^2)
    optimize!(model)

    sens = MadDiff.MadDiffSolver(unsafe_backend(model).inner.solver)
    _test_reverse_rows(sens; atol = 1e-8)

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
    @objective(model_mp, Min, (x + p1)^2 + (y - p2)^2 + z^2)
    optimize!(model_mp)

    sens_mp = MadDiff.MadDiffSolver(unsafe_backend(model_mp).inner.solver)
    _test_reverse_rows(sens_mp; atol = 1e-8)
end

@testset "jacobian_{forward,reverse}_consistency" begin
    function _check_consistency(sens; atol = 1e-8)
        jf = MadDiff.jacobian!(sens)
        jr = MadDiff.jacobian_transpose!(sens)
        @test isapprox(jr.dx, transpose(jf.dx); atol = atol)
        @test isapprox(jr.dy, transpose(jf.dy); atol = atol)
        @test isapprox(jr.dzl, transpose(jf.dzl); atol = atol)
        @test isapprox(jr.dzu, transpose(jf.dzu); atol = atol)
        @test isapprox(jr.dobj, jf.dobj; atol = atol)
    end

    model = Model(MadDiff.diff_optimizer(MadNLP.Optimizer))
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
    @objective(model_mp, Min, (x + p1)^2 + (y - p2)^2 + z^2)
    optimize!(model_mp)
    _check_consistency(MadDiff.MadDiffSolver(unsafe_backend(model_mp).inner.solver); atol = 1e-8)
end