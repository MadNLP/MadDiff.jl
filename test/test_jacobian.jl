using NLPModels, LinearAlgebra, ExaModels

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

@testset "ExaModels JVP/VJP vs FiniteDiff" begin
    p0 = [1.0, 3.0]
    h  = 1e-5
    atol = sqrt(h)

    function _make_exa(p_vals)
        c = ExaCore()
        p = ExaModels.parameter(c, p_vals)
        x = ExaModels.variable(c, 2)
        ExaModels.objective(c, x[1]^2 + x[2]^2 + p[1] * x[1])
        ExaModels.constraint(c, x[1] + x[2] - p[2])
        return ExaModel(c)
    end

    function _solve_exa(p_vals)
        m = _make_exa(p_vals)
        solver = MadNLP.MadNLPSolver(m; print_level = MadNLP.ERROR)
        MadNLP.solve!(solver)
        return solver
    end

    solver0 = _solve_exa(p0)
    sens = MadDiffSolver(solver0)
    x0   = Vector(MadNLP.variable(solver0.x))
    y0   = Vector(solver0.y)
    n_p  = sens.n_p

    for j in 1:n_p
        Δp    = zeros(n_p); Δp[j] = 1.0
        jvp   = MadDiff.jacobian_vector_product!(sens, Δp)
        s_plus = _solve_exa(p0 .+ h .* Δp)
        @test isapprox(jvp.dx, (Vector(MadNLP.variable(s_plus.x)) .- x0) ./ h; atol)
        @test isapprox(jvp.dy, (Vector(s_plus.y)                  .- y0) ./ h; atol)
    end

    rng   = MersenneTwister(42)
    dL_dx = randn(rng, length(x0))
    dL_dy = randn(rng, length(y0))
    vjp   = MadDiff.vector_jacobian_product!(sens; dL_dx, dL_dy)
    for j in 1:n_p
        Δp     = zeros(n_p); Δp[j] = 1.0
        s_plus = _solve_exa(p0 .+ h .* Δp)
        dx_fd  = (Vector(MadNLP.variable(s_plus.x)) .- x0) ./ h
        dy_fd  = (Vector(s_plus.y)                  .- y0) ./ h
        @test isapprox(vjp.grad_p[j], dot(dL_dx, dx_fd) + dot(dL_dy, dy_fd); atol)
    end
end
