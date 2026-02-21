using NLPModels, LinearAlgebra, ExaModels

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
