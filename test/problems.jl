const PROBLEMS = Dict{String, Tuple{Function, Int, Bool}}()  # (builder, n_params, has_equality)

# qp
PROBLEMS["qp_ineq_only"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x >= p)
        @objective(m, Min, x^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["qp_eq_only"] = (
    function (m)
        @variable(m, x)
        @variable(m, y)
        @variable(m, p in MOI.Parameter(2.0))
        @constraint(m, x + y == p)
        @objective(m, Min, x^2 + y^2)
        return [x, y], p
    end, 1, true,
)

PROBLEMS["qp_mixed"] = (
    function (m)
        @variable(m, x)
        @variable(m, y)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x + y == 2)
        @constraint(m, x >= p)
        @objective(m, Min, x^2 + y^2)
        return [x, y], p
    end, 1, true,
)

PROBLEMS["qp_ineq_fix"] = (
    function (m)
        @variable(m, x)
        @variable(m, y == 1)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x >= p)
        @objective(m, Min, x^2 + y^2)
        return [x, y], p
    end, 1, true,
)

PROBLEMS["qp_lessthan"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x <= p)
        @objective(m, Min, -x)
        return [x], p
    end, 1, false,
)

PROBLEMS["qp_quadobj"] = (
    function (m)
        @variable(m, x)
        @variable(m, y)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x + y >= p)
        @objective(m, Min, x^2 + 2 * x * y + 3 * y^2)
        return [x, y], p
    end, 1, false,
)

PROBLEMS["qp_param_in_obj"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(2.0))
        @objective(m, Min, (x - p)^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["qp_multi_var"] = (
    function (m)
        @variable(m, x[1:3])
        @variable(m, p in MOI.Parameter(3.0))
        @constraint(m, sum(x) == p)
        @objective(m, Min, sum(x[i]^2 for i in 1:3))
        return x, p
    end, 1, true,
)

PROBLEMS["qp_multi_con"] = (
    function (m)
        @variable(m, x)
        @variable(m, y)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x >= p)
        @constraint(m, y >= p)
        @constraint(m, x + y >= 2p)
        @objective(m, Min, x^2 + y^2)
        return [x, y], p
    end, 1, false,
)

PROBLEMS["qp_coupled"] = (
    function (m)
        @variable(m, x)
        @variable(m, y)
        @variable(m, z)
        @variable(m, p in MOI.Parameter(3.0))
        @constraint(m, x + y == p)
        @constraint(m, y + z == 2p)
        @objective(m, Min, x^2 + y^2 + z^2)
        return [x, y, z], p
    end, 1, true,
)

PROBLEMS["qp_literature"] = (
    function (m)
        @variable(m, x[1:2])
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x[1] + x[2] >= p)
        @constraint(m, x[1] >= 0)
        @constraint(m, x[2] >= 0)
        @objective(m, Min, x[1]^2 + x[2]^2 + x[1] * x[2])
        return x, p
    end, 1, false,
)

PROBLEMS["qp_maximization"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x <= p)
        @objective(m, Max, -x^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["qp_var_bounds"] = (
    function (m)
        @variable(m, x >= 0)
        @variable(m, p in MOI.Parameter(0.5))
        @constraint(m, x >= p)
        @objective(m, Min, x^2)
        return [x], p
    end, 1, false,
)

# nlp
PROBLEMS["nlp_quad_con"] = (
    function (m)
        @variable(m, x >= 0.1, start = 1.0)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x^2 >= p)
        @objective(m, Min, x)
        return [x], p
    end, 1, false,
)

PROBLEMS["nlp_trig"] = (
    function (m)
        @variable(m, x >= 0.1, start = 0.8)
        @variable(m, p in MOI.Parameter(0.7))
        @constraint(m, sin(x) >= p)
        @objective(m, Min, x)
        return [x], p
    end, 1, false,
)

PROBLEMS["nlp_exp"] = (
    function (m)
        @variable(m, x >= 0.1, start = 1.0)
        @variable(m, p in MOI.Parameter(2.0))
        @constraint(m, exp(x) >= p)
        @objective(m, Min, x)
        return [x], p
    end, 1, false,
)

PROBLEMS["nlp_softmax"] = (
    function (m)
        @variable(m, x[1:3], start = 0.5)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, sum(exp(x[i]) for i in 1:3) == p * 3)
        @objective(m, Min, sum(x[i]^2 for i in 1:3))
        return x, p
    end, 1, true,
)

PROBLEMS["inactive"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(0.0))
        @constraint(m, x >= p)
        @objective(m, Min, (x - 1)^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["large_coef"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(1000.0))
        @constraint(m, 1000x >= p)
        @objective(m, Min, x^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["small_coef"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(0.001))
        @constraint(m, 0.001x >= p)
        @objective(m, Min, x^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["qp_param_coef"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(2.0))
        @constraint(m, p * x >= 3.0)
        @objective(m, Min, x^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["qp_param_coef_obj"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(2.0))
        @constraint(m, x >= 1.0)
        @objective(m, Min, p * x^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["qp_param_both_sides"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(2.0))
        @constraint(m, p * x >= p + 1)
        @objective(m, Min, x^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["nlp_param_coef"] = (
    function (m)
        @variable(m, x, start = 1.0)
        @variable(m, p in MOI.Parameter(1.5))
        @constraint(m, x * sin(p) == 1)
        @objective(m, Min, x^2)
        return [x], p
    end, 1, true,
)

PROBLEMS["nlp_param_coef_con"] = (
    function (m)
        @variable(m, x >= 0.1, start = 1.0)
        @variable(m, y, start = 0.5)
        @variable(m, p in MOI.Parameter(3.0))
        @constraint(m, y >= p * sin(x))
        @constraint(m, x + y == p)
        @objective(m, Min, x^2 + y^2)
        return [x, y], p
    end, 1, true,
)

PROBLEMS["nlp_param_obj_coef"] = (
    function (m)
        @variable(m, x, start = 0.5)
        @variable(m, y, start = 0.5)
        @variable(m, p in MOI.Parameter(100.0))
        @constraint(m, x + y == 1)
        @objective(m, Min, (1 - x)^2 + p * (y - x^2)^2)
        return [x, y], p
    end, 1, true,
)

# max
PROBLEMS["qp_max_basic"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(1.0))
        @constraint(m, x >= p)
        @objective(m, Max, -x^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["qp_max_lessthan"] = (
    function (m)
        @variable(m, x)
        @variable(m, p in MOI.Parameter(2.0))
        @constraint(m, x <= p)
        @objective(m, Max, x)
        return [x], p
    end, 1, false,
)

PROBLEMS["nlp_max_exp"] = (
    function (m)
        @variable(m, x >= 0.1, start = 1.0)
        @variable(m, p in MOI.Parameter(2.0))
        @constraint(m, exp(x) <= p)
        @objective(m, Max, x)
        return [x], p
    end, 1, false,
)

PROBLEMS["nlp_max_quad"] = (
    function (m)
        @variable(m, x, start = 1.0)
        @variable(m, y, start = 1.0)
        @variable(m, p in MOI.Parameter(2.0))
        @constraint(m, x^2 + y^2 <= p)
        @objective(m, Max, x + y)
        return [x, y], p
    end, 1, false,
)


# diffopt create_jump_model_1
PROBLEMS["diffopt_model_1"] = (
    function (m)
        @variable(m, p in MOI.Parameter(1.5))
        @variable(m, x)
        @constraint(m, x >= p)
        @constraint(m, x >= 2)
        @objective(m, Min, x^2)
        return [x], p
    end, 1, false,
)

# diffopt create_jump_model_2
PROBLEMS["diffopt_model_2"] = (
    function (m)
        @variable(m, p in MOI.Parameter(1.5))
        @variable(m, x >= 2.0)
        @constraint(m, x >= p)
        @objective(m, Min, x^2)
        return [x], p
    end, 1, false,
)

# diffopt create_jump_model_3
PROBLEMS["diffopt_model_3"] = (
    function (m)
        @variable(m, p in MOI.Parameter(-1.5))
        @variable(m, x)
        @constraint(m, x <= p)
        @constraint(m, x <= -2)
        @objective(m, Min, -x)
        return [x], p
    end, 1, false,
)

# diffopt create_jump_model_4
PROBLEMS["diffopt_model_4"] = (
    function (m)
        @variable(m, p in MOI.Parameter(1.5))
        @variable(m, x)
        @constraint(m, x <= p)
        @constraint(m, x <= 2)
        @objective(m, Max, x)
        return [x], p
    end, 1, false,
)

# diffopt create_jump_model_5
PROBLEMS["diffopt_model_5"] = (
    function (m)
        @variable(m, p in MOI.Parameter(1.5))
        @variable(m, x)
        @constraint(m, x >= p)
        @constraint(m, x >= 2)
        @objective(m, Max, -x)
        return [x], p
    end, 1, false,
)

# diffopt create_jump_model_7
PROBLEMS["diffopt_model_7"] = (
    function (m)
        @variable(m, p in MOI.Parameter(1.5))
        @variable(m, x)
        @constraint(m, x * sin(p) == 1)
        @objective(m, Min, 0)
        return [x], p
    end, 1, true,
)

# diffopt create_nonlinear_jump_model_sipopt
PROBLEMS["diffopt_sipopt"] = (
    function (m)
        @variable(m, p1 in MOI.Parameter(4.5))
        @variable(m, x[i = 1:3] >= 0, start = 0.5)
        @constraint(m, 6x[1] + 3x[2] + 2x[3] - p1 == 0)
        @constraint(m, x[1] + x[2] - x[3] - 1 == 0)
        @objective(m, Min, x[1]^2 + x[2]^2 + x[3]^2)
        return x, p1
    end, 1, true,
)

# diffopt create_nonlinear_jump_model_6
PROBLEMS["diffopt_nlp_6"] = (
    function (m)
        @variable(m, p in MOI.Parameter(100.0))
        @variable(m, x[1:2])
        @variable(m, z)
        @variable(m, w)
        @constraint(m, x[2] - 0.0001 * x[1]^2 - 0.2 * z^2 - 0.3 * w^2 >= p + 1)
        @constraint(m, x[1] + 0.001 * x[2]^2 + 0.5 * w^2 + 0.4 * z^2 <= 10 * p + 2)
        @constraint(m, z^2 + w^2 == 13)
        @objective(m, Min, x[2] - x[1] + z - w)
        return [x[1], x[2], z, w], p
    end, 1, true,
)

# >1 param
PROBLEMS["mp_two_params"] = (
    function (m)
        @variable(m, x)
        @variable(m, y)
        @variable(m, p1 in MOI.Parameter(1.0))
        @variable(m, p2 in MOI.Parameter(2.0))
        @constraint(m, x >= p1)
        @constraint(m, y >= p2)
        @objective(m, Min, x^2 + y^2)
        return [x, y], [p1, p2]
    end, 2, false,
)

PROBLEMS["mp_coupled_params"] = (
    function (m)
        @variable(m, x)
        @variable(m, y)
        @variable(m, p1 in MOI.Parameter(1.0))
        @variable(m, p2 in MOI.Parameter(2.0))
        @constraint(m, x + y == p1 + p2)
        @objective(m, Min, x^2 + y^2)
        return [x, y], [p1, p2]
    end, 2, true,
)

PROBLEMS["mp_param_product"] = (
    function (m)
        @variable(m, x)
        @variable(m, p1 in MOI.Parameter(2.0))
        @variable(m, p2 in MOI.Parameter(3.0))
        @constraint(m, p1 * x >= p2)
        @objective(m, Min, x^2)
        return [x], [p1, p2]
    end, 2, false,
)

PROBLEMS["mp_three_params"] = (
    function (m)
        @variable(m, x)
        @variable(m, y)
        @variable(m, z)
        @variable(m, p1 in MOI.Parameter(1.0))
        @variable(m, p2 in MOI.Parameter(2.0))
        @variable(m, p3 in MOI.Parameter(3.0))
        @constraint(m, x + y + z == p1 + p2 + p3)
        @constraint(m, x >= p1)
        @objective(m, Min, x^2 + y^2 + z^2)
        return [x, y, z], [p1, p2, p3]
    end, 3, true,
)

# diffopt create_nonlinear_jump_model
PROBLEMS["diffopt_nlp"] = (
    function (m)
        @variable(m, p in MOI.Parameter(1.0))
        @variable(m, p2 in MOI.Parameter(2.0))
        @variable(m, p3 in MOI.Parameter(100.0))
        @variable(m, x[i = 1:2], start = 0.75)  # lapack struggles without a really good start
        @constraint(m, x[1]^2 <= p)
        @constraint(m, p * (x[1] + x[2])^2 <= p2)
        @objective(m, Min, (1 - x[1])^2 + p3 * (x[2] - x[1]^2)^2)
        return x, [p, p2, p3]
    end, 3, false,
)

# diffopt create_nonlinear_jump_model_sipopt
PROBLEMS["diffopt_sipopt_multi"] = (
    function (m)
        @variable(m, p1 in MOI.Parameter(4.5))
        @variable(m, p2 in MOI.Parameter(1.0))
        @variable(m, x[i = 1:3] >= 0)
        @constraint(m, 6x[1] + 3x[2] + 2x[3] - p1 == 0)
        @constraint(m, p2 * x[1] + x[2] - x[3] - 1 == 0)
        @objective(m, Min, x[1]^2 + x[2]^2 + x[3]^2)
        return x, [p1, p2]
    end, 2, true,
)

# diffopt create_nonlinear_jump_model_1
PROBLEMS["diffopt_nlp_1"] = (
    function (m)
        @variable(m, p[i = 1:3] in MOI.Parameter.([1.0, 2.0, 100.0]))
        @variable(m, x)
        @variable(m, y)
        @constraint(m, y >= p[1] * sin(x))
        @constraint(m, x + y == p[1])
        @constraint(m, p[2] * x >= 0.1)
        @objective(m, Min, (1 - x)^2 + p[3] * (y - x^2)^2)
        return [x, y], p
    end, 3, true,
)

# diffopt create_nonlinear_jump_model_2
PROBLEMS["diffopt_nlp_2"] = (
    function (m)
        @variable(m, p[i = 1:3] in MOI.Parameter.([3.0, 2.0, 10.0]))
        @variable(m, x <= 10)
        @variable(m, y)
        @constraint(m, y >= p[1] * sin(x))
        @constraint(m, x + y == p[1])
        @constraint(m, p[2] * x >= 0.1)
        @objective(m, Min, (1 - x)^2 + p[3] * (y - x^2)^2)
        return [x, y], p
    end, 3, true,
)

PROBLEMS["bound_active_lb"] = (
    function (m)
        @variable(m, x >= 1.0, start=1.0)
        @variable(m, p in MOI.Parameter(0.0))
        @objective(m, Min, (x - p)^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["bound_active_ub"] = (
    function (m)
        @variable(m, x <= 1.0, start=1.0)
        @variable(m, p in MOI.Parameter(2.0))
        @objective(m, Min, (x - p)^2)
        return [x], p
    end, 1, false,
)

PROBLEMS["bound_active_lb_and_ub"] = (
    function (m)
        @variable(m, x >= 1.0, start=1.0)
        @variable(m, y <= 2.0, start=2.0)
        @variable(m, p in MOI.Parameter(0.0))
        @objective(m, Min, (x - p)^2 + (y - 3 - p)^2)
        return [x, y], p
    end, 1, false,
)

PROBLEMS["bound_lb_with_constraint"] = (
    function (m)
        @variable(m, x >= 0.0, start=1.0)
        @variable(m, y >= 0.0, start=1.0)
        @variable(m, p in MOI.Parameter(2.0))
        @constraint(m, x + y >= p)
        @objective(m, Min, x^2 + y^2)
        return [x, y], p
    end, 1, false,
)

PROBLEMS["bound_nlp_exp"] = (
    function (m)
        @variable(m, x >= 0.0, start=0.0)
        @variable(m, p in MOI.Parameter(-1.0))
        @constraint(m, x >= p)
        @objective(m, Min, exp(x))
        return [x], p
    end, 1, false,
)

PROBLEMS["bound_two_params"] = (
    function (m)
        @variable(m, x >= 0.0, start=0.0)
        @variable(m, y >= 0.0, start=0.0)
        @variable(m, p1 in MOI.Parameter(1.0))
        @variable(m, p2 in MOI.Parameter(1.0))
        @constraint(m, x >= p1)
        @constraint(m, y >= p2)
        @objective(m, Min, x^2 + y^2)
        return [x, y], [p1, p2]
    end, 2, false,
)
