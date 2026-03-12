function identify_active_set(solver::MadNLPSolver{T}; sigma = T(0.75)) where T
    cb = solver.cb

    # proximity measure ψ (Oberlin & Wright 2005, eq 11)
    f_full, zl_full, zu_full = full(solver.f), full(solver.zl), full(solver.zu)
    psi = sum(abs, f_full .- zl_full .+ zu_full .+ solver.jacl; init=zero(T)) +
          sum(abs, solver.c; init=zero(T)) +
          sum(abs.(min.(solver.zl_r, solver.x_lr .- solver.xl_r)); init=zero(T)) +
          sum(abs.(min.(solver.zu_r, solver.xu_r .- solver.x_ur)); init=zero(T))
    threshold = psi^sigma

    xs, xls, xus = solver.x.s, solver.xl.s, solver.xu.s
    slack_gap = min.(xs .- xls, xus .- xs)
    ineq_active_mask = slack_gap .<= threshold
    active_ineq = cb.ind_ineq[ineq_active_mask]

    result = vcat(cb.ind_eq, active_ineq)
    AK.sort!(result)
    return result
end

function _active_slack_mapping(active_cons, ind_ineq, m::Int)
    ineq_to_slack = fill!(similar(ind_ineq, m), 0)
    seq = similar(ind_ineq, length(ind_ineq))
    seq .= 1:length(ind_ineq)
    ineq_to_slack[ind_ineq] .= seq

    slack_positions = ineq_to_slack[active_cons]
    mask = slack_positions .> 0
    active_slack_map = slack_positions[mask]
    ind_ineq_active = findall(mask)

    return active_slack_map, ind_ineq_active, length(active_slack_map)
end
