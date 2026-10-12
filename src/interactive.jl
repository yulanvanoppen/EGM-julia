# Back end of the interactive smoothing app (GMGTS/EGM-v2 smoothing_new.mlapp).
#
# `Smoother` is a mutable smoother for one or more trajectories (T × L × N), as
# GMGTS's Smoother.m: per-cell spline coefficients, noise parameters shared by all
# cells of a state, knots changeable per state, and smoothing of selected states.
# With N = 1 it reproduces `smooth` exactly.
#
# `SmoothingSession` holds the app's state (knots per state, selected states, cells
# shown) and implements every app action as a plain function, so the behaviour can be
# tested without a GUI. The window itself lives in the GLMakie extension
# (ext/EGMMakieExt.jl), which provides `smoothing_app`.

"""
    CellSmoothingResult

Smoothing result for several trajectories. As [`SmoothingResult`](@ref), with an extra
cell dimension:
- `delta[k]` is `card × N` (one column of coefficients per cell);
- `variances_sm` is `T × L × N`, `variances_gm` is `Tg × L × N`;
- `smoothed`, `dsmoothed` are `Tg × L × N`, `smoothed_fine`, `dsmoothed_fine`
  are `Tf × L × N`;
- `lincomb[k, i]` and `var_delta[k, i]` belong to state `k` and cell `i`.

The noise parameters `sigma2[k]`, `tau2[k]` are shared by all cells.
"""
struct CellSmoothingResult
    bases::SplineBases
    delta::Vector{Matrix{Float64}}
    sigma2::Vector{Float64}
    tau2::Vector{Float64}
    variances_sm::Array{Float64,3}
    variances_gm::Array{Float64,3}
    smoothed::Array{Float64,3}
    dsmoothed::Array{Float64,3}
    smoothed_fine::Array{Float64,3}
    dsmoothed_fine::Array{Float64,3}
    lincomb::Matrix{Matrix{Float64}}
    var_delta::Matrix{Matrix{Float64}}
    iterations::Int
end

Base.show(io::IO, r::CellSmoothingResult) =
    print(io, "CellSmoothingResult(", length(r.delta), " states, ", size(r.smoothed, 3), " cells)")

_cells(traces::AbstractMatrix, T) = reshape(Matrix{Float64}(traces), T, :, 1)
_cells(traces::AbstractArray{<:Real,3}, T) = Array{Float64,3}(traces)

"""
    Smoother(t, traces; t_gm = t, knots = nothing, autoknots = true, order = 4,
             niter = 20, tol = 1e-3, positive = true, t_fine = nothing)

Mutable B-spline smoother for `traces` (`T × L` or `T × L × N`). Knots follow
[`resolve_knots`](@ref) (time units; automatic placement over all cells by
default). Use [`set_knots!`](@ref), [`optimize!`](@ref) and [`smoothing_result`](@ref).
"""
mutable struct Smoother
    t::Vector{Float64}
    t_gm::Vector{Float64}
    t_fine::Vector{Float64}
    traces::Array{Float64,3}
    order::Int
    knots::Vector{Vector{Float64}}                                    # normalized, including 0 and 1
    bases::SplineBases
    delta::Vector{Matrix{Float64}}
    sigma2::Vector{Float64}
    tau2::Vector{Float64}
    variances::Array{Float64,3}
    variances_gm::Array{Float64,3}
    lincomb::Matrix{Matrix{Float64}}
    niter::Int
    tol::Float64
    positive::Bool
    iterations::Int
end

function Smoother(t, traces; t_gm = t, knots = nothing, autoknots::Bool = true, order::Integer = 4,
                  niter::Integer = 20, tol = 1e-3, positive::Bool = true, t_fine = nothing)
    t = _fvec(t)
    tr = _cells(traces, length(t))
    T, L, N = size(tr)
    kn = if knots !== nothing
        resolve_knots(t, tr[:, :, 1]; knots = knots)
    elseif autoknots
        place_knots(t, tr)
    else
        fill(default_knots(t), L)
    end
    sb = SplineBases(t, _fvec(t_gm), kn; order = order, t_fine = t_fine)
    variances = tr .^ 2 .+ 0.1 .* (sum(tr; dims = 1) ./ T) .^ 2              # mostly multiplicative start
    return Smoother(t, sb.t_gm, sb.t_fine, tr, order, kn, sb, [zeros(b.card, N) for b in sb.splines],
                    zeros(L), zeros(L), variances, zeros(length(sb.t_gm), L, N),
                    [zeros(sb.splines[k].card, T) for k in 1:L, _ in 1:N], max(1, niter), max(1e-12, tol),
                    positive, 0)
end

Base.show(io::IO, s::Smoother) =
    print(io, "Smoother(", size(s.traces, 2), " states, ", size(s.traces, 3), " cells, cards ",
          [b.card for b in s.bases.splines], ")")

"""
    set_knots!(s::Smoother, k, knots)

Replace the knots of state `k` (normalized to [0, 1]; endpoints are added) and reset
its coefficients, as `Smoother.update_knots`. The noise model is kept.
"""
function set_knots!(s::Smoother, k::Integer, knots)
    s.knots[k] = sort(unique(vcat(0.0, clamp.(_fvec(knots), 0.0, 1.0), 1.0)))
    s.bases = SplineBases(s.t, s.t_gm, s.knots; order = s.order, t_fine = s.t_fine)
    T, N = size(s.traces, 1), size(s.traces, 3)
    s.delta[k] = zeros(s.bases.splines[k].card, N)
    for i in 1:N
        s.lincomb[k, i] = zeros(s.bases.splines[k].card, T)
    end
    return s
end

"""
    optimize!(s::Smoother, states = all) -> s

FWLS smoothing of the given states (`Smoother.optimize(states)`): alternate weighted
least-squares coefficients per cell and maximum-likelihood noise parameters per
state (shared by the cells), until the relative change in all coefficients is below
`s.tol`.
"""
function optimize!(s::Smoother, states = 1:size(s.traces, 2))
    T, _, N = size(s.traces)
    sb = s.bases
    for iter in 1:s.niter
        s.iterations = iter
        old = reduce(vcat, [copy(vec(d)) for d in s.delta])            # copy: vec shares memory
        for i in 1:N, k in states
            s.lincomb[k, i], δ = wls_coefficients(sb.B[k], s.traces[:, k, i], s.variances[:, k, i])
            s.delta[k][:, i] = δ
        end
        for k in states
            pred = vec(sb.B[k] * s.delta[k])                                # T·N, cell by cell
            y = vec(s.traces[:, k, :])
            design = hcat(ones(length(pred)), pred .^ 2)
            response = (pred .- y) .^ 2
            start = nnls2(design, response)                                # (eps floor: exact fits, many knots)
            sum(start) == 0 && (start[1] = max(sum(response) / length(response), eps()))
            c = noise_mle(start, pred, y)
            sum(c) == 0 && (c[1] = max(sum(response) / length(response), eps()))
            s.sigma2[k], s.tau2[k] = c
            s.variances[:, k, :] = reshape(design * c, T, N)
            pred_gm = vec(sb.B_gm[k] * s.delta[k])
            s.variances_gm[:, k, :] = reshape(hcat(ones(length(pred_gm)), pred_gm .^ 2) * c, :, N)
        end
        relative_change(old, reduce(vcat, [vec(d) for d in s.delta])) < s.tol && break
    end
    return s
end

"""
    smoothed_fine(s::Smoother, k) -> Matrix

Smoothed values of state `k` on the fine grid `s.t_fine` (`Tf × N`), for plotting.
"""
smoothed_fine(s::Smoother, k::Integer) = s.bases.B_fine[k] * s.delta[k]

"""
    smoothing_result(s::Smoother)

The current smoothing as a [`SmoothingResult`](@ref) (one trajectory) or a
[`CellSmoothingResult`](@ref) (several), as `Smoother.fit`.
"""
function smoothing_result(s::Smoother)
    sb = s.bases
    T, L, N = size(s.traces)
    sm = cat((cat((sb.B_gm[k] * s.delta[k][:, i] for k in 1:L)...; dims = 2) for i in 1:N)...; dims = 3)
    dsm = cat((cat((sb.dB_gm[k] * s.delta[k][:, i] for k in 1:L)...; dims = 2) for i in 1:N)...; dims = 3)
    smf = cat((cat((sb.B_fine[k] * s.delta[k][:, i] for k in 1:L)...; dims = 2) for i in 1:N)...; dims = 3)
    dsmf = cat((cat((sb.dB_fine[k] * s.delta[k][:, i] for k in 1:L)...; dims = 2) for i in 1:N)...; dims = 3)
    s.positive && (sm = max.(sm, 1e-12))
    var_delta = [tryinv(sb.B[k]' * (Diagonal(max.(s.variances[:, k, i], 1e-10)) \ sb.B[k])) for k in 1:L, i in 1:N]
    if N == 1
        return SmoothingResult(sb, [s.delta[k][:, 1] for k in 1:L], copy(s.sigma2), copy(s.tau2),
                               s.variances[:, :, 1], s.variances_gm[:, :, 1], sm[:, :, 1], dsm[:, :, 1],
                               smf[:, :, 1], dsmf[:, :, 1], [s.lincomb[k, 1] for k in 1:L],
                               var_delta[:, 1], s.iterations)
    end
    return CellSmoothingResult(sb, deepcopy(s.delta), copy(s.sigma2), copy(s.tau2), copy(s.variances),
                               copy(s.variances_gm), sm, dsm, smf, dsmf, deepcopy(s.lincomb), var_delta,
                               s.iterations)
end

# --- App session ----------------------------------------------------------------------------

"""
    SmoothingSession(t, traces; state_names = nothing, ncells = 20, live = true, kwargs...)

State of the interactive smoothing app: a [`Smoother`](@ref) (keywords as for
`Smoother`), the selected states (`active`: all, or one), whether smoothing is live,
how many cells are shown, and the display order of the cells. Every app action is a
function on the session:
- [`select_states!`](@ref), [`click_knot!`](@ref);
- [`equidistant_knots!`](@ref), [`automatic_knots!`](@ref);
- [`smooth!`](@ref), [`finish!`](@ref).

The GLMakie window calls these functions; without GLMakie they can be used directly.
"""
mutable struct SmoothingSession
    smoother::Smoother
    state_names::Vector{String}
    active::Vector{Int}
    live::Bool
    ncells::Int
    cells_sorted::Vector{Int}
    stale::Set{Int}                                                   # states whose knots changed since smoothing
end

function SmoothingSession(t, traces; state_names = nothing, ncells::Integer = 20, live::Bool = true, kwargs...)
    s = Smoother(t, traces; kwargs...)
    L, N = size(s.traces, 2), size(s.traces, 3)
    names = state_names === nothing ? ["state $k" for k in 1:L] : String.(collect(state_names))
    length(names) == L || throw(DimensionMismatch("need $L state names"))
    sess = SmoothingSession(s, names, collect(1:L), live, clamp(ncells, 1, N), cell_order(s.traces), Set{Int}())
    optimize!(s)                                                      # initial smoothing with the initial knots
    return sess
end

"""
    cell_order(traces) -> Vector{Int}

Display order of the cells (`smoothing_new.mlapp`). Cells are ranked by their
mean squared deviation from the mean trajectory, scaled per state. The most
typical cell comes first, followed by the others from most to least atypical.
"""
function cell_order(traces::AbstractArray{<:Real,3})
    N = size(traces, 3)
    N == 1 && return [1]
    mean_trace = sum(traces; dims = 3) ./ N
    scales = sum(traces .^ 2; dims = (1, 3)) ./ (size(traces, 1) * N)
    dist = vec(sum((traces .- mean_trace) .^ 2 ./ scales; dims = (1, 2)))
    order = sortperm(dist)
    return vcat(order[1], reverse(order[2:end]))
end

"Maximum number of interior knots per state (`T − order`, so the basis fits the data)."
max_knots(sess::SmoothingSession) = size(sess.smoother.traces, 1) - sess.smoother.order

"State whose knots are displayed and edited: the selected state, or the first if all are selected."
displayed_state(sess::SmoothingSession) = first(sess.active)

"Interior knots (normalized) of state `k`."
interior_knots(sess::SmoothingSession, k) = sess.smoother.knots[k][2:end-1]

"Interior knots of state `k` in time units."
interior_knots_time(sess::SmoothingSession, k) =
    interior_knots(sess, k) .* (sess.smoother.t[end] - sess.smoother.t[1]) .+ sess.smoother.t[1]

"Cells shown in the plots (the first `ncells` in display order)."
shown_cells(sess::SmoothingSession) = sess.cells_sorted[1:sess.ncells]

"""
    select_states!(sess, selection)

Select `:all` states or a single state (index or name), as the State(s) dropdown.
"""
function select_states!(sess::SmoothingSession, selection)
    L = length(sess.state_names)
    if selection === :all || selection == "All"
        sess.active = collect(1:L)
    else
        k = selection isa Integer ? selection : findfirst(==(String(selection)), sess.state_names)
        (k === nothing || !(1 ≤ k ≤ L)) && throw(ArgumentError("unknown state $selection"))
        sess.active = [k]
    end
    return sess
end

# Apply new interior knots to the active states, and smooth if live.
function _apply_knots!(sess::SmoothingSession, knots_by_state)
    for (k, kn) in knots_by_state
        set_knots!(sess.smoother, k, kn)
        push!(sess.stale, k)
    end
    sess.live && smooth!(sess)
    return sess
end

"""
    click_knot!(sess, x; remove = false)

Manual knot selector. Clicking at time `x` adds a knot there. With `remove = true`
(right click), the knot nearest to `x` is removed instead. Knots are edited for
the displayed state; with all states selected, the result is copied to every
state. A knot is only added if the state has fewer than [`max_knots`](@ref).
"""
function click_knot!(sess::SmoothingSession, x::Real; remove::Bool = false)
    s = sess.smoother
    k = displayed_state(sess)
    t1, Δ = s.t[1], s.t[end] - s.t[1]
    kn = interior_knots(sess, k)
    if remove
        if length(kn) > 1
            d = abs.(x .- (kn .* Δ .+ t1))
            kn = kn[d .> minimum(d)]
        else
            kn = Float64[]
        end
    elseif length(kn) < max_knots(sess)
        xn = (x - t1) / Δ
        0 < xn < 1 && (kn = vcat(kn, xn))
    end
    kn = sort(kn)
    return _apply_knots!(sess, [k => kn for k in sess.active])
end

"""
    equidistant_knots!(sess, n)

Replace the knots of the selected state(s) by `n` equidistant interior knots
(#Knots slider).
"""
function equidistant_knots!(sess::SmoothingSession, n::Integer)
    n = clamp(n, 0, max_knots(sess))
    kn = collect(range(0, 1; length = n + 2))[2:end-1]
    return _apply_knots!(sess, [k => kn for k in sess.active])
end

"""
    automatic_knots!(sess)

Replace the knots of the selected state(s) using the knot placement heuristic
(Automatic knot placement button).
"""
function automatic_knots!(sess::SmoothingSession)
    s = sess.smoother
    placed = place_knots(s.t, s.traces)
    return _apply_knots!(sess, [k => placed[k][2:end-1] for k in sess.active])
end

"""
    smooth!(sess)

Smooth the selected states with the current knots (Smooth button, or automatic when
live). States edited earlier but not smoothed yet are included.
"""
function smooth!(sess::SmoothingSession)
    states = sort(union(sess.active, sess.stale))
    optimize!(sess.smoother, states)
    setdiff!(sess.stale, states)
    return sess
end

"""
    finish!(sess) -> (; knots, knots_time, smoothing)

Complete smoothing. All states whose knots changed since they were last smoothed
are smoothed first. (In the MATLAB app, finishing with Live off and unsmoothed
changes left those states with reset coefficients.)

Returns:
- `knots`: the knots per state, normalized to [0, 1] with endpoints;
- `knots_time`: the same in time units, for `egm(…; knots = …)`;
- `smoothing`: a [`SmoothingResult`](@ref) or [`CellSmoothingResult`](@ref).
"""
function finish!(sess::SmoothingSession)
    isempty(sess.stale) || optimize!(sess.smoother, sort(collect(sess.stale)))
    empty!(sess.stale)
    s = sess.smoother
    Δ = s.t[end] - s.t[1]
    return (; knots = deepcopy(s.knots), knots_time = [kn .* Δ .+ s.t[1] for kn in s.knots],
            smoothing = smoothing_result(s))
end

"""
    smoothing_app(t, traces; state_names, knots, autoknots, t_gm, ncells, live,
                  estimation = false, kwargs...) -> (; knots, knots_time, smoothing, action)

Interactive smoothing app (port of the EGM-v2/GMGTS `smoothing_new` app). Requires
GLMakie: run `using GLMakie` first. `traces` is `T × L` (one trajectory) or
`T × L × N` (N cells). The window blocks until it is finished or closed, and then
returns the result of [`finish!`](@ref) plus `action`:
- `:finish`: Finish pressed (or "Finish without estimation");
- `:estimate`: "Finish and run estimation" pressed (only with `estimation = true`);
- `:closed`: the window was closed.

With `estimation = true` (as used by `egm(…; interactive = true)`), the app offers
two buttons: "Finish and run estimation", and "Finish without estimation" for when
only the smoothing is wanted. See the README for the controls.
"""
function smoothing_app end

function __init__()
    if isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
            if exc.f === smoothing_app
                print(io, "\nThe smoothing app needs GLMakie: run `using GLMakie` (install it with `] add GLMakie`).")
            end
        end
    end
end
