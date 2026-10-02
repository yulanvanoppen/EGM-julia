# Phase 2: B-splines and knot placement.
#
# Replaces BSpline.m, the knot handling in the EGM constructor, and
# Smoother.place_knots / Smoother.update_knots. The basis recursion is a literal
# port of BSpline.recursion(_diff), including its conventions at x = 1 and its
# max(eps, ·) guards on zero-length knot intervals, so the basis matrices match
# MATLAB to machine precision.
#
# Orientation: unlike BSpline.m, `basis` and `basis_diff` return design matrices
# (length(x) × card), i.e. the transpose of the MATLAB output.

"""
    matlab_linspace(a, b, n)

`n` equally spaced points from `a` to `b`, computed as MATLAB's `linspace`
(`a + k(b − a)/(n − 1)` with exact endpoints). Used where tie-breaking or
bit-level agreement with MATLAB matters.
"""
function matlab_linspace(a, b, n::Integer)
    n ≥ 2 || return [float(b)]
    y = [a + (k * (b - a)) / (n - 1) for k in 0:n-1]
    y[1], y[end] = a, b
    return float.(y)
end

"""
    BSpline(order, knots)
    BSpline(order, n::Integer)

B-spline basis of the given order (4 = cubic) on `[0, 1]`, with clamped
boundary knots. `knots` are the knot locations in `[0, 1]`; endpoints 0 and 1 are
removed if present. `BSpline(order, n)` uses `n − 1` equidistant interior knots
(`n` segments). The basis cardinality is `order + number of interior knots`.
"""
struct BSpline
    ord::Int
    knots::Vector{Float64}                                            # interior knots
    card::Int
end

BSpline(ord::Integer, n::Integer) = BSpline(ord, collect((1:n-1) ./ n), ord + n - 1)

function BSpline(ord::Integer, knots::AbstractVector{<:Real})
    interior = Float64[k for k in knots if k != 0 && k != 1]
    return BSpline(ord, interior, ord + length(interior))
end

# Knot sequence index → location: 0 for k ≤ 0, interior knots for 1 ≤ k ≤ nseg − 1,
# and 1 for k ≥ nseg (as the anonymous function t(k) in BSpline.recursion).
function _t(b::BSpline, k::Integer)
    nseg = length(b.knots) + 1
    nseg == 1 && return k > 0 ? 1.0 : 0.0
    return (0 < k ≤ nseg - 1 ? b.knots[clamp(k, 1, nseg - 1)] : 0.0) + (k > nseg - 1 ? 1.0 : 0.0)
end

function _rec(b::BSpline, x, j, q)
    if q == 1
        tj, tj1 = _t(b, j), _t(b, j + 1)
        return ((tj ≤ x && x < tj1) || (x == 1 && tj1 == 1)) ? 1.0 : 0.0
    end
    return (x - _t(b, j)) / max(eps(), _t(b, j + q - 1) - _t(b, j)) * _rec(b, x, j, q - 1) +
           (_t(b, j + q) - x) / max(eps(), _t(b, j + q) - _t(b, j + 1)) * _rec(b, x, j + 1, q - 1)
end

function _rec_diff(b::BSpline, x, j, q)
    nseg = length(b.knots) + 1
    q == 1 && return 0.0
    if q == 2
        tj, tj1, tj2 = _t(b, j), _t(b, j + 1), _t(b, j + 2)
        i1 = ((tj ≤ x && x < tj1) || (x == 1 && j == nseg - 1)) ? 1.0 : 0.0
        i2 = ((tj1 ≤ x && x < tj2) || (x == 1 && j == nseg - 2)) ? 1.0 : 0.0
        return i1 / max(eps(), tj1 - tj) - i2 / max(eps(), tj2 - tj1)
    end
    return ((x - _t(b, j)) * _rec_diff(b, x, j, q - 1) + _rec(b, x, j, q - 1)) /
               max(eps(), _t(b, j + q - 1) - _t(b, j)) +
           ((_t(b, j + q) - x) * _rec_diff(b, x, j + 1, q - 1) - _rec(b, x, j + 1, q - 1)) /
               max(eps(), _t(b, j + q) - _t(b, j + 1))
end

"""
    basis(b::BSpline, x) -> Matrix

Basis functions evaluated at the points `x ⊂ [0, 1]`: a `length(x) × b.card`
design matrix.
"""
basis(b::BSpline, x::AbstractVector) = [_rec(b, xi, s - b.ord, b.ord) for xi in x, s in 1:b.card]

"""
    basis_diff(b::BSpline, x) -> Matrix

First derivatives (with respect to the scaled coordinate `x`) of the basis
functions: a `length(x) × b.card` matrix.
"""
basis_diff(b::BSpline, x::AbstractVector) = [_rec_diff(b, xi, s - b.ord, b.ord) for xi in x, s in 1:b.card]

# --- Knots ------------------------------------------------------------------------

"""
    normalize_knots(knots, t)

Clean up user-supplied knots (in time units) as the EGM constructor does: clamp
to `[t[1], t[end]]`, add both endpoints, sort, remove duplicates, and map to
`[0, 1]`.
"""
function normalize_knots(knots, t)
    t1, tT = first(t), last(t)
    truncated = clamp.(collect(float.(knots)), t1, tT)
    arranged = sort(unique(vcat(t1, truncated, tT)))
    return (arranged .- t1) ./ (tT - t1)
end

"""
    default_knots(t)

Equidistant knots used by EGM when automatic placement is switched off and no
knots are given: `round((T − 1)/2) + 1` points from `t[1]` to `t[end]`,
normalized to `[0, 1]`.
"""
function default_knots(t)
    n = round(Int, (length(t) - 1) / 2, RoundNearestTiesAway) + 1
    return normalize_knots(matlab_linspace(first(t), last(t), n), t)
end

_std(x) = sqrt(sum(abs2, x .- sum(x) / length(x)) / (length(x) - 1))  # sample standard deviation

"""
    place_knots(t, traces) -> Vector{Vector{Float64}}

Knot placement heuristic of `Smoother.place_knots` (single trajectory). The
steps are:
1. start from the data points closest to three equidistant interior locations;
2. add estimated peaks and troughs (sign changes of the normalized slope with
   noticeable curvature), shifting adjacent base knots;
3. add the end of any initial fast dynamics (high curvature).

`traces` is `T × L`. Returns normalized knots (including 0 and 1) per state.
"""
function place_knots(t::AbstractVector, traces::AbstractMatrix)
    t = collect(float.(t))
    T, L = size(traces)
    length(t) == T || throw(DimensionMismatch("length(t) = $(length(t)) but traces has $T rows"))
    T ≥ 5 || throw(ArgumentError("automatic knot placement needs at least 5 time points"))

    # three equidistant base knots at the closest data points
    base = falses(T - 2, L)
    targets = matlab_linspace(t[1], t[end], 5)
    base_ind = sort(unique([argmin(abs.(t .- τ)) for τ in targets]))
    for i in base_ind[2:end-1]
        1 ≤ i - 1 ≤ T - 2 && (base[i-1, :] .= true)
    end

    # finite-difference slopes and curvatures on the (possibly non-uniform) grid
    d21 = t[2:end-1] .- t[1:end-2]
    d31 = t[3:end] .- t[1:end-2]
    d32 = t[3:end] .- t[2:end-1]
    y1, y2, y3 = traces[1:end-2, :], traces[2:end-1, :], traces[3:end, :]
    dy = (y3 .- y1) ./ (d32 .+ d21)
    ddy = 2 .* (y1 ./ d21 ./ d31 .- y2 ./ d32 ./ d21 .+ y3 ./ d32 ./ d31)
    zdy = dy ./ permutedims([_std(c) for c in eachcol(dy)])
    zddy = ddy ./ permutedims([_std(c) for c in eachcol(ddy)])

    # peaks and troughs: sign changes of the slope, moved to the smaller |slope|
    crossings = vcat(abs.(diff(sign.(zdy); dims = 1)) .== 2, falses(1, L))
    for state in 1:L, idx in 1:size(crossings, 1)-1                  # sequential, as in MATLAB
        if crossings[idx, state]
            sub = abs.(zdy[idx:idx+1, state])
            crossings[idx:idx+1, state] .= sub .== minimum(sub)
        end
    end
    peaks_troughs = crossings .& (abs.(zddy) .> 0.25)

    base_pt = copy(base)
    for state in 1:L, idx in 1:size(peaks_troughs, 1)
        peaks_troughs[idx, state] || continue
        _insert_knot!(base_pt, base, idx, state, T)
    end

    # end of initial fast dynamics
    curvature = abs.(zddy) .> 2 .* permutedims([_std(c) for c in eachcol(zddy)])
    base_curv = copy(base_pt)
    for state in 1:L
        pos = findall(curvature[:, state])
        isempty(pos) && continue
        n_fast = findfirst(>(2), diff(vcat(pos, T + 2)))
        _insert_knot!(base_curv, base_pt, pos[n_fast], state, T)
    end

    subset = trues(T, L)
    subset[2:end-1, :] = base_curv
    return [(t[subset[:, state]] .- t[1]) ./ (t[end] - t[1]) for state in 1:L]
end

# Add a knot at interior row idx, absorbing adjacent knots of `reference` (MATLAB logic).
function _insert_knot!(target, reference, idx, state, T)
    if idx > 1 && idx < T - 2 && reference[idx-1, state] && reference[idx+1, state]
        target[idx-1:idx+1, state] = [false, true, false]
    elseif idx > 1 && reference[idx-1, state]
        target[idx-1:idx, state] = [false, true]
    elseif idx < T - 2 && reference[idx+1, state]
        target[idx:idx+1, state] = [true, false]
    else
        target[idx, state] = true
    end
    return target
end

"""
    resolve_knots(t, traces; knots = nothing, autoknots = true)

Knots per observed state, normalized to `[0, 1]`, following the EGM
constructor:
- `knots` given: normalized with [`normalize_knots`](@ref). Pass either one
  vector (shared by all states) or one vector per state.
- otherwise, if `autoknots`: [`place_knots`](@ref);
- otherwise: [`default_knots`](@ref).
"""
function resolve_knots(t, traces::AbstractMatrix; knots = nothing, autoknots = true)
    L = size(traces, 2)
    if knots !== nothing
        per_state = eltype(knots) <: Real ? fill(knots, L) : collect(knots)
        length(per_state) == L || throw(ArgumentError("need one knot vector per observed state ($L)"))
        return [normalize_knots(k, t) for k in per_state]
    end
    autoknots && return place_knots(t, traces)
    return fill(default_knots(t), L)
end

# --- Basis matrices on the three time grids ----------------------------------------

"""
    SplineBases(t, t_gm, knots; order = 4, t_fine = matlab_linspace(t[1], t[end], 81))

B-spline bases for each observed state, evaluated on the data grid `t`, the
inference grid `t_gm` and a fine plotting grid. Times are scaled to `[0, 1]`
with `(t − t[1]) / (t[end] − t[1])`; derivatives are with respect to time.
Fields hold one matrix per state: `B`, `dB` (data grid), `B_gm`, `dB_gm`,
`B_fine`, `dB_fine`.
"""
struct SplineBases
    order::Int
    knots::Vector{Vector{Float64}}
    splines::Vector{BSpline}
    t::Vector{Float64}
    t_gm::Vector{Float64}
    t_fine::Vector{Float64}
    scaled::Vector{Float64}
    scaled_gm::Vector{Float64}
    scaled_fine::Vector{Float64}
    B::Vector{Matrix{Float64}}
    dB::Vector{Matrix{Float64}}
    B_gm::Vector{Matrix{Float64}}
    dB_gm::Vector{Matrix{Float64}}
    B_fine::Vector{Matrix{Float64}}
    dB_fine::Vector{Matrix{Float64}}
end

function SplineBases(t, t_gm, knots; order = 4, t_fine = nothing)
    t = collect(float.(vec(t)))
    t_gm = collect(float.(vec(t_gm)))
    t_fine = t_fine === nothing ? matlab_linspace(t[1], t[end], 81) : collect(float.(vec(t_fine)))
    Δ = t[end] - t[1]
    scale(τ) = (τ .- t[1]) ./ Δ
    s, s_gm, s_fine = scale(t), scale(t_gm), scale(t_fine)
    knots = [collect(float.(vec(k))) for k in knots]
    splines = [BSpline(order, k) for k in knots]
    return SplineBases(order, knots, splines, t, t_gm, t_fine, s, s_gm, s_fine,
                       [basis(b, s) for b in splines], [basis_diff(b, s) ./ Δ for b in splines],
                       [basis(b, s_gm) for b in splines], [basis_diff(b, s_gm) ./ Δ for b in splines],
                       [basis(b, s_fine) for b in splines], [basis_diff(b, s_fine) ./ Δ for b in splines])
end

Base.show(io::IO, sb::SplineBases) =
    print(io, "SplineBases(order ", sb.order, ", cards ", [b.card for b in sb.splines], ")")
