# IQM Tools model files (the format used by EGM-v2 and GMGTS).
#
# Supported sections: MODEL NAME, MODEL STATES (d/dt(x) = … and x(0) = …), MODEL
# PARAMETERS, MODEL VARIABLES, MODEL REACTIONS (both evaluated in order, as named
# intermediate quantities) and MODEL FUNCTIONS. MODEL EVENTS must be empty. Text
# after % and annotations in {…} are ignored. Statements may continue over several
# lines while brackets are open.
#
# Expressions use MATLAB syntax, which is translated to Julia (.* ./ .^ ~=).
# `time` refers to the current time. `interp1IQM(xs, ys, time)` and
# `interp1(xs, ys, time)` with literal vectors become `LinearInterpolant`s, and
# their breakpoints become the model's `tstops`.

const _IQM_SECTION = r"^\*+\s*MODEL\s+(.+?)\s*$"i

function _iqm_sections(text::AbstractString)
    sections = Dict{String,Vector{String}}()
    current = ""
    for raw in split(replace(text, '\ufeff' => "", "\r" => ""), '\n')
        mt = match(_IQM_SECTION, strip(raw))
        if mt !== nothing
            current = uppercase(strip(mt[1]))
            sections[current] = String[]
        elseif !isempty(current)
            push!(sections[current], String(raw))
        end
    end
    return sections
end

# Join lines into statements (brackets balanced), dropping comments and annotations.
function _iqm_statements(lines)
    stmts = String[]
    buf, depth = "", 0
    for raw in lines
        line = replace(first(split(raw, '%'; limit = 2)), r"\{[^}]*\}" => "")
        isempty(strip(line)) && isempty(buf) && continue
        buf = isempty(buf) ? strip(line) : buf * " " * strip(line)
        depth = count(in("(["), buf) - count(in(")]"), buf)
        if depth ≤ 0 && !isempty(strip(buf))
            push!(stmts, strip(buf))
            buf = ""
        end
    end
    isempty(strip(buf)) || push!(stmts, strip(buf))
    return stmts
end

function _iqm_expr(s::AbstractString)
    js = replace(s, ".^" => "^", ".*" => "*", "./" => "/", "~=" => "!=")
    ex = try
        Meta.parse(js)
    catch err
        throw(ArgumentError("cannot parse expression `$s`: $(sprint(showerror, err))"))
    end
    ex isa Expr && ex.head === :incomplete && throw(ArgumentError("incomplete expression `$s`"))
    return ex
end

_iqm_number(s) = Float64(Core.eval(Module(), _iqm_expr(s)))

# Replace interpolation calls with literal vectors by constant LinearInterpolants.
function _iqm_interpolants!(ex, interps)
    ex isa Expr || return ex
    args = [_iqm_interpolants!(a, interps) for a in ex.args]
    if ex.head === :call && args[1] in (:interp1IQM, :interp1) && length(args) == 4 &&
       all(a -> a isa Expr && a.head in (:vect, :hcat), args[2:3])
        xs = [_iqm_number(string(v)) for v in args[2].args]
        ys = [_iqm_number(string(v)) for v in args[3].args]
        push!(interps, LinearInterpolant(xs, ys))
        return Expr(:call, Symbol("__interp", length(interps)), args[4])
    end
    return Expr(ex.head, args...)
end

"""
    load_iqm(path; fixed = String[], fixed_values = nothing, M = I, check = true) -> ODEModel
    parse_iqm(text; ...)

Build an [`ODEModel`](@ref) from an IQM Tools model file, as EGM-v2's `System`:
- **Free parameters**: all parameters except `fixed`, in file order. Their
  nominal values become `β0`.
- **Fixed parameters**: kept at their nominal values, or at `fixed_values` if
  given.
- **Initial conditions**: from the `x(0) = …` lines.
- **`M`**: passed on as the process-noise mixing matrix.
- **Check**: the model must be affine in the free parameters; this is checked
  unless `check = false`.

The generated right-hand side is compiled at load time and called through
`Base.invokelatest`. This is slightly slower than a hand-written Julia function;
for large studies, write the model in Julia (the parsed model's states and
parameters show the order to use).
"""
load_iqm(path::AbstractString; kwargs...) = parse_iqm(read(path, String); kwargs...)

function parse_iqm(text::AbstractString; fixed = String[], fixed_values = nothing, M = I, check = true)
    sec = _iqm_sections(text)
    haskey(sec, "STATES") || throw(ArgumentError("no MODEL STATES section"))
    names = filter(!isempty, strip.(get(sec, "NAME", String[])))
    name = isempty(names) ? "" : String(first(names))
    isempty(_iqm_statements(get(sec, "EVENTS", String[]))) ||
        throw(ArgumentError("MODEL EVENTS are not supported"))

    states, rhs, ics = String[], String[], Dict{String,String}()
    for st in _iqm_statements(sec["STATES"])
        if (mt = match(r"^d/dt\(\s*(\w+)\s*\)\s*=\s*(.+)$", st)) !== nothing
            push!(states, mt[1]); push!(rhs, mt[2])
        elseif (mt = match(r"^(\w+)\(\s*0\s*\)\s*=\s*(.+)$", st)) !== nothing
            ics[mt[1]] = mt[2]
        else
            throw(ArgumentError("unrecognized state statement `$st`"))
        end
    end
    x0 = [haskey(ics, s) ? _iqm_number(ics[s]) : 0.0 for s in states]

    params, values = String[], Float64[]
    for st in _iqm_statements(get(sec, "PARAMETERS", String[]))
        mt = match(r"^(\w+)\s*=\s*(.+)$", st)
        mt === nothing && throw(ArgumentError("unrecognized parameter statement `$st`"))
        push!(params, mt[1]); push!(values, _iqm_number(mt[2]))
    end
    fixed = String.(collect(fixed))
    all(in(params), fixed) || throw(ArgumentError("unknown fixed parameters: $(setdiff(fixed, params))"))
    if fixed_values !== nothing
        length(fixed_values) == length(fixed) || throw(ArgumentError("fixed and fixed_values differ in length"))
        for (n, v) in zip(fixed, fixed_values)
            values[findfirst(==(n), params)] = v
        end
    end
    free = [p for p in params if !(p in fixed)]

    assigns = Pair{String,String}[]                                    # variables, then reactions
    for key in ("VARIABLES", "REACTIONS"), st in _iqm_statements(get(sec, key, String[]))
        mt = match(r"^(\w+)\s*=\s*(.+)$", st)
        mt === nothing && throw(ArgumentError("unrecognized $(lowercase(key)) statement `$st`"))
        push!(assigns, mt[1] => mt[2])
    end

    mod = Module(:IQMModel)
    interps = LinearInterpolant[]
    for st in _iqm_statements(get(sec, "FUNCTIONS", String[]))
        mt = match(r"^(\w+)\s*\(([^)]*)\)\s*=\s*(.+)$", st)
        mt === nothing && throw(ArgumentError("unrecognized function statement `$st`"))
        fargs = Symbol.(strip.(filter(!isempty, split(mt[2], ','))))
        body = _iqm_interpolants!(_iqm_expr(mt[3]), interps)
        Core.eval(mod, Expr(:(=), Expr(:call, Symbol(mt[1]), fargs...), body))
    end

    body = Expr[]
    for (i, s) in enumerate(states)
        push!(body, :($(Symbol(s)) = __x[$i]))
    end
    for (j, p) in enumerate(free)
        push!(body, :($(Symbol(p)) = __β[$j]))
    end
    for (p, v) in zip(params, values)
        p in fixed && push!(body, :($(Symbol(p)) = $v))
    end
    push!(body, :(time = __t))
    for (n, e) in assigns
        push!(body, :($(Symbol(n)) = $(_iqm_interpolants!(_iqm_expr(e), interps))))
    end
    rhs_exprs = [_iqm_interpolants!(_iqm_expr(e), interps) for e in rhs]
    push!(body, :(return [$(rhs_exprs...)]))
    for (i, u) in enumerate(interps)
        Core.eval(mod, :(const $(Symbol("__interp", i)) = $u))
    end
    fgen = Core.eval(mod, :(function __rhs(__x, __β, __t)
                                $(body...)
                            end))
    f = (x, β, t) -> Base.invokelatest(fgen, x, β, t)
    tstops = isempty(interps) ? Float64[] : sort(unique(reduce(vcat, breakpoints.(interps))))
    β0 = [values[findfirst(==(p), params)] for p in free]
    return ODEModel(f; states = states, parameters = free, x0 = x0, β0 = β0, M = M, tstops = tstops,
                    name = name, check = check)
end
