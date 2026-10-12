# GLMakie window for the interactive smoothing app (port of smoothing_new.mlapp from
# EGM-v2 / GMGTS). Loaded automatically when both EGM and GLMakie are loaded.
#
# All app behaviour lives in EGM's SmoothingSession functions (src/interactive.jl);
# this file only builds the window, forwards user actions to those functions, and
# redraws. Layout (as in the MATLAB app):
#   left:  State(s) menu, Automatic knot placement, #Knots slider, manual knot selector
#          (left click adds, right click removes the nearest knot), #Cells slider,
#          Smooth button, Live toggle, Finish button (from egm: "Finish and run estimation" and
#          "Finish without estimation")
#   right: one plot per selected state with measurements (solid) and smoothed curves
#          (dashed), coloured per cell with the turbo colour map

module EGMMakieExt

using EGM
using EGM: SmoothingSession, select_states!, click_knot!, equidistant_knots!, automatic_knots!, smooth!,
           finish!, max_knots, displayed_state, interior_knots, interior_knots_time, shown_cells, smoothed_fine
using GLMakie
using GLMakie: Makie

function EGM.smoothing_app(t, traces; state_names = nothing, ncells::Integer = 20, live::Bool = true,
                           estimation::Bool = false, figure_size = (1500, 850), kwargs...)
    sess = SmoothingSession(t, traces; state_names = state_names, ncells = ncells, live = live, kwargs...)
    return run_app(sess; estimation = estimation, figure_size = figure_size)
end

"Colours for `n` cells from the turbo colour map."
cell_colors(n) = n == 1 ? Makie.resample_cmap(:turbo, 2)[1:1] : Makie.resample_cmap(:turbo, n)

"Displayed cell counts for the #Cells slider (roughly logarithmic, as in the MATLAB app)."
cell_counts(N) = sort(unique(min.(vcat(1:10, 20:10:100, 200:100:1000, 2000:1000:10000, N), N)))

# Run an app action without letting an error kill the window's event handling.
function guarded(f)
    try
        f()
    catch err
        @error "smoothing app: action failed" exception = (err, catch_backtrace())
    end
    return nothing
end

function run_app(sess::SmoothingSession; estimation::Bool = false, figure_size = (1500, 850))
    s = sess.smoother
    t, tf = s.t, s.t_fine
    T, L, N = size(s.traces)

    fig = Figure(size = figure_size)
    left = fig[1, 1] = GridLayout(valign = :top, tellheight = false)
    right = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Fixed(340))

    # --- controls -------------------------------------------------------------------------
    Label(left[1, 1], "State(s)"; halign = :right)
    menu = Menu(left[1, 2:3]; options = vcat("All", sess.state_names), default = "All")

    btn_auto = Button(left[2, 1:3]; label = "Automatic knot placement", tellwidth = false)

    Label(left[3, 1], "#Knots"; halign = :right)
    sl_knots = Slider(left[3, 2]; range = 0:max_knots(sess), update_while_dragging = false,
                      startvalue = length(interior_knots(sess, displayed_state(sess))))
    Label(left[3, 3], lift(string, sl_knots.value); width = 30)

    ax_k = Axis(left[4, 1:3]; title = "Manual knot selector", xlabel = "time", height = 230,
                yticksvisible = false, yticklabelsvisible = false)
    for name in (:rectanglezoom, :dragpan, :scrollzoom, :limitreset)            # clicks edit knots instead
        try
            deregister_interaction!(ax_k, name)
        catch
        end
    end
    Label(left[5, 1:3], "left click: add knot   ·   right click: remove nearest"; fontsize = 12, color = :gray40)

    if N > 1
        Label(left[6, 1], "#Cells"; halign = :right)
        sl_cells = Slider(left[6, 2]; range = cell_counts(N), startvalue = sess.ncells,
                          update_while_dragging = false)
        Label(left[6, 3], lift(string, sl_cells.value); width = 30)
    else
        Label(left[6, 1:3], "single trajectory"; fontsize = 12, color = :gray40)
    end

    controls = left[7, 1:3] = GridLayout()
    btn_smooth = Button(controls[1, 1]; label = "Smooth")
    tog_live = Toggle(controls[1, 2]; active = sess.live)
    Label(controls[1, 3], "Live")
    if estimation                                                         # called from egm: continue or stop
        btn_finish = Button(left[8, 1:3]; label = "Finish and run estimation", tellwidth = false)
        btn_stop = Button(left[9, 1:3]; label = "Finish without estimation", tellwidth = false)
    else
        btn_finish = Button(left[8, 1:3]; label = "Finish", tellwidth = false)
    end
    rowgap!(left, 14)

    # --- drawing --------------------------------------------------------------------------
    data_axes = Axis[]
    smoothed_plots = Any[]
    syncing = Ref(false)                                                  # ignore slider events we cause

    function draw_knot_selector()
        empty!(ax_k)
        k = displayed_state(sess)
        B = s.bases.B_fine[k]
        for j in axes(B, 2)
            lines!(ax_k, tf, B[:, j]; color = Cycled(j))
        end
        kt = interior_knots_time(sess, k)
        isempty(kt) || vlines!(ax_k, kt; color = :black, linestyle = :dash)
        xlims!(ax_k, t[1], t[end])
        ylims!(ax_k, 0, 1)
    end

    function sync_knots_slider()
        syncing[] = true
        try
            set_close_to!(sl_knots, length(interior_knots(sess, displayed_state(sess))))
        finally
            syncing[] = false
        end
    end

    function setup_plots()
        foreach(delete!, data_axes)
        empty!(data_axes)
        empty!(smoothed_plots)
        trim!(right)
        active = sess.active
        ncols = length(active) == 1 ? 1 : ceil(Int, sqrt(L))
        cells = shown_cells(sess)
        cols = cell_colors(length(cells))
        for (n, k) in enumerate(active)
            r, c = divrem(n - 1, ncols) .+ 1                               # fill row by row
            ax = Axis(right[r, c]; title = sess.state_names[k], xlabel = "time", ylabel = "value")
            for (j, i) in enumerate(cells)
                lines!(ax, t, s.traces[:, k, i]; color = cols[j])
            end
            push!(data_axes, ax)
        end
    end

    function draw_smoothed()
        for (ax, p) in smoothed_plots
            delete!(ax, p)
        end
        empty!(smoothed_plots)
        cells = shown_cells(sess)
        cols = cell_colors(length(cells))
        for (n, k) in enumerate(sess.active)
            k in sess.stale && continue                                   # not smoothed with its current knots
            S = smoothed_fine(s, k)
            for (j, i) in enumerate(cells)
                push!(smoothed_plots, (data_axes[n], lines!(data_axes[n], tf, S[:, i]; color = cols[j],
                                                              linestyle = :dash, linewidth = 1.8)))
            end
        end
    end

    function after_knot_change()
        sync_knots_slider()
        draw_knot_selector()
        sess.live && draw_smoothed()
    end

    function update_smooth_button()
        btn_smooth.labelcolor[] = sess.live ? :gray60 : :black
    end

    # --- events ---------------------------------------------------------------------------
    on(menu.selection) do sel
        sel === nothing && return
        guarded() do
            select_states!(sess, sel == "All" ? :all : sel)
            setup_plots()
            draw_smoothed()
            sync_knots_slider()
            draw_knot_selector()
        end
    end

    on(btn_auto.clicks) do _
        guarded(() -> (automatic_knots!(sess); after_knot_change()))
    end

    on(sl_knots.value) do n
        syncing[] && return
        guarded(() -> (equidistant_knots!(sess, n); after_knot_change()))
    end

    on(events(fig).mousebutton; priority = 2) do ev
        (ev.action == Mouse.press && is_mouseinside(ax_k)) || return Consume(false)
        menu.is_open[] && return Consume(false)                           # click meant for the open menu
        ev.button in (Mouse.left, Mouse.right) || return Consume(false)
        x = mouseposition(ax_k)[1]
        guarded(() -> (click_knot!(sess, x; remove = ev.button == Mouse.right); after_knot_change()))
        return Consume(true)
    end

    if N > 1
        on(sl_cells.value) do n
            guarded() do
                sess.ncells = n
                setup_plots()
                draw_smoothed()
            end
        end
    end

    on(btn_smooth.clicks) do _
        guarded(() -> (smooth!(sess); draw_smoothed()))
    end

    on(tog_live.active) do live
        guarded() do
            sess.live = live
            update_smooth_button()
            if live
                smooth!(sess)
                draw_smoothed()
            end
        end
    end

    # --- run until Finish or window closed --------------------------------------------------
    draw_knot_selector()
    setup_plots()
    draw_smoothed()
    update_smooth_button()

    done = Base.Condition()
    action = Ref(:closed)                                                 # unless a button is pressed
    on(btn_finish.clicks) do _
        action[] = estimation ? :estimate : :finish
        notify(done)
    end
    if estimation
        on(btn_stop.clicks) do _
            action[] = :finish
            notify(done)
        end
    end
    on(events(fig).window_open) do open
        open || notify(done)
    end
    screen = display(GLMakie.Screen(), fig)                              # own window, also inside VS Code
    wait(done)
    result = finish!(sess)
    try
        close(screen)
    catch
    end
    return merge(result, (; action = action[]))
end

end # module
