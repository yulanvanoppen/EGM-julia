function ref = export_case(name, system, truth, est, beta0, outdir, opts)
%EXPORT_CASE  Run EGM step by step on one case and save all intermediates.
%   ref = EXPORT_CASE(name, system, truth, est, beta0, outdir, opts) saves
%   <outdir>/<name>.mat (MAT v7, readable with MAT.jl). Top-level variables:
%   meta, data, settings, system, spline, smoother, filter, ic, steps,
%   iterations, gm, final, selfcheck. See test/reference/README.md for the
%   array layouts.
%
%   opts (optional struct):
%     M       - noise-mixing matrix = Phi(0) used by System.auxiliary
%               (default eye(K)); asserted against the auxiliary output
%     patched - use FilterPatched (M only in the noise); default false.
%               The self-check against EGM.estimate is skipped when true.

    if nargin < 7, opts = struct(); end
    if ~isfield(opts, 'M'), opts.M = eye(system.K); end
    if ~isfield(opts, 'patched'), opts.patched = false; end

    if opts.patched
        make_filter = @(sys, out, set) FilterPatched(sys, out, set, opts.M);
    else
        make_filter = @(sys, out, set) Filter(sys, out, set);
    end

    fprintf('=== %s ===\n', name);
    R = run_egm_manual(est, beta0, make_filter);
    K = system.K;
    L = est.data.L;

    %% Meta, data, settings ------------------------------------------------
    ref.meta = struct('name', name, 'patched', opts.patched, 'M', opts.M, 'beta0', beta0, ...
                      'egm_commit', '3c83609', 'matlab_version', version, ...
                      'date', char(datetime('now')));

    d = est.data;
    ref.data = struct();
    ref.data.traces = d.traces;
    ref.data.t = d.t;
    ref.data.observed = d.observed;
    ref.data.T = d.T;
    ref.data.L = d.L;
    ref.data.N = d.N;
    ref.data.t_gm = est.t_gm;
    ref.data.t_fine = d.t_fine;
    ref.data.truth_original = truth.original;
    ref.data.truth_beta = truth.beta;

    ref.settings = struct();
    ref.settings.smoothing = est.settings_smoothing;
    ref.settings.filter = est.settings_filter;
    ref.settings.inference = est.settings_inference;

    %% System (Phase 1) ----------------------------------------------------
    s = struct();
    s.name = char(system.name);
    s.K = K;
    s.P = system.P;
    s.states = cellstr(system.states);
    s.parameters = cellstr(system.parameters);
    s.parameters_variable = cellstr(system.parameters_variable);
    s.k0 = system.k0(:)';
    s.x0 = system.x0(:)';
    s.fixed_names = cellstr(system.fixed.names);
    s.fixed_values = double(system.fixed.values(:)');

    X = R.final.xbar';                                                      % test points: final smoothed states
    tt = R.output.t(:)';
    b = R.final.beta;
    s.eval_states = X;
    s.eval_times = tt;
    s.eval_beta = b;
    s.g = system.g(X, tt);
    s.h = system.h(X, tt);
    s.rhs = system.rhs(X, tt, b);
    s.df = system.df(X, tt, b);
    s.dg = system.dg(X, tt);
    s.dh = system.dh(X, tt);

    ic = struct('init', system.x0(:)', 't', d.t);
    s.traj_default = system.integrate(system.k0(:)', ic);                   % reltol 1e-6 (default)
    s.traj_tight = system.integrate(system.k0(:)', ic, d.t, 1e-10);
    s.aux_tight = system.auxiliary(system.k0(:)', system.x0(:)', d.t, 1e-10);
    s.aux_initial_Phi = reshape(s.aux_tight(1, K+1:end), K, K);
    ref.system = s;

    assert(norm(s.aux_initial_Phi - opts.M) < 1e-12, ...
           'opts.M does not match the initial Phi used by System.auxiliary for %s', name);

    %% Splines (Phase 2) and smoother (Phase 3) ----------------------------
    sm = R.sm;
    out = R.output;
    sp = struct();
    sp.order = est.settings_smoothing.order;
    sp.autoknots = est.settings_smoothing.autoknots;
    sp.knots_input = est.knots;                                             % normalized, before auto-placement
    sp.knots = sm.settings.knots;                                           % normalized, as used
    sp.card = cellfun(@(bs) bs.card, sm.bsplines);
    sp.scaled = sm.scaled;
    sp.scaled_gm = sm.scaled_gm;
    sp.scaled_fine = sm.scaled_fine;
    sp.B = sm.B;
    sp.dB = sm.dB;
    sp.B_fine = sm.B_fine;
    sp.dB_fine = sm.dB_fine;
    sp.basis_gm = out.basis_gm;
    sp.dbasis_gm = out.dbasis_gm;
    sp.grid = linspace(0, 1, 101);                                          % direct BSpline evaluation incl. x = 1
    sp.grid_basis = cell(1, L);
    sp.grid_dbasis = cell(1, L);
    for k = 1:L
        bs = BSpline(sp.order, sm.settings.knots{k});
        sp.grid_basis{k} = bs.basis(sp.grid)';
        sp.grid_dbasis{k} = bs.basis_diff(sp.grid)';
    end
    ref.spline = sp;

    sm1 = Smoother(system, est.data, est.settings_smoothing);               % single FWLS step
    st = struct();
    st.variances_init = sm1.variances_sm;
    sm1.update_coefficients(1:L);
    st.delta_step1 = sm1.delta;
    st.noise_start_step1 = zeros(L, 2);                                     % lsqnonneg start of the noise MLE
    for k = 1:L
        predicted = sm1.B{k} * sm1.delta{k};
        design = [ones(numel(predicted), 1) predicted(:).^2];
        response = (predicted(:) - reshape(d.traces(:, k, :), [], 1)).^2;
        c0 = lsqnonneg(design, response)';
        if sum(c0) == 0, c0(1) = mean(response); end
        st.noise_start_step1(k, :) = c0;
    end
    sm1.update_variances(1:L);
    st.sigma2_step1 = sm1.sigma2;
    st.tau2_step1 = sm1.tau2;
    st.variances_sm_step1 = sm1.variances_sm;

    st.delta = sm.delta;                                                    % converged smoother
    st.sigma2 = sm.sigma2;
    st.tau2 = sm.tau2;
    st.variances_sm = out.variances_sm;
    st.variances_gm = out.variances;
    st.smoothed = out.smoothed;
    st.dsmoothed = out.dsmoothed;
    st.smoothed_fine = out.smoothed_fine;
    st.dsmoothed_fine = out.dsmoothed_fine;
    st.lincomb_data = out.lincomb_data;
    st.var_delta = out.var_delta;
    ref.smoother = st;

    %% Filter (Phase 4), GM (Phase 5), iterations (Phase 6) ----------------
    f = struct();
    f.sigma = R.flt.sigma;
    f.P_init = R.flt.P_init;
    f.C = R.flt.C;
    f.x_init_chosen = R.flt.x_init;
    ref.filter = f;

    g = struct();
    g.fdiff = full(R.gm.fdiff);
    g.C = R.gm.C;
    g.observed = R.gm.observed;
    g.hidden = R.gm.hidden;
    ref.gm = g;

    ref.ic = R.ic;
    ref.steps = R.steps;
    ref.iterations = R.iterations;
    ref.final = R.final;

    %% Self-check against EGM.estimate -------------------------------------
    sc = struct('performed', false, 'beta_relerr', NaN, 'history_relerr', NaN, ...
                'icmean_relerr', NaN, 'qp_max_relerr', max(R.iterations.qp_relerr));
    if ~opts.patched
        out_est = est.estimate(beta0, true);
        sc.performed = true;
        sc.beta_relerr = eucl_rel(out_est.beta, R.final.beta);
        sc.icmean_relerr = eucl_rel(est.filter.x_init, R.ic.chosen);
        if isequal(size(est.beta), size(R.iterations.beta))
            sc.history_relerr = eucl_rel(est.beta, R.iterations.beta);
        end
        if ~(sc.beta_relerr < 1e-10 && sc.icmean_relerr < 1e-10)
            warning('export_case:selfcheck', ...
                    '%s: manual replay deviates from EGM.estimate (beta %.2e, IC %.2e)', ...
                    name, sc.beta_relerr, sc.icmean_relerr);
        end
    end
    ref.selfcheck = sc;

    %% Save ------------------------------------------------------------------
    if ~exist(outdir, 'dir'), mkdir(outdir); end
    file = fullfile(outdir, [name '.mat']);
    save(file, '-struct', 'ref', '-v7');
    fprintf('saved %s (%d iterations, self-check beta relerr %.2e)\n', ...
            file, R.iterations.n, sc.beta_relerr);
end
