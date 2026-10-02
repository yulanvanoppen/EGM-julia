function R = run_egm_manual(est, beta_init, make_filter)
%RUN_EGM_MANUAL  Step-by-step replica of EGM.estimate that records intermediates.
%   R = RUN_EGM_MANUAL(est, beta_init, make_filter) repeats exactly the steps
%   of EGM.estimate (EGM-v2 commit 3c83609) using the public methods of the
%   Smoother, Filter and GM classes, and records every intermediate quantity
%   needed for the Julia reference tests. est is a constructed EGM object
%   (only its data and settings are used; est itself is not modified).
%   make_filter is a handle @(system, output, settings) returning a Filter
%   (or FilterPatched) object.

    system = est.system;
    K = system.K;

    %% Smoothing and subsampling (as in EGM.estimate) ----------------------
    sm = Smoother(system, est.data, est.settings_smoothing);
    output = sm.smooth();

    indices = ismember(est.data.t, est.t_gm);
    output.traces = output.traces(indices, :);
    output.t_data = est.t_gm;
    for k = 1:est.data.L
        output.basis{k} = output.basis{k}(indices, :);
        output.dbasis{k} = output.dbasis{k}(indices, :);
    end

    flt = make_filter(system, output, est.settings_filter);
    gm = GM(system, output, est.settings_inference);

    %% Initial-condition prior mean selection (EGM.init_ICmean) ------------
    conditions = ic_candidates(system, est.data.observed, est.settings_filter.x_init, output.smoothed(1, :));
    ncond = size(conditions, 1);
    ic_errors = zeros(1, ncond);
    ic_betas = zeros(ncond, system.P);
    for c = 1:ncond
        flt.update_ICmean(conditions(c, :));
        filtered = flt.EKS(beta_init);
        filtered.covar = est.augment_covariances(filtered.covar, output.variances_sm);
        [ic_betas(c, :), ic_errors(c)] = gm.estimate(beta_init, filtered);
    end
    [~, cbest] = min(ic_errors(:));
    flt.update_ICmean(conditions(cbest, :));

    R.ic = struct();
    R.ic.x_init_start = est.settings_filter.x_init;
    R.ic.conditions = conditions;
    R.ic.betas = ic_betas;
    R.ic.errors = ic_errors;
    R.ic.chosen_index = cbest;
    R.ic.chosen = conditions(cbest, :);

    %% E-M iterations ------------------------------------------------------
    niter = est.settings_inference.niter;
    tol = est.settings_inference.tol;
    beta = zeros(niter + 1, system.P);
    beta(1, :) = ic_betas(cbest, :);

    C = struct('xbar', {{}}, 'P', {{}}, 'sd', {{}}, 'covar_raw', {{}}, 'covar_aug', {{}}, ...
               'V', {{}}, 'variances_XdX', {{}}, 'smoothed_filtered', {{}}, ...
               'dsmoothed_dfiltered', {{}}, 'design', {{}}, 'response', {{}});
    SS = zeros(1, niter);
    qp_relerr = zeros(1, niter);

    for iter = 1:niter
        filtered = flt.EKS(beta(iter, :));
        covar_raw = filtered.covar;
        filtered.covar = est.augment_covariances(filtered.covar, output.variances_sm);

        if iter == 1                                                        % single prediction steps (Phase 4a)
            R.steps = prediction_steps(flt, beta(iter, :), filtered.xbar, output.t_data, K);
        end

        [beta(iter + 1, :), SS(iter)] = gm.estimate(beta(iter, :), filtered);

        V = gm.covariances(beta(iter, :), filtered);                        % recomputed from the same GM state
        [design, response] = gm_design(system, gm, output.t);
        b_chk = Optimization.QPGLS(design, response, nearestSPD(V), est.settings_inference.lb, ...
                                   est.settings_inference.ub, est.settings_inference.prior);
        qp_relerr(iter) = eucl_rel(beta(iter + 1, :), b_chk);

        C.xbar{end+1} = filtered.xbar;
        C.P{end+1} = filtered.P;
        C.sd{end+1} = filtered.sd;
        C.covar_raw{end+1} = covar_raw;
        C.covar_aug{end+1} = filtered.covar;
        C.V{end+1} = V;
        C.variances_XdX{end+1} = gm.variances_XdX;
        C.smoothed_filtered{end+1} = gm.smoothed_filtered;
        C.dsmoothed_dfiltered{end+1} = gm.dsmoothed_dfiltered;
        C.design{end+1} = design;
        C.response{end+1} = response;

        if iter == niter || eucl_rel(beta(iter, :), beta(iter + 1, :)) < tol
            beta = beta(1:iter + 1, :);
            break
        end
    end

    n = size(beta, 1) - 1;
    R.iterations = struct();
    R.iterations.n = n;
    R.iterations.beta = beta;                                               % (n+1) x P, row 1 = after IC selection
    R.iterations.SS = SS(1:n);
    R.iterations.qp_relerr = qp_relerr(1:n);
    fields = fieldnames(C);
    for f = 1:numel(fields)
        R.iterations.(fields{f}) = stack(C.(fields{f}));
    end

    %% Final fit (as in EGM.estimate) --------------------------------------
    beta_final = beta(end, :);
    data_updated = est.data;
    data_updated.init = filtered.xbar(:, 1)';
    fitted = system.integrate(beta_final, data_updated);
    fitted_fine = system.integrate(beta_final, data_updated, est.data.t_fine);
    dfitted = system.rhs(fitted, est.data.t, beta_final);
    dfitted_fine = system.rhs(fitted_fine, est.data.t_fine, beta_final);
    if est.positive
        fitted = max(1e-12, fitted);
        fitted_fine = max(1e-12, fitted_fine);
    end

    R.final = struct();
    R.final.beta = beta_final;
    R.final.init = data_updated.init;
    R.final.fitted = fitted;
    R.final.dfitted = dfitted;
    R.final.fitted_fine = fitted_fine;
    R.final.dfitted_fine = dfitted_fine;
    R.final.t_fine = est.data.t_fine;
    R.final.xbar = filtered.xbar;

    R.sm = sm;                                                              % objects, for further inspection
    R.output = output;
    R.flt = flt;
    R.gm = gm;
end


function S = prediction_steps(flt, beta, xbar, t, K)                    % One EKF prediction per interval
    T = numel(t);
    S = struct();
    S.beta = beta;
    S.x0 = zeros(K, T-1);
    S.tspan = zeros(2, T-1);
    S.tout = zeros(21, T-1);
    S.out = zeros(21, K*(K+1), T-1);
    S.Q = zeros(K, K, T-1);
    S.Phi = zeros(K, K, T-1);
    for n = 2:T
        x0 = xbar(:, n-1);                                                  % any state works as a test input
        [tout, out] = flt.integrate_augmented(beta, x0, [t(n-1) t(n)]);
        [Qn, Phin] = flt.cumulative_noise(tout, out);
        S.x0(:, n-1) = x0;
        S.tspan(:, n-1) = [t(n-1); t(n)];
        S.tout(:, n-1) = tout(:);
        S.out(:, :, n-1) = out;
        S.Q(:, :, n-1) = Qn;
        S.Phi(:, :, n-1) = Phin;
    end
end


function [design, response] = gm_design(system, gm, t)                  % GLS design as in GM.estimate / GSS.m
    sf = gm.smoothed_filtered;
    dsf = gm.dsmoothed_dfiltered;
    design = system.g(sf(2:end-1, :), t(2:end-1));
    response = dsf(2:end-1, :) - system.h(sf(2:end-1, :), t(2:end-1));
end


function A = stack(c)                                                   % Stack along a new trailing dimension
    A = cat(ndims(c{1}) + 1, c{:});
end
