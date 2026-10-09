%EXPORT_DATASETS  Export the simulated datasets of accuracy_*.m and test_*.m for Julia.
%
%   Julia cannot reproduce MATLAB's random numbers, so for a case-by-case comparison
%   the Julia benchmark scripts (benchmarks/*.jl) can run on exactly the datasets that
%   the MATLAB scripts generate. This script repeats only the data generation of
%   those scripts, with the same rng calls in the same order, and writes one MAT file
%   per script to ../benchmarks/datasets. No estimation is run, so it takes seconds.
%
%   Requirements: MATLAB with IQM Tools and the EGM-v2 repository (set REPO below).
%
%   Contents of each file (n = number of cases, row r = case r):
%     index     n x 4 [dt_idx noise_idx init_idx seed] (accuracy) or n x 1 seed (test)
%     settings  n x 3 [dt noise init_error] (accuracy only)
%     ic_mult   n x K initial-condition multipliers exp(normrnd(0, init_error)) (accuracy only)
%     t, traces, original, noisevar   n x 1 cells (generator output and ground truth)
%     init      n x K initial conditions used for the simulation

clearvars

REPO = '';                                                                  % <-- path to the EGM-v2 clone
here = fileparts(mfilename('fullpath'));
OUTDIR = fullfile(here, '..', 'benchmarks', 'datasets');

assert(~isempty(REPO) && exist(fullfile(REPO, 'EGM', 'EGM.m'), 'file') == 2, ...
       'Set REPO to the folder containing the EGM-v2 clone.');
addpath(fullfile(REPO, 'EGM'));
if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end
previous_folder = cd(REPO);
restore_folder = onCleanup(@() cd(previous_folder));


%% accuracy_HS.m ----------------------------------------------------------------
system = System('model_HS.txt', 'auxiliary_HS.txt', FixedParameters="c");
export_accuracy('accuracy_HS', system, [.25 .5 1], [.005 .01 .02 .05], 4*[0 .125 .25 .375 .5], 1:50, OUTDIR);


%% accuracy_GLV.m ---------------------------------------------------------------
system = System('model_GLV.txt', 'auxiliary_GLV.txt', FixedParameters="r");
export_accuracy('accuracy_GLV', system, [.25 .5 1], [.005 .02 .05 .1], 4*[0 .125 .25 .375 .5], 1:50, OUTDIR);


%% test_STAT5.m -----------------------------------------------------------------
system = System('model_STAT5.txt', 'auxiliary_STAT5.txt', FixedParameters=["tauinv" "k2"]);
ty = readmatrix('STAT5-data.csv');
generator = Generator(system, N=1, t=ty(:, 1)', error_std=.02, D_mult=eps, observed=1:2);
export_test('test_STAT5', system, generator, 100, OUTDIR);                  % full data; Julia omits the first point


%% test_TCS.m -------------------------------------------------------------------
system = System('model_TCS.txt', 'auxiliary_TCS.txt', FixedParameters=["k2" "k3" "k6" "k8"]);
generator = Generator(system, N=1, t=[0 1 3 5:5:50], error_std=.05, D_mult=eps, observed=[3 5:6]);
export_test('test_TCS', system, generator, 100, OUTDIR);


%% test_repressilator.m -----------------------------------------------------------
system = System('model_repressilator.txt', 'auxiliary_repressilator.txt', ...
                FixedParameters=["DNAT" "kf" "Kd" "m1" "p1"]);
generator = Generator(system, N=1, t=0:5:100, error_std=.05, D_mult=eps, observed=1:3);
export_test('test_repressilator', system, generator, 100, OUTDIR);

clear restore_folder


%% Local functions ------------------------------------------------------------------
function export_accuracy(name, system, dt_values, noise_levels, init_error, seeds, outdir)
    n = numel(dt_values) * numel(noise_levels) * numel(init_error) * numel(seeds);
    index = zeros(n, 4);
    settings = zeros(n, 3);
    ic_mult = zeros(n, system.K);
    init = zeros(n, system.K);
    [t, traces, original, noisevar] = deal(cell(n, 1));
    r = 0;
    for dt_idx = 1:length(dt_values)
    for noise_idx = 1:length(noise_levels)
    for init_idx = 1:length(init_error)
    for seed = seeds
        r = r + 1;
        dt = dt_values(dt_idx);
        noise = noise_levels(noise_idx);
        err = init_error(init_idx);
                                                                            % same calls, same order as accuracy_*.m
        rng(10000*dt_idx + 1000*noise_idx + 100*init_idx + seed - 1)
        generator = Generator(system, N=1, t=0:dt:10, error_std=noise, D_mult=eps, observed=system.K);
        [data, ground_truth] = generator.generate();
        mult = exp(normrnd(0, err, 1, system.K));

        index(r, :) = [dt_idx noise_idx init_idx seed];
        settings(r, :) = [dt noise err];
        ic_mult(r, :) = mult;
        init(r, :) = data.init;
        t{r} = data.t;
        traces{r} = data.traces;
        original{r} = ground_truth.original;
        noisevar{r} = ground_truth.noisevar;
    end
    end
    end
    end
    save(fullfile(outdir, [name '.mat']), 'index', 'settings', 'ic_mult', 'init', 't', 'traces', ...
         'original', 'noisevar', '-v7');
    fprintf('%s: %d datasets\n', name, n);
end


function export_test(name, system, generator, nseeds, outdir)
    index = (1:nseeds)';
    init = zeros(nseeds, system.K);
    [t, traces, original, noisevar] = deal(cell(nseeds, 1));
    for seed = 1:nseeds
        rng(seed)                                                           % same calls as test_*.m
        [data, ground_truth] = generator.generate();
        init(seed, :) = data.init;
        t{seed} = data.t;
        traces{seed} = data.traces;
        original{seed} = ground_truth.original;
        noisevar{seed} = ground_truth.noisevar;
    end
    save(fullfile(outdir, [name '.mat']), 'index', 'init', 't', 'traces', 'original', 'noisevar', '-v7');
    fprintf('%s: %d datasets\n', name, nseeds);
end
