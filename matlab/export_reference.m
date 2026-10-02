%EXPORT_REFERENCE  Generate golden reference files for the Julia translation.
%
%   Requirements: MATLAB with IQMtools (as for EGM-v2) and a clone of
%   https://github.com/yulanvanoppen/EGM-v2 at commit 3c83609.
%
%   Set REPO below, then run this script from any folder. It writes one .mat
%   file per case to ../test/reference (relative to this script):
%
%     HS_P           Higgins-Selkov, only P observed, uniform grid, auto knots
%     HS_P_sub       as HS_P, data at dt = .25, inference grid dt = .5
%                    (exercises TimePoints subsampling and the variances fix)
%     GLV_X3         generalized Lotka-Volterra, only X3 observed, user knots
%                    (setup of example.m)
%     STAT5          STAT5, non-uniform times, input function, prior, per-state
%                    knots; current MATLAB behaviour (Phi(t_{n-1}) = M)
%     STAT5_patched  same data, M only in the process noise (Julia default)

clearvars

REPO = '';                                                                  % <-- path to the EGM-v2 clone
here = fileparts(mfilename('fullpath'));
OUTDIR = fullfile(here, '..', 'test', 'reference');

assert(~isempty(REPO) && exist(fullfile(REPO, 'EGM', 'EGM.m'), 'file') == 2, ...
       'Set REPO to the folder containing the EGM-v2 clone.');
addpath(here);
addpath(fullfile(REPO, 'EGM'));
previous_folder = cd(REPO);                                                 % model files and MEX files live here
restore_folder = onCleanup(@() cd(previous_folder));


%% Case 1: Higgins-Selkov, P observed ---------------------------------------
system = System('model_HS.txt', 'auxiliary_HS.txt', FixedParameters="c");

rng(1)
generator = Generator(system, N=1, t=0:.5:10, error_std=.02, D_mult=eps, observed=system.K);
[data, truth] = generator.generate();
est = EGM(system, data, InitialConditions=[1.5 .8].*system.x0', Sigma=.01);
export_case('HS_P', system, truth, est, .5*system.k0', OUTDIR);


%% Case 2: Higgins-Selkov, subsampled inference grid ------------------------
rng(2)
generator = Generator(system, N=1, t=0:.25:10, error_std=.02, D_mult=eps, observed=system.K);
[data, truth] = generator.generate();
est = EGM(system, data, InitialConditions=[1.5 .8].*system.x0', Sigma=.01, TimePoints=0:.5:10);
export_case('HS_P_sub', system, truth, est, .5*system.k0', OUTDIR);


%% Case 3: generalized Lotka-Volterra, X3 observed (example.m) ---------------
system = System('model_GLV.txt', 'auxiliary_GLV.txt', FixedParameters="r");

rng(0)
generator = Generator(system, N=1, t=0:.5:10, error_std=.02, error_const=.01, ...
                      D_mult=eps, observed=system.K);
[data, truth] = generator.generate();
est = EGM(system, data, Knots=2.5:1.25:7.5, InitialConditions=2.25*system.x0');
export_case('GLV_X3', system, truth, est, [.5 1], OUTDIR);


%% Cases 4-5: STAT5 (test_STAT5.m setup) -------------------------------------
system = System('model_STAT5.txt', 'auxiliary_STAT5.txt', FixedParameters=["tauinv" "k2"]);

ty = readmatrix('STAT5-data.csv');
rng(1)
generator = Generator(system, N=1, t=ty(:, 1)', error_std=.02, D_mult=eps, observed=1:2);
[data, truth] = generator.generate();
data.t = data.t(2:end);                                                     % omit first data point
data.traces = data.traces(2:end, :);
data.T = data.T - 1;

est = EGM(system, data, Knots={[12 18 30 40], [6 10 14 18 25 40]}, ...
          InitialConditions=[1 .1 .01 .001 0 0 0]+1E-8, Sigma=.001, ...
          LB=.1*system.k0', UB=10*system.k0', ...
          Prior=struct('mean', [1 1 1], 'prec', diag([0 0 0])));

M = eye(system.K);                                                          % Phi(0) hard-coded in System.auxiliary
M(1, [2 3]) = [1 2];
M(2, 3) = 2;

export_case('STAT5_patched', system, truth, est, 2*system.k0', OUTDIR, struct('M', M, 'patched', true));
export_case('STAT5', system, truth, est, 2*system.k0', OUTDIR, struct('M', M, 'patched', false));

clear restore_folder                                                        % return to the starting folder
