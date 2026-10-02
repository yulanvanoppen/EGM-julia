classdef FilterPatched < Filter
%FILTERPATCHED  Filter variant with the noise-mixing matrix M kept out of the
%   covariance propagation (the formulation adopted as the Julia default).
%
%   System.auxiliary starts the variational system at Phi(t_{n-1}) = M
%   (for STAT5, the hard-coded init_Phi), so the integrated matrix is
%
%       Phi(s) = Phi_x(s, t_{n-1}) * M,
%
%   with Phi_x the true propagator. The noise integrand Phi(s)*diag(sigma*x)
%   therefore already equals Phi_x(s)*M*diag(sigma*x), i.e. Q is unchanged.
%   Only the propagator returned for the prediction/smoothing steps changes:
%
%       Phi_n = Phi(t_n) / M = Phi_x(t_n, t_{n-1}).
%
%   This is exact (same integration, no extra solver error), but relies on
%   System.auxiliary starting at M; export_case asserts this.

    properties (SetAccess = private)
        M                                                                   % noise-mixing matrix
    end

    methods
        function obj = FilterPatched(system, data, settings, M)
            obj@Filter(system, data, settings);
            obj.M = M;
        end

        function [Q_it, Phi_it] = cumulative_noise(obj, tout, out)
            [Q_it, Phi_it] = cumulative_noise@Filter(obj, tout, out);
            Phi_it = Phi_it / obj.M;                                        % remove M from the propagator
        end
    end
end
