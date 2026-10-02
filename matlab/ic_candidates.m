function conditions = ic_candidates(system, observed, x_init, smoothed_first)
%IC_CANDIDATES  Candidate initial-condition vectors scanned by EGM.init_ICmean.
%   Verbatim copy of the candidate construction in EGM.init_ICmean
%   (EGM-v2 commit 3c83609), including its model-name special cases, so the
%   grid and per-candidate errors can be exported. export_case verifies that
%   the selected candidate matches the one chosen by EGM.estimate, so any
%   divergence between this copy and the original is detected.
%
%   Returns an (ncandidates x K) matrix.

    center = x_init;                                                        % row vector
    center(observed) = smoothed_first;
    multipliers = 4.^(-1:.5:1);
    uncertain = find(~ismember(1:system.K, observed));
    switch string(system.name)
        case "STAT5"
            uncertain = [3 4];
        case "repressilator"
            uncertain = [4 7];
        case "generalizedLV"
            uncertain = 1;
        case "bifunctional_TCS"
            uncertain = 4;
    end

    conditions = repmat(center', [1 length(multipliers) * ones(1, length(uncertain))]);
    switch length(uncertain)
        case 1
            conditions(uncertain, :) = center(uncertain) * multipliers;
            conditions = conditions';
            if string(system.name) == "generalizedLV"
                conditions(:, 2) = conditions(:, 1);
            end
        case 2
            mult = ones(size(conditions));
            mult(uncertain(1), :, :) = repmat(multipliers, 1, 1, length(multipliers));
            mult(uncertain(2), :, :) = repmat(reshape(multipliers, 1, 1, []), 1, length(multipliers), 1);
            conditions = reshape(conditions .* mult, length(center), [])';
            if string(system.name) == "repressilator"
                conditions(:, [5 6]) = repmat(conditions(:, 4), 1, 2);
                conditions(:, [8 9]) = repmat(conditions(:, 7), 1, 2);
            end
    end

    if string(system.name) == "STAT5"
        conditions(:, 4) = min(conditions(:, 3), conditions(:, 4));
    end
end
