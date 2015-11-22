function [fidx val max_ig] = dt_choose_feature_multi(X, Z, Xrange, colidx)
% DT_CHOOSE_FEATURE_MULTI - Selects feature with maximum multi-class IG.
%
% Usage:
%
%   [FIDX FVAL MAX_IG] = dt_choose_feature(X, Z, XRANGE, COLIDX)
%
% Given N x D data X and N x K indicator labels Z, where X(:,j) can take on values in XRANGE{j}, chooses
% the split X(:,FIDX) <= VAL to maximize information gain MAX_IG. I.e., FIDX is
% the index (chosen from COLIDX) of the feature to split on with value
% FVAL. MAX_IG is the corresponding information gain of the feature split.
%
% Note: The relationship between Y and Z is that Y(i) = find(Z(i,:)).
% Z is the categorical representation of Y: Z(i,:) is a vector of all zeros
% except for a one in the Y(i)'th column.
%
% Hint: It is easier to compute entropy, etc. when using Z instead of Y.
%
% SEE ALSO
%    DT_TRAIN_MULTI

% YOUR CODE GOES HERE

% Get entropy of the Y distribution.
Sums_Ys=sum(Z);
totalnum_Ys=size(Z,1);
Prob_Ys=Sums_Ys/totalnum_Ys;
H = multi_entropy(Prob_Ys);

% Compute conditional entropy for each feature.
ig = zeros(numel(Xrange), 1);
split_vals = zeros(numel(Xrange), 1);

% Compute the IG of the best split with each feature. This is vectorized
% so that, for each feature, we compute the best split without a second for
% loop. Note that if we were guaranteed binary features, we could vectorize
% this entire loop with the same procedure.
t = CTimeleft(numel(colidx));
fprintf('Evaluating features on %d examples: ', numel(Z));
for i = colidx
    t.timeleft();
    
    % Check for constant values.
    if numel(Xrange{i}) == 1
        ig(i) = 0; split_vals(i) = 0;
        continue;
    end
    
    % Compute up to 10 possible splits of the feature.
    r = linspace(double(Xrange{i}(1)), double(Xrange{i}(end)), min(10, numel(Xrange{i})));
    split_f = bsxfun(@le, X(:,i), r(1:end-1));
    
    nx=size(split_f,2); %how many quantized x bins
    ny=size(Z,2); % how many possible ys
    %XY_table=zeros(ny,nx);
    % Compute conditional entropy of all possible splits.
    px = mean(split_f);
    pyi_given_x = zeros(ny,nx);%bsxfun(@and, Y, split_f);
    pyi_given_notx = zeros(ny,nx);%bsxfun(@and, Y, ~split_f);
    
    for yi=1:ny
        yi_given_x=bsxfun(@and, Z(:,yi), split_f);
        yi_given_notx=bsxfun(@and, Z(:,yi), ~split_f);
        pyi_given_x(yi,:)=sum(yi_given_x)./sum(split_f);
        pyi_given_notx(yi,:)=sum(yi_given_notx)./sum(~split_f);
    end
    cond_H = px.*multi_entropy(pyi_given_x) + ...
        (1-px).*multi_entropy(pyi_given_notx);
    
    % Choose split with best IG, and record the value split on.
    [ig(i) best_split] = max(H-cond_H);
    split_vals(i) = r(best_split);
end

% Choose feature with best split.
[max_ig fidx] = max(ig);
val = split_vals(fidx);