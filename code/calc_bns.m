% Cited: Copyright (c) 2013, Chao Qu, Gareth Cross, Pimkhuan Hannanta-Anan. All
% rights reserved.
function [ bns, idx, labels ] = calc_bns( X, Y, thresh )
% CALC_BNS - Calculate the binormal separation of feature X given labels Y
%
% Parameters:
%   'X'
%       M x N matrix of training observations.
%   'Y'
%       M x 1 matrix of training labels.
%   'thresh'
%       Optional, BNS threshold to use when generating 'idx'.
%
% Return values:
%   'bns'
%       BNS scores, one for each column of X.
%   'idx'
%       Indices of those scores above 'thresh', if it is set.
%   'labels'
%       Unique labels of Y, sorted.
%
%   See: "BNS Feature Scaling: An Improved Representation over TF-IDF for
%   SVM Text Classification" - George Forman
%
% Default thresh = 0
%
if nargin < 3, thresh = 0; end
labels = unique(Y); % unique labels in Y

N = numel(Y);       % no. of observations
M = numel(labels);  % no. of labels (1,2,3,4,5)
K = size(X,2);      % no. of features

dist_labels = zeros(1,M);   % probabilities of each label

% calculate probabilites of each label
for i=1:M
    dist_labels(i) = sum(Y == labels(i)) / N;
end

% Result is an M x K matrix
bns_levels = zeros(M, K);

% for each possible label in the data
for i=1:M
    
    % indices of this label occurring
    label_indices = (Y == labels(i));
    
    % count of positive and negative instances
    pos = nnz(label_indices);
    neg = N - pos;
    
    % get number of true positive, false positives for each feature
    tp = full(sum(X(label_indices, :) > 0));
    fp = full(sum(X(~label_indices, :) > 0));
   
    tpr = tp / pos;
    fpr = fp / neg;
    
    % constrain
    tpr(tpr < 0.0005) = 0.0005;
    tpr(tpr > 0.9995) = 0.9995;
    fpr(fpr < 0.0005) = 0.0005;
    fpr(fpr > 0.9995) = 0.9995;
    
    bns_levels(i,:) = abs(norminv(tpr,0,1) - norminv(fpr,0,1));
end

bns = sum(bsxfun(@times, bns_levels, dist_labels')); 
idx = bns >= thresh;

end
