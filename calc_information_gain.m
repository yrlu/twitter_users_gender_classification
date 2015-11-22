% Cited: Copyright (c) 2013, Chao Qu, Gareth Cross, Pimkhuan Hannanta-Anan. All
% rights reserved.
function [IG] = calc_information_gain(Y, X, featidx, t)
% CALC_INFORMATION_GAIN - Calculate the information gain of each feature 
% in X, for a set of multi-class labels in Y. 
%
%   Consider only those features indexed by featidx. Words must show up 
%   more than t times to be counted.
%
%   NOTE: This is the OLD information gain function! Use fastig() instead!
%

labels = unique(Y); % unique labels in Y

N = numel(Y);       % no. of observations
M = numel(labels);  % no. of labels (1,2,3,4,5)
K = size(X,2);      % no. of features

dist_labels = zeros(1,M);   % probabilities of each label

% calculate probabilites of each label
for i=1:M
    dist_labels(i) = sum(Y == labels(i)) / N;
end

% calculate entropy of labels
H = calc_entropy(dist_labels);

% gain of each feature
IG = zeros(numel(featidx),1);
count=0;

% iterate over all features selected
for i=featidx
   count = count + 1;
    
   % observations of this feature
   X_i = X(:,i);
   
   % observations where this feature appears count or more times
   % this should be a N x 1 column vector
   X_i_present = X_i > t;
    
   % probability of seeing this feature in a given observation
   % scalar
   px = mean(X_i_present);
   
   % row for every possible label
   p_y_given_x = zeros(M,1);
   p_y_given_notx = zeros(M,1);
   
   % iterate over all labels
   for j=1:M
      
       % logical row indices of observations where this label occurred
       % N x 1 column vector
       label_indices = (Y == labels(j));
       
       % calculate probability of P(Y | X = x)
       % we use the formula:
       %
       %    P(Y | X = x) = P(Y AND X = x) / P(X = x)
       %
       y_and_x = bsxfun(@and, X_i_present, label_indices);
       y_and_notx = bsxfun(@and, ~X_i_present, label_indices);
       
       % P(Y | X = x)
       % P(Y | X = !x)
       p_y_given_x(j,1) = sum(y_and_x) / sum(X_i_present);
       p_y_given_notx(j,1) = sum(y_and_notx) / sum(~X_i_present);
       
   end
   
   % calculate conditional entropy
   cond_H = px .* calc_entropy(p_y_given_x) + (1-px) .* calc_entropy(p_y_given_notx);
   
   % information gain
   IG(count,1) = H - cond_H;
   
end

% transpose
IG = IG';
end
