function root = dt_train(X,Y,depth_limit)
% DT_TRAIN - Trains a binary decision tree classifier.
%
% Usage:
%
%    tree = dt_train(X, Y, DEPTH_LIMIT)
%
% Returns a tree of maximum depth DEPTH_LIMIT learned using the ID3
% algorithm. Assumes X is a N x D matrix of N examples with D features. Y
% must be a N x 1 {0,1} vector of labels. 
%
% Returns a linked hierarchy of structs with the following fields:
%
%   node.terminal - whether or not this node is a terminal (leaf) node
%   node.fidx, node.fval - the feature based test (is X(fidx) <= fval?)
%                          associated with this node
%   node.value - P(Y=1) as predicted by this node
%   node.left - child node struct on left branch (f <= val)
%   node.right - child node struct on right branch (f > val)
% 
% SEE ALSO
%   DT_VALUE, DT_CHOOSE_FEATURE

% Y must be 1's and 0's.
assert(isequal(unique(Y), [0; 1]), 'Y must be 1''s and 0''s.');

% Pre-compute the range of each feature.
for i = 1:size(X, 2)
    Xrange{i} = unique(X(:,i));
end

% Recursively split data to build tree.
root = split_node(X, Y, Xrange, mean(Y), 1:size(Xrange, 2), 0, depth_limit);

function [node] = split_node(X, Y, Xrange, default_value, colidx, depth, depth_limit)
% Utility function called recursively; returns a node structure.
%    
%  [node] = split_node(X, Y, Xrange, default_value, colidx, depth, depth_limit)
%  
%  inputs: 
%    Xrange - cell array containing the range of values for each feature
%    default_value - the default value of the node if Y is empty
%    colidx - the indices of features (columns) under consideration
%    depth - current depth of the tree
%    depth_limit - maximum depth of the tree

% The various cases at which we will return a terminal (leaf) node:
%    - we are at the maximum depth
%    - we have Y equal to all 0's or all 1's 
%    - we have only a single (or no) examples left
%    - we have no features left to split on
if depth == depth_limit || all(Y==0) || all(Y==1) || numel(Y) <= 1 || numel(colidx) == 0
    node.terminal = true;
    node.fidx = [];
    node.fval = [];
    if numel(Y) == 0
        node.value = default_value;
    else
        node.value = mean(Y);
    end
    node.left = []; node.right = [];

    fprintf('depth %d [%d/%d]: Leaf node: = %s\n', depth, sum(Y==0), sum(Y==1), ...
        mat2str(node.value));
    return;
end

node.terminal = false;

% Choose a feature to split on using information gain.
[node.fidx node.fval max_ig] = dt_choose_feature(X, Y, Xrange, colidx);

% Remove this feature from future consideration.
colidx(colidx==node.fidx) = [];

% Split the data based on this feature.
leftidx = find(X(:,node.fidx)<=node.fval);
rightidx = find(X(:,node.fidx)>node.fval);

% Store the value of this node in case we later wish to use this node as a
% terminal node (i.e. pruning.)
node.value = mean(Y);

fprintf('depth %d [%d]: Split on feature %d <= %.2f w/ IG = %.2g (L/R = %d/%d)\n', ...
    depth, numel(Y), node.fidx, node.fval, max_ig, numel(leftidx), numel(rightidx));

% Recursively generate left and right branches.
node.left = split_node(X(leftidx, :), Y(leftidx), Xrange, node.value, colidx, depth+1, depth_limit);
node.right = split_node(X(rightidx, :), Y(rightidx), Xrange, node.value, colidx, depth+1, depth_limit);


