function [nfoldErr, testErr] = compareKNN(X, Y) 
%
% randomly split the full data into training and testing sets. 
% For all experiments, use 400 examples for training and the
% remainder for testing.
[l,~] = size(X);
index = randperm(l); 
trainIndex = index(1:400);
testIndex = index(401:end);
[~,lTest] = size(testIndex);

% hold Errs for these folds
nfoldErr = zeros(1,4);
testErr = zeros(1,4);

% For both the original data and the noisy, compute both the N-fold error 
% on the training set and the test error for K-NN with K = 1, for N = {2, 4, 8, 16}.
for i = 1:4
    nfoldErr(1,i) = knn_xval_error(1, X(trainIndex,:), Y(trainIndex,:),make_xval_partition(400,2^i));
    testLabels = sign(knn_test(1, X(trainIndex,:), Y(trainIndex,:), X(testIndex,:)));
    testErr(1,i) = sum(testLabels~= Y(testIndex,:))/lTest;
end
% 